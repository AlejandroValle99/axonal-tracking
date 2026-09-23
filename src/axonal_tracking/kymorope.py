"""KymoRoPE: modelo por-pixel a resolucion nativa (`plan/kymorope-guide.md` SS2).

    kymografo (T, L) nativo
      -> stem convolucional SOLAPADO (stride 16, guarda skips s2/s4)
      -> encoder transformer con RoPE 2D en UNIDADES FISICAS (segundos, um)
      -> decoder FPN a resolucion completa
      -> 3 heads: trackness (3 clases) / embedding (d=8) / orientacion (sin, cos)

Tres decisiones que no son cosmeticas, cada una con su medicion (2026-09-22):

1. **Stem solapado.** Las trazas miden 0.8-2.7 px de ancho (mediana 1.6). Un
   patchify NO solapado corta una diagonal de 1.6 px en cada borde de patch que
   cruza (~uno cada 11 filas a pendientes tipicas). La convolucion solapada
   (estilo SegFormer) mantiene la linea continua dentro del token.
2. **RoPE en unidades fisicas.** Las frecuencias se calculan sobre `fila*dt` en
   segundos y `columna*dx` en um, NO sobre indices. El dataset tiene 5 tasas de
   muestreo (0.193-0.304 s/frame) y las cinco estan en los tres splits, asi que la
   invariancia temporal es testeable en distribucion (SS5.3). El eje espacial es
   portabilidad a otro microscopio, no una invariancia testeada: `pixel_scale_um`
   vale 0.107 en las 1600 muestras.
3. **Sin codificacion posicional aprendida y sin CLS.** La posicion absoluta es
   encuadre del operador: x=0 es donde el investigador empezo a trazar la ROI y
   t=0 es cuando apreto grabar.

**MPS** (el repo desarrolla en Apple Silicon): todo float32 (MPS no soporta
float64), atencion via `F.scaled_dot_product_attention` con mascara BOOL (la
aditiva en fp32 cuesta 4x memoria y en memoria unificada eso es real), rotacion
RoPE por `rotate_half` real en vez de aritmetica compleja, y
`torch.utils.checkpoint` opcional por bloque. Poner
`PYTORCH_ENABLE_MPS_FALLBACK=1` ANTES de `import torch` (mismo patron que
`scripts/entrenar_atencion_v1.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    "D_EMBEDDING",
    "PARCHE",
    "KymoRoPE",
    "RoPE2DFisica",
    "SalidaKymoRoPE",
    "dispositivo_preferido",
    "perdida_discriminativa",
    "perdida_orientacion",
    "perdida_total",
    "perdida_trackness",
    "resumen_mps",
]

PARCHE = 32  # stride total del stem (ver StemSolapado: 32 divide la memoria de atencion por 16)
D_EMBEDDING = 8  # dimension del head de embedding (SS2.5)
N_CLASES_TRACKNESS = 3  # fondo / estatica / movil

# Longitudes de onda extremas de la RoPE, en unidades FISICAS. Elegidas por eje a
# partir del rango real del dataset en vez de compartir una base tipo 10000: el eje
# temporal llega a ~141 s (463 filas x 0.304 s) y el espacial a ~291 um (2722 px x
# 0.107), asi que una base comun deja media banda del espectro sin usar en el eje
# corto. `lambda_min` ~ 2x la resolucion (Nyquist), `lambda_max` ~ 2x la extension
# maxima, de modo que ninguna frecuencia alias ni se queda plana sobre el rango util.
LAMBDA_MIN_T_S, LAMBDA_MAX_T_S = 0.4, 320.0
LAMBDA_MIN_X_UM, LAMBDA_MAX_X_UM = 0.2, 640.0


def dispositivo_preferido() -> torch.device:
    """MPS si esta, si no CUDA, si no CPU. No fuerza nada: si MPS no esta
    disponible el codigo corre igual (mas lento)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resumen_mps() -> dict[str, object]:
    """Estado de MPS + memoria asignada, para los chequeos del notebook."""
    info: dict[str, object] = {
        "disponible": torch.backends.mps.is_available(),
        "compilado": torch.backends.mps.is_built(),
        "torch": torch.__version__,
    }
    if torch.backends.mps.is_available():
        info["asignado_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
        info["driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    return info


# --------------------------------------------------------------------------- #
# SS2.2 -- RoPE 2D axial en unidades fisicas
# --------------------------------------------------------------------------- #
def _rotar_mitad(x: torch.Tensor) -> torch.Tensor:
    """Convencion GPT-NeoX: el par rotatorio del canal `i` es `i + d/2`."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE2DFisica(nn.Module):
    """Tablas `cos`/`sin` de una RoPE axial 2D sobre coordenadas FISICAS.

    La dimension de cabeza se parte al medio: la primera mitad rota con el tiempo
    (segundos), la segunda con la posicion (um). Dentro de cada mitad hay
    `d_cabeza/4` frecuencias, cada una repetida dos veces (una por elemento del par
    rotatorio), espaciadas geometricamente entre `lambda_min` y `lambda_max`.

    No tiene parametros: las frecuencias son fijas y las coordenadas entran por
    `forward`. Eso es lo que hace que el mismo contenido fisico muestreado a otro
    fps produzca la MISMA codificacion -- el test de SS5.3."""

    def __init__(
        self,
        d_cabeza: int,
        *,
        lambda_t: tuple[float, float] = (LAMBDA_MIN_T_S, LAMBDA_MAX_T_S),
        lambda_x: tuple[float, float] = (LAMBDA_MIN_X_UM, LAMBDA_MAX_X_UM),
    ):
        super().__init__()
        if d_cabeza % 4 != 0:
            raise ValueError(f"d_cabeza debe ser multiplo de 4, recibido {d_cabeza}")
        self.d_cabeza = d_cabeza
        n_frec = d_cabeza // 4  # por eje
        self.register_buffer("omega_t", self._omegas(n_frec, *lambda_t), persistent=False)
        self.register_buffer("omega_x", self._omegas(n_frec, *lambda_x), persistent=False)

    @staticmethod
    def _omegas(n: int, lam_min: float, lam_max: float) -> torch.Tensor:
        """`n` frecuencias angulares espaciadas geometricamente (rad por unidad
        fisica): `omega_k = 2*pi / lambda_k`."""
        k = torch.arange(n, dtype=torch.float32)
        lam = lam_min * (lam_max / lam_min) ** (k / max(n - 1, 1))
        return 2.0 * math.pi / lam

    def forward(self, t_seg: torch.Tensor, x_um: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`t_seg`, `x_um`: `(N,)` float32 -- coordenada fisica de cada token.
        Devuelve `cos`, `sin` de forma `(N, d_cabeza)`."""
        ang_t = t_seg[:, None] * self.omega_t[None, :]  # (N, d/4)
        ang_x = x_um[:, None] * self.omega_x[None, :]  # (N, d/4)
        # dentro de cada mitad, la convencion rotate_half exige [f0..fk, f0..fk]
        ang = torch.cat([ang_t, ang_x, ang_t, ang_x], dim=-1)  # (N, d)
        return ang.cos(), ang.sin()


def aplicar_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x`: `(B, H, N, D)`; `cos`/`sin`: `(N, D)`. Real, sin numeros complejos.

    **La rotacion se calcula en float32 y se devuelve en el dtype de `x`**, que es lo
    que hace que esto sea compatible con autocast. Las tablas de la RoPE son float32
    (las frecuencias no se pueden calcular en media precision sin perder resolucion
    angular en las longitudes de onda largas). Si se multiplicaran directamente
    contra un `x` en float16, la promocion de tipos devolveria float32 y q/k
    entrarian a `scaled_dot_product_attention` en float32 con v en float16: no da
    error -- SDPA promociona -- pero la atencion entera corre en float32, o sea AMP
    no acelera la parte que mas pesa. Verificado midiendo los dtypes: sin este
    `.to(x.dtype)`, q/k salian en float32 bajo autocast fp16."""
    if cos.dim() == 2:  # (N, D) -> compartido por todo el lote
        cos, sin = cos[None, None], sin[None, None]
    else:  # (B, N, D) -> una tabla por muestra
        cos, sin = cos[:, None], sin[:, None]
    xf = x.float()
    return (xf * cos + _rotar_mitad(xf) * sin).to(x.dtype)


# --------------------------------------------------------------------------- #
# SS2.1 -- stem solapado
# --------------------------------------------------------------------------- #
@dataclass
class SalidaStem:
    tokens: torch.Tensor  # (Tp*Lp, d)
    grilla: tuple[int, int]  # (Tp, Lp)
    skip_s2: torch.Tensor  # (1, 64, T/2, L/2)
    skip_s4: torch.Tensor  # (1, 128, T/4, L/4)


class StemSolapado(nn.Module):
    """`(1, 1, T, L)` -> tokens a stride `parche` + skips a stride 2 y 4 (SS2.1).

    Todos los kernels son SOLAPADOS (k > s). GroupNorm y no BatchNorm porque el lote
    es de tamano variable y a veces de una sola muestra -- las estadisticas de batch
    no son estables ahi.

    `parche=32` agrega una cuarta convolucion stride 2 sobre la de stride 16. Cuesta
    1.3 M parametros y divide la cantidad de tokens por 4, o sea la memoria de
    atencion por **16** (va como `N^2`) -- ver `AtencionRoPE`. El precio es contexto
    mas grueso: cada token cubre 32x32 px en vez de 16x16. Los skips que alimentan al
    decoder siguen en s2/s4, asi que la resolucion de salida no cambia."""

    def __init__(self, d: int = 384, c1: int = 64, c2: int = 128, parche: int = PARCHE):
        super().__init__()
        if parche not in (16, 32):
            raise ValueError(f"parche debe ser 16 o 32, recibido {parche}")
        self.parche = parche
        self.conv1 = nn.Conv2d(1, c1, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(8, c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, c2)
        self.conv3 = nn.Conv2d(c2, d, kernel_size=7, stride=4, padding=3)
        self.conv4 = (
            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1) if parche == 32 else None
        )
        self.norm3 = nn.LayerNorm(d)

    def forward(self, kymo: torch.Tensor) -> SalidaStem:
        h1 = F.gelu(self.norm1(self.conv1(kymo)))  # s2
        h2 = F.gelu(self.norm2(self.conv2(h1)))  # s4
        h3 = self.conv3(h2)  # s16
        if self.conv4 is not None:
            h3 = self.conv4(F.gelu(h3))  # s32
        _, d, tp, lp = h3.shape
        tokens = h3.flatten(2).transpose(1, 2).reshape(tp * lp, d)
        return SalidaStem(self.norm3(tokens), (tp, lp), h1, h2)


# --------------------------------------------------------------------------- #
# SS2.3 -- encoder
# --------------------------------------------------------------------------- #
class AtencionRoPE(nn.Module):
    """Auto-atencion multi-cabeza con RoPE aplicada a q/k.

    `F.scaled_dot_product_attention` con mascara BOOL (`True` = el par participa);
    la de padding de `KymoRoPE.forward` entra por aca con forma `(B, 1, 1, N_max)`.

    **En MPS no hay camino fusionado**: SDPA materializa la matriz de atencion
    completa, con mascara o sin ella. Medido a `B=3, H=6, N=2726`: 0.57 GB por capa,
    o sea ~6.8 GB en las 12 capas, y es el consumidor dominante de memoria del
    modelo. Escala como `B * N_max^2 * H`, asi que quien manda es el lote mas grande
    y, si hiciera falta bajarlo mas, el lever es `N` (parche del stem), no el ancho
    de canales. Por eso el gradient checkpointing no es opcional a este tamano."""

    def __init__(self, d: int, n_cabezas: int):
        super().__init__()
        if d % n_cabezas != 0:
            raise ValueError(f"d={d} no es divisible por n_cabezas={n_cabezas}")
        self.h = n_cabezas
        self.dh = d // n_cabezas
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.salida = nn.Linear(d, d)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mascara: torch.Tensor | None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q = aplicar_rope(q, cos, sin)
        k = aplicar_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mascara)
        return self.salida(o.transpose(1, 2).reshape(b, n, d))


class BloqueEncoder(nn.Module):
    """Pre-LN estandar: `x + attn(LN(x))`, `x + mlp(LN(x))`."""

    def __init__(self, d: int, n_cabezas: int, mult_mlp: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = AtencionRoPE(d, n_cabezas)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mult_mlp * d), nn.GELU(), nn.Linear(mult_mlp * d, d))

    def forward(self, x, cos, sin, mascara):
        x = x + self.attn(self.ln1(x), cos, sin, mascara)
        return x + self.mlp(self.ln2(x))


# --------------------------------------------------------------------------- #
# SS2.4 -- decoder FPN
# --------------------------------------------------------------------------- #
class DecoderFPN(nn.Module):
    """s16 + skips s4/s2 -> `(1, d_salida, T, L)` a resolucion COMPLETA.

    La precision subpixel sale de aca, no del encoder: el error de posicion de
    DecNet es 0.032 um = 0.30 px, por debajo de cualquier grilla de parches."""

    def __init__(self, d_enc: int = 384, c_s4: int = 128, c_s2: int = 64, d_salida: int = 64):
        super().__init__()
        self.lat_s4 = nn.Conv2d(c_s4, 192, 1)
        self.fuse_s4 = nn.Sequential(nn.Conv2d(192, 192, 3, padding=1), nn.GroupNorm(8, 192), nn.GELU())
        self.red_enc = nn.Conv2d(d_enc, 192, 1)
        self.lat_s2 = nn.Conv2d(c_s2, 96, 1)
        self.red_s4 = nn.Conv2d(192, 96, 1)
        self.fuse_s2 = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU())
        self.salida = nn.Sequential(
            nn.Conv2d(96, d_salida, 3, padding=1), nn.GroupNorm(8, d_salida), nn.GELU()
        )

    @staticmethod
    def _subir_a(x: torch.Tensor, destino: torch.Tensor) -> torch.Tensor:
        """Interpolacion bilineal al tamano exacto del skip. `size=` y no
        `scale_factor=`: con T o L impares el factor 2 no cae en el tamano correcto
        y las formas dejan de calzar."""
        return F.interpolate(x, size=destino.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, enc_s16, skip_s4, skip_s2):
        """Devuelve features a **stride 2**, no a resolucion completa.

        Subir a full ANTES de los heads era el mayor consumidor de memoria del
        modelo: guardaba cuatro tensores full-res de 96/64/64/64 canales, o sea
        ~1.15 GB por Mpx del lote solo en el forward (y el backward los necesita).
        Corriendo los heads a stride 2 y subiendo el LOGIT, lo unico full-res son los
        3+8+2 = 13 canales de salida: ~22x menos. Es lo que hacen SegFormer/DeepLab.
        La precision subpixel no se pierde porque no sale de la grilla del decoder
        sino de `evaluacion.extraer_subpixel`, que calcula un centroide pesado por
        intensidad por fila sobre la mascara full-res y el kymografo."""
        h = self.red_enc(enc_s16)
        h = self.fuse_s4(self._subir_a(h, skip_s4) + self.lat_s4(skip_s4))
        h = self.fuse_s2(self._subir_a(self.red_s4(h), skip_s2) + self.lat_s2(skip_s2))
        return self.salida(h)


# --------------------------------------------------------------------------- #
# SS2.5 -- modelo completo
# --------------------------------------------------------------------------- #
@dataclass
class SalidaKymoRoPE:
    """Salida de UNA muestra, a resolucion completa."""

    trackness: torch.Tensor  # (3, T, L) logits
    embedding: torch.Tensor  # (8, T, L)
    orientacion: torch.Tensor  # (2, T, L), ya L2-normalizado -> (sin, cos)


class KymoRoPE(nn.Module):
    """El modelo. `forward` toma una LISTA de muestras de tamano distinto.

    Empaquetado estilo Pixtral/NaViT: cada muestra pasa por el stem por separado
    (las convoluciones no se pueden batchear con formas distintas), los tokens se
    concatenan en UNA secuencia y la atencion se restringe con una mascara
    bloque-diagonal bool. Con eso no hace falta padding ni resize, que es el punto
    de todo el diseno (SS0)."""

    def __init__(
        self,
        d: int = 384,
        n_capas: int = 12,
        n_cabezas: int = 6,
        d_dec: int = 64,
        d_emb: int = D_EMBEDDING,
        parche: int = PARCHE,
        usar_checkpoint: bool = False,
    ):
        super().__init__()
        self.d = d
        self.parche = parche
        self.usar_checkpoint = usar_checkpoint
        self.stem = StemSolapado(d=d, parche=parche)
        self.rope = RoPE2DFisica(d // n_cabezas)
        self.bloques = nn.ModuleList([BloqueEncoder(d, n_cabezas) for _ in range(n_capas)])
        self.ln_final = nn.LayerNorm(d)
        self.decoder = DecoderFPN(d_enc=d, d_salida=d_dec)
        self.head_trackness = nn.Conv2d(d_dec, N_CLASES_TRACKNESS, 1)
        self.head_embedding = nn.Conv2d(d_dec, d_emb, 1)
        self.head_orientacion = nn.Conv2d(d_dec, 2, 1)

    def _coords_fisicas(
        self, grilla: tuple[int, int], dt: float, dx: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coordenada fisica del CENTRO de cada token. El centro y no la esquina: con
        stride `p` el token `j` cubre `[p*j, p*j+p)` y su centro esta en
        `p*j + (p-1)/2` px."""
        tp, lp = grilla
        p = self.parche
        filas = torch.arange(tp, device=device, dtype=torch.float32) * p + (p - 1) / 2
        cols = torch.arange(lp, device=device, dtype=torch.float32) * p + (p - 1) / 2
        t_seg = (filas * dt).repeat_interleave(lp)
        x_um = (cols * dx).repeat(tp)
        return t_seg, x_um

    def forward(self, muestras: list) -> list[SalidaKymoRoPE]:
        """Lote PADDEADO `(B, N_max, d)`, no una secuencia concatenada.

        Es matematicamente identico a empaquetar todo en una secuencia con mascara
        bloque-diagonal -- las muestras nunca se atienden entre si en ninguno de los
        dos casos -- pero cuesta mucho menos: `scaled_dot_product_attention` calcula
        la matriz completa aunque la mascara despues tape casi todo, asi que
        concatenar B muestras de N tokens cuesta `(B*N)^2` en vez de `B*N^2`, o sea
        B veces de mas. Medido con 7 muestras de 1162 tokens: 1.10 s contra 0.23 s
        por pasada del encoder (4.8x).

        El padding solo entra como CLAVE enmascarada; las filas de query padeadas se
        calculan igual y se descartan al cortar por `longitudes`. Asi ninguna fila de
        la mascara queda toda en False, que es lo que haria NaN en SDPA."""
        device = next(self.parameters()).device
        stems = [self.stem(m.kymo.unsqueeze(0).to(device)) for m in muestras]
        longitudes = [s.tokens.shape[0] for s in stems]
        b, n_max = len(stems), max(longitudes)

        x = stems[0].tokens.new_zeros(b, n_max, self.d)
        cos = stems[0].tokens.new_zeros(b, n_max, self.rope.d_cabeza)
        sin = torch.zeros_like(cos)
        valido = torch.zeros(b, n_max, dtype=torch.bool, device=device)
        for i, (s, m) in enumerate(zip(stems, muestras)):
            n = longitudes[i]
            x[i, :n] = s.tokens
            t_seg, x_um = self._coords_fisicas(s.grilla, m.dt_segundos, m.dx_um, device)
            cos[i, :n], sin[i, :n] = self.rope(t_seg, x_um)
            valido[i, :n] = True

        # (B, 1, 1, N_max): mascara de CLAVE, difunde sobre cabezas y queries
        mascara = valido[:, None, None, :] if b > 1 else None

        for bloque in self.bloques:
            if self.usar_checkpoint and self.training:
                x = checkpoint(bloque, x, cos, sin, mascara, use_reentrant=False)
            else:
                x = bloque(x, cos, sin, mascara)
        x = self.ln_final(x)

        salidas = []
        for i, (s, m) in enumerate(zip(stems, muestras)):
            tokens = x[i, : longitudes[i]]
            tp, lp = s.grilla
            enc = tokens.transpose(0, 1).reshape(1, self.d, tp, lp)
            feats = self.decoder(enc, s.skip_s4, s.skip_s2)  # stride 2
            tamano = tuple(m.kymo.shape[-2:])

            # los heads corren a stride 2 y se sube el LOGIT (ver DecoderFPN.forward).
            # La normalizacion de orientacion va DESPUES de subir: interpolar dos
            # vectores unitarios no da uno unitario.
            subir = partial(F.interpolate, size=tamano, mode="bilinear", align_corners=False)
            salidas.append(
                SalidaKymoRoPE(
                    trackness=subir(self.head_trackness(feats)).squeeze(0),
                    embedding=subir(self.head_embedding(feats)).squeeze(0),
                    orientacion=F.normalize(
                        subir(self.head_orientacion(feats)), dim=1, eps=1e-6
                    ).squeeze(0),
                )
            )
        return salidas


# --------------------------------------------------------------------------- #
# Perdidas (SS2.5)
# --------------------------------------------------------------------------- #
def perdida_trackness(
    logits: torch.Tensor, objetivo: torch.Tensor, *, peso_dice: float = 1.0, eps: float = 1.0
) -> torch.Tensor:
    """CE + Dice suave macro sobre las clases 1 (estatica) y 2 (movil).

    El Dice va solo sobre las clases de traza: el fondo es el 84% de los pixeles y
    meterlo en el Dice lo domina y lo vuelve ciego a las trazas, que es lo unico
    que interesa."""
    # `objetivo` llega en int8 (ver datos_pixel): CE exige long, pero el cast es
    # transitorio y no queda en la cola del DataLoader.
    ce = F.cross_entropy(logits.unsqueeze(0), objetivo.long().unsqueeze(0))
    probas = logits.softmax(dim=0)
    dados = []
    for clase in (1, 2):
        p = probas[clase].reshape(-1)
        g = (objetivo == clase).reshape(-1).float()
        inter = (p * g).sum()
        dados.append(1.0 - (2 * inter + eps) / (p.sum() + g.sum() + eps))
    return ce + peso_dice * torch.stack(dados).mean()


def perdida_discriminativa(
    embedding: torch.Tensor,
    instancias: torch.Tensor,
    mask_pull: torch.Tensor,
    *,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    peso_reg: float = 0.001,
) -> torch.Tensor:
    """Perdida discriminativa de De Brabandere et al. 2017 (`L_var + L_dist + L_reg`).

    `mask_pull` excluye los pixeles reclamados por >=2 particulas (SS3 opcion 1):
    ahi el termino de atraccion tendria targets contradictorios y el efecto seria
    promediar dos identidades, que es exactamente el modo de falla que el head de
    embedding existe para evitar. El termino de REPULSION no los necesita: opera
    sobre las medias de cluster.

    Las medias se calculan con `index_add_` y no con un bucle por `particle_id` --
    son ~190k pixeles de traza en una muestra de 463x2722."""
    d = embedding.shape[0]
    e = embedding.reshape(d, -1).transpose(0, 1)  # (P, d)
    ids = instancias.reshape(-1)
    sel = mask_pull.reshape(-1) & (ids > 0)
    if sel.sum() == 0:
        return embedding.sum() * 0.0

    e_sel, id_sel = e[sel], (ids[sel] - 1).long()  # 0..K-1; solo los px seleccionados
    k = int(id_sel.max().item()) + 1
    sumas = torch.zeros(k, d, device=e.device, dtype=e.dtype).index_add_(0, id_sel, e_sel)
    cuentas = torch.zeros(k, device=e.device, dtype=e.dtype).index_add_(
        0, id_sel, torch.ones_like(id_sel, dtype=e.dtype)
    )
    vivos = cuentas > 0
    medias = sumas[vivos] / cuentas[vivos, None]

    # L_var: atraccion con margen, promediada por instancia y no por pixel (una
    # traza larga no debe pesar mas que una corta)
    remapeo = torch.full((k,), -1, device=e.device, dtype=torch.long)
    remapeo[vivos] = torch.arange(int(vivos.sum()), device=e.device)
    idx = remapeo[id_sel]
    dist = (e_sel - medias[idx]).norm(dim=1)
    hinge = F.relu(dist - delta_v) ** 2
    l_var = (
        torch.zeros(medias.shape[0], device=e.device, dtype=e.dtype)
        .index_add_(0, idx, hinge)
        .div(cuentas[vivos])
        .mean()
    )

    # L_dist: repulsion entre medias
    n_inst = medias.shape[0]
    if n_inst > 1:
        # distancias par a par a mano y no `torch.cdist`: su backward no esta
        # implementado en MPS y cae a CPU (sincroniza el dispositivo en cada paso).
        # K ~ 5 instancias, asi que el costo O(K^2 d) es irrelevante.
        # `clamp_min` antes de la raiz: sqrt'(0) = inf en la diagonal y aunque la
        # diagonal se descarte, autograd la evalua igual y propagaria NaN.
        dif = medias[:, None, :] - medias[None, :, :]
        d_ij = dif.pow(2).sum(-1).clamp_min(1e-12).sqrt()
        fuera = ~torch.eye(n_inst, dtype=torch.bool, device=e.device)
        l_dist = (F.relu(2 * delta_d - d_ij[fuera]) ** 2).mean()
    else:
        l_dist = embedding.sum() * 0.0

    return l_var + l_dist + peso_reg * medias.norm(dim=1).mean()


def perdida_orientacion(
    prediccion: torch.Tensor, objetivo: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """`1 - <pred, obj>` sobre pixeles de traza (SS2.5).

    Coseno y no MSE sobre theta: theta vive en un circulo y el MSE se rompe en el
    envolvimiento. `prediccion` ya viene L2-normalizada del head."""
    if mask.sum() == 0:
        return prediccion.sum() * 0.0
    p = prediccion.permute(1, 2, 0)[mask]
    o = objetivo.permute(1, 2, 0)[mask].float()  # el target viene en float16
    return (1.0 - (p * o).sum(dim=-1)).mean()


def perdida_total(
    salida: SalidaKymoRoPE,
    muestra,
    *,
    lambda_trackness: float = 1.0,
    lambda_embedding: float = 1.0,
    lambda_orientacion: float = 0.5,
) -> dict[str, torch.Tensor]:
    """`L = l1*CE_Dice + l2*discriminativa + l3*coseno`. Devuelve el desglose ademas
    del total -- en el entrenamiento hay que mirar los tres por separado: si el
    embedding colapsa mientras el trackness baja, el total no lo muestra."""
    l_trk = perdida_trackness(salida.trackness, muestra.trackness)
    l_emb = perdida_discriminativa(salida.embedding, muestra.instancias, muestra.mask_pull)
    l_ori = perdida_orientacion(salida.orientacion, muestra.theta, muestra.mask_theta)
    total = lambda_trackness * l_trk + lambda_embedding * l_emb + lambda_orientacion * l_ori
    return {"total": total, "trackness": l_trk, "embedding": l_emb, "orientacion": l_ori}
