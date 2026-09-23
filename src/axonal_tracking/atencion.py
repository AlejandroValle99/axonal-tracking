"""Modulo de atencion sobre segmentos -- Stage 2 (`plan/association-transformer-guide.md`
SS5), la contribucion de la tesis: reemplaza el costo clasico de Stage 1
(`asociacion.costo_asociacion`) por un transformer encoder entrenado desde cero
sobre tokens de segmento, decodificado por el MISMO nucleo de matching
(`asociacion.decodificar_bipartito`/`decodificar_greedy`) que Stage 1 usa -- la
comparacion aisla la funcion de scoring, no el decodificador (SS5.5).

**Terminologia**: esto es un transformer encoder sobre tokens de segmento, linaje
SuperGlue (atencion sobre keypoints) -- NO un Vision Transformer (un ViT tokeniza
parches de imagen). "From scratch" (SS5.1) = inicializacion aleatoria, sin pesos
preentrenados; el diseno (que es un token, el feature set, la codificacion
posicional, el head de enlace) es del autor (SS5.1b) -- lo importado es
`nn.TransformerEncoderLayer`/`nn.Linear`/`nn.GELU`/`BCEWithLogitsLoss`.

**Sin fuga de ground truth (SS9 riesgo 5)**: `construir_features_tokens` no recibe
`AsignacionGT`/`particle_id` en su firma -- es estructuralmente imposible que el GT
llegue al feature builder, no hace falta un assert en runtime que lo verifique.
"""
from __future__ import annotations

import numpy as np
import torch
from kymobutler.config import VISION_MODULE_TILE_SIZE
from kymobutler.vision_module import get_tile
from scipy.spatial import KDTree
from torch import nn

from axonal_tracking.asociacion import CaracteristicasSegmento, Segmento, geom_feats

__all__ = [
    "N_FEATURES",
    "N_GEOM_FEATURES",
    "CodificadorTile",
    "ModeloAtencionV1",
    "ModeloAtencionV2",
    "ModeloAtencionV3",
    "construir_features_tokens",
    "construir_geom_feats_pares",
    "construir_grupos",
    "costos_dustbin",
    "costos_modelo",
    "extraer_tile_segmento",
    "perdida_dustbin",
    "preparar_contexto_tiles",
]

# t_ini/T, t_fin/T, x_ini/L, x_fin/L, pendiente_norm, n_filas/T, int_media_norm,
# int_std_norm, curvatura_norm (SS5.2)
N_FEATURES = 9
N_GEOM_FEATURES = 4  # d_pos, cos_theta, dv, gap (SS5.3, mismas 4 que el costo clasico)


def construir_features_tokens(
    segmentos: list[Segmento],
    caracteristicas: list[CaracteristicasSegmento],
    T: int,
    L: int,
    kymo: np.ndarray,
    *,
    percentil_intensidad: float = 99.5,
) -> np.ndarray:
    """`(N, 9)` -- SS5.2. Normalizacion (SS5.2 pide normalizar a ~escala unitaria
    "usando T y L" pero deja el resto abierto; decidido asi, documentado porque la
    afirmacion de generalizacion entre tamanos de kymografo depende de esto):

    - `t_ini/T, t_fin/T, x_ini/L, x_fin/L`: caen en [0,1] por construccion.
    - `pendiente`: dividida por `L/T` (escala caracteristica de "px por frame" de
      ESTE kymografo) -- la misma velocidad fisica en px/frame pesa distinto segun
      cuan angosto/ancho sea el kymografo; sin esto el feature no generaliza entre
      tamanos.
    - `n_filas/T`: en [0,1].
    - `intensidad_media/std`: divididas por el percentil 99.5 de intensidad de ESTE
      kymografo, no una constante fija tipo 255 -- sesiones/bit-depths distintos
      tienen escalas de brillo distintas (mismo espiritu que
      `etiquetas_deteccion.kymografo_a_rgb_uint8`).
    - `curvatura`: dividida por `L` -- misma unidad de posicion que `x_ini/x_fin`.
    """
    escala_v = (L / T) if T > 0 else 1.0
    percentil_int = float(np.percentile(kymo, percentil_intensidad)) or 1.0
    filas = []
    for seg, car in zip(segmentos, caracteristicas):
        filas.append([
            seg.t_ini / T, seg.t_fin / T,
            seg.x_ini / L, seg.x_fin / L,
            car.pendiente / escala_v,
            car.n_filas / T,
            car.intensidad_media / percentil_int,
            car.intensidad_std / percentil_int,
            car.curvatura / L,
        ])
    return np.array(filas, dtype=np.float32)


def construir_geom_feats_pares(
    segmentos: list[Segmento],
    caracteristicas: list[CaracteristicasSegmento],
    pares: list[tuple[int, int]],
    *,
    max_salto_px: float,
    max_gap_frames: float,
    escala_v: float,
) -> np.ndarray:
    """`(P, 4)` = `(d_pos, cos_theta, dv, gap)` para el head de enlace (SS5.3),
    normalizados: `d_pos/max_salto_px`, `cos_theta` tal cual (ya en [-1,1]),
    `dv/escala_v` (misma normalizacion de velocidad que el token, SS5.2),
    `gap/max_gap_frames`. Sin normalizar, `d_pos` y `gap` (decenas) dominarian
    sobre `cos_theta` en la capa lineal del head -- el mismo problema de escala que
    motivo normalizar el costo clasico (SS4.2, `entrenar_costo_clasico.ajustar_pesos`)."""
    filas = []
    for i, j in pares:
        d_pos, cos_theta, dv, gap = geom_feats(segmentos[i], caracteristicas[i], segmentos[j], caracteristicas[j])
        filas.append([d_pos / max_salto_px, cos_theta, dv / escala_v, gap / max_gap_frames])
    return np.array(filas, dtype=np.float32)


def _pe_sinusoidal(valores_norm: torch.Tensor, d: int, escala: float = 1000.0) -> torch.Tensor:
    """PE sinusoidal estandar (Vaswani et al. 2017) sobre un valor continuo YA
    normalizado a [0,1] -- reescalado por `escala` para ocupar el rango util de las
    frecuencias (sin esto, todo el rango [0,1] cae en la zona plana de baja
    frecuencia de la PE). Guia SS17 / plan SS5.2: `E(x,t) = E_x(x) + E_t(t)`, dos
    bancos SEPARADOS sumados -- no una PE 2D generica de imagen, que trataria
    posicion y tiempo como el mismo eje."""
    pos = valores_norm.unsqueeze(-1) * escala  # (N, 1)
    i = torch.arange(d, device=valores_norm.device, dtype=torch.float32)
    tasas = 1.0 / torch.pow(10000.0, (2 * torch.div(i, 2, rounding_mode="floor")) / d)
    angulos = pos * tasas  # (N, d)
    pe = torch.zeros_like(angulos)
    pe[:, 0::2] = torch.sin(angulos[:, 0::2])
    pe[:, 1::2] = torch.cos(angulos[:, 1::2])
    return pe


class ModeloAtencionV1(nn.Module):
    """Transformer encoder sobre tokens de segmento (SS5.3):

    features (N, 9) -> Linear(9 -> d) -> + PE_t(t_ini) + PE_x(x_ini) ->
    TransformerEncoder (n_capas, n_cabezas, GELU, pre-LN) -> tokens (N, d) ->
    por cada par candidato (i,j): MLP([h_i; h_j; h_i-h_j; geom_feats(i,j)]) -> logit

    `usar_pe=False` es la fila 7 del ablation matrix (SS6, ablacion de la
    codificacion posicional) -- mismos datos, un flag del constructor."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d: int = 128,
        n_capas: int = 4,
        n_cabezas: int = 4,
        n_geom: int = N_GEOM_FEATURES,
        usar_pe: bool = True,
    ):
        super().__init__()
        self.d = d
        self.usar_pe = usar_pe
        self.proyeccion = nn.Linear(n_features, d)
        capa = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_cabezas, dim_feedforward=d * 4,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(capa, num_layers=n_capas)
        self.head_enlace = nn.Sequential(
            nn.Linear(3 * d + n_geom, d), nn.GELU(), nn.Linear(d, 1),
        )

    def contextualizar(
        self, features: torch.Tensor, t_ini_norm: torch.Tensor, x_ini_norm: torch.Tensor
    ) -> torch.Tensor:
        """`(N, F) -> (N, d)`, un kymografo (una "secuencia") a la vez -- N es chico
        (50-200, SS5.3), no hace falta batchear/paddear varios kymografos juntos."""
        h = self.proyeccion(features)
        if self.usar_pe:
            h = h + _pe_sinusoidal(t_ini_norm, self.d) + _pe_sinusoidal(x_ini_norm, self.d)
        h = self.encoder(h.unsqueeze(0))
        return h.squeeze(0)

    def forward(
        self,
        features: torch.Tensor,
        t_ini_norm: torch.Tensor,
        x_ini_norm: torch.Tensor,
        pares_idx: torch.Tensor,
        geom_feats_pares: torch.Tensor,
    ) -> torch.Tensor:
        """`pares_idx`: `(P, 2)` long. `geom_feats_pares`: `(P, N_GEOM_FEATURES)`,
        ya normalizado (`construir_geom_feats_pares`). Devuelve `(P,)` logits."""
        h = self.contextualizar(features, t_ini_norm, x_ini_norm)
        hi, hj = h[pares_idx[:, 0]], h[pares_idx[:, 1]]
        entrada = torch.cat([hi, hj, hi - hj, geom_feats_pares], dim=-1)
        return self.head_enlace(entrada).squeeze(-1)


def costos_modelo(
    logits: torch.Tensor, pares: list[tuple[int, int]], *, piso_prob: float = 1e-6
) -> dict[tuple[int, int], float]:
    """`-log sigmoid(logit)`, clampeado (SS5.5): un logit muy negativo da
    `sigmoid -> 0` y costo sin cota, que desborda la centinela de
    `asociacion.decodificar_bipartito` (1e6) y corrompe la asignacion en silencio.
    `piso_prob=1e-6` acota el costo maximo en `-log(1e-6) ~= 13.8`, muy por debajo
    de la centinela."""
    probs = torch.sigmoid(logits).clamp(min=piso_prob, max=1.0)
    costos = (-torch.log(probs)).detach().cpu().numpy()
    return {par: float(c) for par, c in zip(pares, costos)}


# --------------------------------------------------------------------------- #
# v2 -- el hibrido con tiles de DecNet (SS5.6)
# --------------------------------------------------------------------------- #
def preparar_contexto_tiles(
    preprocessed: np.ndarray, skel: np.ndarray, *, dim: int = VISION_MODULE_TILE_SIZE
) -> tuple[np.ndarray, np.ndarray, int, KDTree]:
    """`(padkym, allyx_padded, pad_size, kdtree)` compartidos por todos los
    segmentos de UNA muestra (SS5.6) -- replica el padding de
    `vision_module.get_candidates` (`pad_size = round(1 + dim/2)`, constante 0.1) y
    arma `allyx` como el esqueleto COMPLETO (mismo `np.where` que
    `kymobutler.tracking.track_bidirectional` usa para su propio `allyx_coords`, no
    solo los puntos del segmento). El KD-tree se construye UNA vez aca y se pasa a
    `extraer_tile_segmento` -- construirlo de nuevo por cada extremo de cada
    segmento (hasta ~400 veces por muestra) fue el primer intento y es el cuello de
    botella real de esta etapa, mismo patron que el `np.where(etiquetas==i)` por
    componente de `asociacion.partir_en_segmentos` (ver su docstring)."""
    pad_size = round(1 + dim / 2)
    padkym = np.pad(preprocessed, pad_size, mode="constant", constant_values=0.1)
    allyx = np.array(list(zip(*np.where(skel))), dtype=np.float64)
    allyx_padded = allyx + pad_size
    kdtree = KDTree(allyx_padded)
    return padkym, allyx_padded, pad_size, kdtree


def extraer_tile_segmento(
    segmento: Segmento,
    extremo: str,
    padkym: np.ndarray,
    allyx_padded: np.ndarray,
    pad_size: int,
    kdtree: KDTree,
    *,
    dim: int = VISION_MODULE_TILE_SIZE,
    n_filas_track: int = 20,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Replica `vision_module.get_candidates` (lineas 203-224) para UN extremo de
    UN segmento en vez de una traza creciendo (SS5.6): restringe `allyx` (canal de
    estructura) a `dim*1.5` px por KD-tree, ordena por fila, y **descarta el
    ultimo punto** (`track_padded[:-1]`) antes de centrar el tile -- igual que
    DecNet. Eso significa que el tile queda centrado un frame *antes* del extremo
    geometrico real que usan `geom_feats`/`costo_asociacion`: desvio de 1 fila
    documentado, no corregido a proposito -- corregirlo haria que el tile dejara
    de ser bit-identico al que DecNet realmente ve, que es el punto entero de v2.

    `extremo="fin"` usa los puntos del segmento tal cual (el canal de track
    significa "camino ya andado", igual que en DecNet). `extremo="ini"` usa los
    puntos INVERTIDOS -- el canal de track pasa a significar "el segmento por
    delante", la semantica opuesta a la de DecNet; irreducible dado que un
    segmento no tiene un "antes" real en su propio extremo inicial (SS5.6).
    Ambos casos truncan a las ultimas `n_filas_track` filas (frames unicos, no
    puntos crudos) antes de aplicar el `[:-1]`.

    Devuelve `(tile, centro_esperado)`: `tile` es `(3, dim, dim)` =
    `[recorte, mascara_track, mascara_estructura]` apilados (VisionNet real toma
    tres tensores separados; apilar para nuestro propio encoder CNN es una
    desviacion documentada, SS5.6 lo permite). `centro_esperado` = `(fila, col)`
    donde debe caer el punto de conexion DENTRO del tile (el ultimo punto de
    `track_for_tile`, no el extremo geometrico) -- para el chequeo de alineacion
    obligatorio (SS10 checklist). Devuelve `(ceros, None)` si el segmento es
    demasiado corto (menos de 2 puntos tras truncar + descartar el ultimo) --
    contar estos casos, son una tercera clase de entrada que el modelo aprende
    a manejar (tile en blanco), no un error silencioso."""
    puntos = segmento.puntos if extremo == "fin" else segmento.puntos[::-1]
    puntos = puntos[-n_filas_track:]
    if len(puntos) < 2:
        return np.zeros((3, dim, dim), dtype=np.float32), None

    track_padded = [(float(r) + pad_size, float(c) + pad_size) for r, c in puntos]
    track_for_tile = track_padded[:-1]
    if len(track_for_tile) < 2:
        return np.zeros((3, dim, dim), dtype=np.float32), None

    last_padded = np.array(track_padded[-1])
    nearby_idx = kdtree.query_ball_point(last_padded, r=dim * 1.5)
    nearby = allyx_padded[nearby_idx]
    nearby = nearby[nearby[:, 0].argsort()] if len(nearby) else nearby

    tile, track_mask, struct_mask, win = get_tile(padkym, track_for_tile, nearby, dim)
    if tile.shape != (dim, dim):
        return np.zeros((3, dim, dim), dtype=np.float32), None

    centro_esperado = (track_for_tile[-1][0] - win[0][0], track_for_tile[-1][1] - win[1][0])
    apilado = np.stack([tile, track_mask, struct_mask], axis=0).astype(np.float32)
    return apilado, centro_esperado


class CodificadorTile(nn.Module):
    """Encoder CNN para un tile `(3, dim, dim)` (SS5.6): tres bloques conv 3x3
    stride-2 (3->32->64->128), GELU, global average pool, `Linear -> d`. ~110k
    parametros (la guia estima ~120k) -- chico a proposito, el punto de v2 es
    agregar la informacion visual de DecNet sin salirse de "unos minutos por
    epoca en CPU/MPS"."""

    def __init__(self, d: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
        )
        self.proyeccion = nn.Linear(128, d)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """`(N, 3, dim, dim) -> (N, d)`."""
        h = self.conv(tiles)
        h = h.mean(dim=(2, 3))  # global average pool
        return self.proyeccion(h)


class ModeloAtencionV2(nn.Module):
    """v1 (SS5.3) + un `CodificadorTile` por cada extremo, fusionado por SUMA en
    el token (SS5.6) -- "misma informacion visual que DecNet, razonamiento global
    en vez de local-greedy": el mismo tile 48x48 de tres canales que DecNet
    consume en cada paso, pero visto junto con el resto del kymografo en vez de
    aislado. Concatenacion-y-proyeccion es la alternativa que la guia menciona;
    suma es mas simple y mantiene `d` fijo, misma eleccion que v1 hace para la
    codificacion posicional."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d: int = 128,
        n_capas: int = 4,
        n_cabezas: int = 4,
        n_geom: int = N_GEOM_FEATURES,
        usar_pe: bool = True,
    ):
        super().__init__()
        self.d = d
        self.usar_pe = usar_pe
        self.proyeccion = nn.Linear(n_features, d)
        self.codificador_tile = CodificadorTile(d)
        capa = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_cabezas, dim_feedforward=d * 4,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(capa, num_layers=n_capas)
        self.head_enlace = nn.Sequential(
            nn.Linear(3 * d + n_geom, d), nn.GELU(), nn.Linear(d, 1),
        )

    def contextualizar(
        self,
        features: torch.Tensor,
        t_ini_norm: torch.Tensor,
        x_ini_norm: torch.Tensor,
        tiles_fin: torch.Tensor,
        tiles_ini: torch.Tensor,
    ) -> torch.Tensor:
        """`tiles_fin`/`tiles_ini`: `(N, 3, dim, dim)`, un tile por segmento y
        extremo (`extraer_tile_segmento`)."""
        h = self.proyeccion(features)
        h = h + self.codificador_tile(tiles_fin) + self.codificador_tile(tiles_ini)
        if self.usar_pe:
            h = h + _pe_sinusoidal(t_ini_norm, self.d) + _pe_sinusoidal(x_ini_norm, self.d)
        h = self.encoder(h.unsqueeze(0))
        return h.squeeze(0)

    def forward(
        self,
        features: torch.Tensor,
        t_ini_norm: torch.Tensor,
        x_ini_norm: torch.Tensor,
        tiles_fin: torch.Tensor,
        tiles_ini: torch.Tensor,
        pares_idx: torch.Tensor,
        geom_feats_pares: torch.Tensor,
    ) -> torch.Tensor:
        h = self.contextualizar(features, t_ini_norm, x_ini_norm, tiles_fin, tiles_ini)
        hi, hj = h[pares_idx[:, 0]], h[pares_idx[:, 1]]
        entrada = torch.cat([hi, hj, hi - hj, geom_feats_pares], dim=-1)
        return self.head_enlace(entrada).squeeze(-1)


# --------------------------------------------------------------------------- #
# v3 -- dustbin: el rechazo como decision aprendida, sin umbral global
# --------------------------------------------------------------------------- #
# Motivacion medida (`scripts/diagnostico_rechazo.py` ->
# `results/asociacion/diagnostico_rechazo.json`): v1/v2 NO fallan al desambiguar
# (0.58/0.60 de exactitud de argmax en grupos ambiguos, contra 0.42 de eleccion
# aleatoria) sino al RECHAZAR: el 50.7% de los segmentos con candidatos no tiene
# sucesor verdadero, y en su punto de operacion v1 los enlaza al 100% igual. Cada
# enlace falso encadena dos particulas en una polilinea = un ID-switch.
#
# La causa es estructural, no de capacidad: v1/v2 emiten una sigmoide INDEPENDIENTE
# por par y el decodificador aplica UN umbral global. En el punto que iguala la
# fragmentacion de DecNet (1.282) ese umbral acepta cualquier par por encima de
# ~1% de probabilidad, asi que rechazar nunca ocurre. DecNet, en cambio, devuelve
# vacio en el 63% de sus consultas: es conservador por construccion, y por eso gana.
#
# v3 cambia exactamente eso y nada mas: cada segmento recibe un puntaje "sin
# sucesor" (dustbin) aprendido, y se toma un softmax por segmento sobre
# [sus candidatos ; dustbin]. El rechazo compite LOCALMENTE contra los candidatos
# reales en vez de contra una constante global. Es el dustbin de SuperGlue, el
# linaje que `ModeloAtencionV1` ya citaba sin implementarlo. El head de pares, los
# tokens, la codificacion posicional y el encoder son identicos a v1: la
# comparacion v1-vs-v3 aisla el rechazo.

_NEG_INF = -1e9  # no usar float("-inf"): 0 * inf = nan en el backward del softmax


class ModeloAtencionV3(ModeloAtencionV1):
    """v1 + dos cabezas de dustbin (SS "v3"). Hereda tokens, PE, encoder y head de
    pares de `ModeloAtencionV1` sin tocarlos -- lo unico nuevo es el puntaje de
    "este segmento no se enlaza".

    Dos lados, porque un enlace falso puede romper cualquiera de los dos extremos:
    `dustbin_sucesor(i)` = "i no continua en ningun segmento", `dustbin_predecesor(j)`
    = "j no viene de ningun segmento". El decodificador exige que ambos lados
    prefieran el enlace sobre su dustbin (`admisibles_dustbin`)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.d
        self.head_dustbin_sucesor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.head_dustbin_predecesor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(
        self,
        features: torch.Tensor,
        t_ini_norm: torch.Tensor,
        x_ini_norm: torch.Tensor,
        pares_idx: torch.Tensor,
        geom_feats_pares: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve `(logits_pares (P,), dustbin_sucesor (N,), dustbin_predecesor (N,))`."""
        h = self.contextualizar(features, t_ini_norm, x_ini_norm)
        hi, hj = h[pares_idx[:, 0]], h[pares_idx[:, 1]]
        entrada = torch.cat([hi, hj, hi - hj, geom_feats_pares], dim=-1)
        logits = self.head_enlace(entrada).squeeze(-1)
        return logits, self.head_dustbin_sucesor(h).squeeze(-1), self.head_dustbin_predecesor(h).squeeze(-1)


def construir_grupos(
    pares: list[tuple[int, int]],
    n_segmentos: int,
    verdaderos: set[tuple[int, int]] | None,
    ids_asignados: list[int] | None,
    *,
    lado: str,
    es_estatico: list[bool] | None = None,
) -> dict | None:
    """Estructuras (estaticas por muestra, se precomputan UNA vez) para el softmax
    por segmento de v3.

    `lado="sucesor"`: un grupo por segmento `i` con candidatos, sobre
    `{j : (i,j) in pares}`. `lado="predecesor"`: un grupo por `j`, sobre
    `{i : (i,j) in pares}`.

    Devuelve `{"idx", "mask", "objetivo", "segmentos"}` donde `idx` es `(G, K)` de
    indices sobre el vector concatenado `[logits_pares ; dustbin]` -- la ranura
    `K-1` de cada grupo apunta al dustbin de SU segmento. `objetivo[g]` es la
    posicion del enlace verdadero dentro del grupo, o la ranura del dustbin si ese
    segmento no tiene enlace verdadero entre sus candidatos.

    **Se construyen grupos para TODOS los segmentos con candidatos**, y el flag
    `en_perdida` marca cuales entran a la cross-entropy: solo aquellos cuyo PROPIO
    segmento esta asignado (`particle_id != -1`). Para un segmento asignado sabemos
    la verdad -- o su particula continua en un candidato concreto, o no continua -- y
    ese "o no continua" es exactamente la senal de rechazo que a v1/v2 les falta.
    Para un segmento sin asignar no la sabemos, y etiquetarlo dustbin seria inventar.

    **El grupo se construye igual aunque no entre a la perdida**, y esto NO es un
    detalle: `costos_dustbin` exige que los dos lados de un par prefieran el enlace a
    su dustbin, asi que si los grupos de los segmentos sin asignar no existieran,
    ningun par que los toque podria ser admisible JAMAS -- el decodificador quedaria
    mutilado exactamente sobre los segmentos mas ambiguos. (Se detecto en el smoke
    test de v3: con grupos filtrados, 0 de 8 pares resultaban admisibles a cualquier
    margen.) Los candidatos sin asignar tambien participan del softmax de un grupo
    asignado: el modelo aprende a no enlazarse con ellos poniendo masa en el dustbin.

    Ruido residual conocido: si el sucesor verdadero existe pero la poda de
    `pares_candidatos` lo dejo fuera (SS4.1: 5.0% de los enlaces verdaderos), el
    objetivo queda en dustbin siendo que si habia sucesor. Se documenta, no se
    corrige: corregirlo exigiria GT en inferencia.

    **`es_estatico` -- la variante que corrige la calibracion del dustbin.** Sin el,
    solo los segmentos asignados entran a la perdida, y eso deja al dustbin calibrado
    sobre la poblacion equivocada: medido sobre el v3 base, el modelo elige dustbin en
    el 2.2% de los grupos en inferencia cuando la tasa real de rechazo es ~54.4%,
    reproduciendo fielmente el 6.2% de sus etiquetas. Con `es_estatico`, los segmentos
    SIN asignar cuyo desplazamiento propio queda por debajo del umbral movil (bandas
    estaticas: mediana 0.0 px, 92% por debajo de 2 px) entran a la perdida con objetivo
    DUSTBIN.

    Esto es una decision de modelado, no ground truth, y hay que decirlo asi: el GT no
    afirma "esta banda estatica no continua", afirma que no es una particula movil. La
    etiqueta se justifica por la tarea -- el sistema solo reporta trayectorias moviles
    (`es_movil` filtra el resto), asi que un segmento estatico enlazado a algo nunca es
    util y a veces es danino (encadena una estatica con una movil). Se pasa por flag
    para que quede como ablacion limpia contra el v3 base."""
    if lado not in ("sucesor", "predecesor"):
        raise ValueError(f"lado debe ser 'sucesor' o 'predecesor', recibido {lado!r}")
    clave = 0 if lado == "sucesor" else 1
    otro = 1 - clave

    por_segmento: dict[int, list[int]] = {}
    for k, par in enumerate(pares):
        por_segmento.setdefault(par[clave], []).append(k)
    if not por_segmento:
        return None

    con_gt = verdaderos is not None and ids_asignados is not None
    enlace_de: dict[int, int] = {}
    if con_gt:
        for i, j in verdaderos:
            enlace_de[i if clave == 0 else j] = j if clave == 0 else i

    grupos = sorted(por_segmento.items())
    n_grupos = len(grupos)
    k_max = max(len(ks) for _, ks in grupos) + 1  # +1 = ranura del dustbin
    idx = np.zeros((n_grupos, k_max), dtype=np.int64)
    mask = np.zeros((n_grupos, k_max), dtype=bool)
    objetivo = np.zeros(n_grupos, dtype=np.int64)
    segmentos = np.zeros(n_grupos, dtype=np.int64)
    en_perdida = np.zeros(n_grupos, dtype=bool)
    n_pares = len(pares)

    for g, (s, ks) in enumerate(grupos):
        n_k = len(ks)
        idx[g, :n_k] = ks
        idx[g, n_k] = n_pares + s  # dustbin de ESTE segmento
        mask[g, : n_k + 1] = True
        segmentos[g] = s
        objetivo[g] = n_k  # dustbin por defecto
        if not con_gt:
            continue
        if ids_asignados[s] != -1:
            en_perdida[g] = True
        elif es_estatico is not None and es_estatico[s]:
            en_perdida[g] = True
            continue  # objetivo ya es el dustbin, y `enlace_de` no tiene entradas para -1
        if s in enlace_de:
            pareja = enlace_de[s]
            for posicion, k in enumerate(ks):
                if pares[k][otro] == pareja:
                    objetivo[g] = posicion
                    break

    return {"idx": idx, "mask": mask, "objetivo": objetivo,
            "segmentos": segmentos, "en_perdida": en_perdida}


def _matriz_grupos(
    logits_pares: torch.Tensor, dustbin: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """`(G, K)` de logits, con `_NEG_INF` en el relleno -- listo para
    `cross_entropy`/`softmax` por fila."""
    todos = torch.cat([logits_pares, dustbin])
    return torch.where(mask, todos[idx], torch.full_like(mask, _NEG_INF, dtype=todos.dtype))


def perdida_dustbin(
    logits_pares: torch.Tensor,
    dustbin_suc: torch.Tensor,
    dustbin_pred: torch.Tensor,
    grupos_suc: dict | None,
    grupos_pred: dict | None,
) -> tuple[torch.Tensor, int]:
    """Cross-entropy sobre el softmax por segmento de los dos lados (SS v3).

    Reemplaza la `BCEWithLogitsLoss` por par de v1/v2. No lleva `pos_weight`: el
    desbalance positivo/negativo desaparece por construccion, porque cada grupo
    tiene exactamente UNA respuesta correcta (un candidato o el dustbin) en vez de
    N decisiones binarias independientes."""
    total = torch.zeros((), dtype=logits_pares.dtype, device=logits_pares.device)
    n = 0
    for dustbin, grupos in ((dustbin_suc, grupos_suc), (dustbin_pred, grupos_pred)):
        if grupos is None:
            continue
        sel = grupos["en_perdida"]
        if not bool(sel.any()):
            continue
        m = _matriz_grupos(logits_pares, dustbin, grupos["idx"][sel], grupos["mask"][sel])
        total = total + nn.functional.cross_entropy(m, grupos["objetivo"][sel], reduction="sum")
        n += m.shape[0]
    return (total / max(n, 1)), n


@torch.no_grad()
def costos_dustbin(
    logits_pares: torch.Tensor,
    dustbin_suc: torch.Tensor,
    dustbin_pred: torch.Tensor,
    grupos_suc: dict | None,
    grupos_pred: dict | None,
    pares: list[tuple[int, int]],
    *,
    margen: float = 1.0,
    piso_prob: float = 1e-6,
) -> dict[tuple[int, int], float]:
    """Pares ADMISIBLES y su costo, sin umbral global (SS v3).

    Un par `(i,j)` es admisible solo si los dos lados prefieren el enlace a su
    propio dustbin: `p_suc(i->j) > margen * p_suc(i->nada)` y
    `p_pred(j<-i) > margen * p_pred(j<-nada)`. Costo = `-log p_suc(i->j)`, para que
    `asociacion.decodificar_bipartito` resuelva la restriccion de un-sucesor/
    un-predecesor sobre el conjunto ya filtrado.

    `margen=1.0` es "decide el dustbin", el punto sin parametros. Barrerlo
    (0.5 = mas permisivo, 2+ = mas conservador) genera la curva
    `frac_id_switch` vs `fragmentos_por_gt` comparable con la de las filas 2/3/3b,
    pero a diferencia del umbral global de v1/v2 el punto `margen=1.0` es
    autocontenido: no hay que elegirlo en val."""
    n_pares = len(pares)
    prob_par: dict[int, float] = {}
    prefiere: dict[str, set[int]] = {"sucesor": set(), "predecesor": set()}
    for nombre, dustbin, grupos in (
        ("sucesor", dustbin_suc, grupos_suc), ("predecesor", dustbin_pred, grupos_pred),
    ):
        if grupos is None:
            continue
        m = _matriz_grupos(logits_pares, dustbin, grupos["idx"], grupos["mask"])
        p = torch.softmax(m, dim=1).cpu().numpy()
        idx = grupos["idx"].cpu().numpy()
        mask = grupos["mask"].cpu().numpy()
        for g in range(p.shape[0]):
            k_dust = int(mask[g].sum()) - 1
            p_dust = p[g, k_dust]
            for k in range(k_dust):
                k_par = int(idx[g, k])
                if p[g, k] > margen * p_dust:
                    prefiere[nombre].add(k_par)
                if nombre == "sucesor":
                    prob_par[k_par] = float(p[g, k])

    admisibles = prefiere["sucesor"] & prefiere["predecesor"]
    salida = {}
    for k in admisibles:
        if k >= n_pares:
            continue
        salida[pares[k]] = float(-np.log(max(prob_par.get(k, piso_prob), piso_prob)))
    return salida
