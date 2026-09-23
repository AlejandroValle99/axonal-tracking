"""Capa de datos por-pixel para KymoRoPE (`plan/kymorope-guide.md` SS3 y SS6.1).

Construye los tres targets del modelo a partir del `positions.csv` de synthkymo,
**reusando las mismas funciones que exportan las etiquetas de los otros pipelines**
(`etiquetas_deteccion.mascara_traza` / `ids_que_se_mueven`), incluido el espejo L-R
de la columna. No hay un segundo rasterizador aca a proposito: duplicarlo es como se
reintroduce el bug que la docstring de `mascara_traza(dilatar=False)` documenta.

Targets (SS2.5 de la guia):

  trackness   (T, L) int8    0=fondo / 1=estatica / 2=movil
  instancias  (T, L) int16   0=fondo, 1..K = particulas MOVILES (el embedding solo
                             se supervisa aca)
  theta       (T, L, 2)      (sin, cos) del angulo de la traza en coordenadas de
                             IMAGEN (dcol/dfila), no en pos_um -- ver `_theta_por_particula`

Pixeles reclamados por >=2 particulas (mediana 7.7% de los pixeles de traza, maximo
24%, medido 2026-09-22): se excluyen del termino de atraccion de la perdida
discriminativa y de la perdida de orientacion, via `mask_pull` / `mask_theta`. Es la
**opcion 1** de SS3 de la guia. La opcion 2 (target K=2 en las junciones,
construible porque `mascara_traza(dilatar=False)` conserva el solape) no esta
implementada; si se prueba, va como un flag aca y se registra en la guia.

Resolucion nativa: NADA se redimensiona. Cada muestra pasa por el stem por
separado (las convoluciones no se batchean con formas distintas) y los tokens se
apilan PADDEADOS a `(B, N_max, d)` con una mascara de clave, no concatenados en una
sola secuencia: la mascara bloque-diagonal de una secuencia concatenada tapa lo que
corresponde pero `scaled_dot_product_attention` calcula la matriz completa igual,
o sea `(B*N)^2` en vez de `B*N^2` (ver `kymorope.KymoRoPE.forward`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from axonal_tracking.etiquetas_deteccion import ids_que_se_mueven, mascara_traza
from axonal_tracking.parametros import PIXEL_SIZE_UM

__all__ = [
    "DatasetKymografos",
    "LoteEmpaquetado",
    "MuestraPixel",
    "TargetsPixel",
    "agrupar_por_tokens",
    "collate_empaquetado",
    "construir_targets",
    "normalizar_kymografo",
]

MIN_DESPLAZAMIENTO_PX = 4.0  # mismo umbral de "movil" que el resto del repo
PERCENTILES_CONTRASTE = (50.0, 99.8)  # mismo criterio que preprocesamiento.frame_a_rgb_uint8

# Clases del head de trackness (SS2.5)
CLASE_FONDO, CLASE_ESTATICA, CLASE_MOVIL = 0, 1, 2


@dataclass
class TargetsPixel:
    """Los tres targets + las mascaras de supervision de una muestra."""

    trackness: np.ndarray  # (T, L) int8, {0, 1, 2}
    instancias: np.ndarray  # (T, L) int16, 0 = fondo, 1..K = moviles
    theta: np.ndarray  # (T, L, 2) float32, (sin, cos); (0, 1) donde no aplica
    mask_pull: np.ndarray  # (T, L) bool: pixel movil reclamado por EXACTAMENTE una particula
    mask_theta: np.ndarray  # (T, L) bool: pixel de traza reclamado por exactamente una
    n_instancias: int


def normalizar_kymografo(kymo: np.ndarray) -> np.ndarray:
    """`(T, L)` cualquier dtype -> float32 en [0, 1] por estiramiento p50-p99.8.

    Mismo criterio de contraste que `preprocesamiento.frame_a_rgb_uint8` y que
    `etiquetas_deteccion.kymografo_a_rgb_uint8`: anclado al fondo, no al minimo
    absoluto. Se normaliza POR KYMOGRAFO porque el bit-depth y el brillo varian
    entre sesiones (mismo argumento que la normalizacion de features de SS5.2 de
    `atencion.py`)."""
    k = np.asarray(kymo, dtype=np.float32)
    lo, hi = np.percentile(k, PERCENTILES_CONTRASTE)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((k - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _theta_por_particula(
    positions: pd.DataFrame, particle_id: int, pixel_scale_um: float, ancho_px: int
) -> dict[int, float]:
    """`{frame: theta_rad}` de UNA particula, en coordenadas de IMAGEN.

    **Espejo L-R**: `mascara_traza` rasteriza en `col = (L-1) - pos_um/pixel_scale`,
    asi que una particula anterograda (pos_um creciente) va hacia columnas MENORES.
    La pendiente en imagen es por lo tanto `-(d pos_um/d frame) / pixel_scale`. El
    post-procesador de SS4.3 vuelve a pos_um para la columna `Inclination angle` de
    la planilla del lab; el modelo ve la imagen, asi que el target vive en imagen.

    `theta = atan(dcol/dfila)` en px/frame, con `dfila = 1` por construccion (el
    tiempo siempre avanza hacia abajo), asi que theta in (-pi/2, pi/2) y
    `cos(theta) > 0` siempre -- la parametrizacion (sin, cos) no es ambigua.
    Diferencias centradas; en los extremos, diferencia hacia adelante/atras."""
    sub = positions[positions["particle_id"] == particle_id].sort_values("frame")
    frames = sub["frame"].to_numpy()
    cols = (ancho_px - 1) - sub["pos_um"].to_numpy() / pixel_scale_um
    if len(frames) < 2:
        return {int(frames[0]): 0.0} if len(frames) else {}
    # gradiente respecto del frame (no del indice): tolera huecos en `frame`
    dcol_dframe = np.gradient(cols, frames.astype(np.float64))
    return {int(f): float(np.arctan(v)) for f, v in zip(frames, dcol_dframe)}


def construir_targets(
    positions: pd.DataFrame,
    shape: tuple[int, int],
    *,
    pixel_scale_um: float = PIXEL_SIZE_UM,
    min_desplazamiento_px: float = MIN_DESPLAZAMIENTO_PX,
) -> TargetsPixel:
    """`positions.csv` -> los tres targets por-pixel de SS2.5.

    Las moviles se pintan DESPUES de las estaticas en `trackness` a proposito: donde
    una movil pasa por encima de una estatica el pixel es de la movil. No es un
    detalle -- el 32.7% de los frames tiene una movil a <2 px de una estatica
    (medido sobre 20 muestras de train), y es la ambiguedad dominante del dataset.
    """
    ancho = shape[1]
    moviles = sorted(ids_que_se_mueven(positions, pixel_scale_um, min_desplazamiento_px))
    estaticas = [p for p in sorted(positions["particle_id"].unique()) if p not in moviles]

    trackness = np.zeros(shape, np.int8)
    instancias = np.zeros(shape, np.int16)
    conteo = np.zeros(shape, np.int16)  # cuantas particulas reclaman cada pixel
    theta_rad = np.zeros(shape, np.float32)

    orden_pintado = [(p, CLASE_ESTATICA) for p in estaticas] + [(p, CLASE_MOVIL) for p in moviles]
    for pid, clase in orden_pintado:
        m = mascara_traza(positions, pid, pixel_scale_um, shape)
        if not m.any():
            continue
        conteo += m
        trackness[m] = clase
        th = _theta_por_particula(positions, pid, pixel_scale_um, ancho)
        filas, columnas = np.where(m)
        # la mascara esta dilatada: una fila puede no estar en `th` si la particula
        # no existe en ese frame pero la dilatacion la alcanzo -> theta 0 por defecto
        theta_rad[filas, columnas] = np.array(
            [th.get(int(f), 0.0) for f in filas], dtype=np.float32
        )
        if clase == CLASE_MOVIL:
            instancias[m] = moviles.index(pid) + 1

    es_traza = conteo >= 1
    unico = conteo == 1
    theta = np.stack([np.sin(theta_rad), np.cos(theta_rad)], axis=-1).astype(np.float32)
    theta[~es_traza] = (0.0, 1.0)  # neutro donde no se supervisa

    return TargetsPixel(
        trackness=trackness,
        instancias=instancias,
        theta=theta,
        mask_pull=(trackness == CLASE_MOVIL) & unico,
        mask_theta=es_traza & unico,
        n_instancias=len(moviles),
    )


@dataclass
class MuestraPixel:
    """Una muestra lista para el modelo. Todo float32 (MPS no soporta float64)."""

    nombre: str
    kymo: torch.Tensor  # (1, T, L) float32 en [0, 1]
    trackness: torch.Tensor  # (T, L) int8  {0, 1, 2}
    instancias: torch.Tensor  # (T, L) int16, 0 = fondo
    theta: torch.Tensor  # (2, T, L) float16
    mask_pull: torch.Tensor  # (T, L) bool
    mask_theta: torch.Tensor  # (T, L) bool
    n_instancias: int
    dt_segundos: float  # s/frame  -> eje temporal de la RoPE fisica (SS2.2)
    dx_um: float  # um/px    -> eje espacial de la RoPE fisica (SS2.2)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.kymo.shape[-2:])  # type: ignore[return-value]


class DatasetKymografos(Dataset):
    """`datasets/{split}/sample_*/` -> `MuestraPixel`, a RESOLUCION NATIVA.

    No redimensiona, no recorta y no paddea: el tamano variable es el punto (SS2.1).
    `dt_segundos` sale del `fps` del `config.yaml` de cada muestra -- es lo que hace
    testeable la invariancia temporal de SS5.3 (el dataset tiene 5 tasas de muestreo
    y las cinco estan en los tres splits).

    `cache_targets=True` guarda los targets construidos en memoria; con 800 muestras
    de ~220x1300 son ~2 GB, asi que por defecto se reconstruyen (el costo dominante
    es `mascara_traza`, ~30 particulas por muestra)."""

    def __init__(
        self,
        raiz_split: Path | str,
        *,
        pixel_scale_um: float = PIXEL_SIZE_UM,
        min_desplazamiento_px: float = MIN_DESPLAZAMIENTO_PX,
        fps_permitidos: set[float] | None = None,
        limite: int | None = None,
        cache_targets: bool = False,
        cache_dir: Path | str | None = None,
    ):
        self.raiz = Path(raiz_split)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pixel_scale_um = pixel_scale_um
        self.min_desplazamiento_px = min_desplazamiento_px
        self.cache_targets = cache_targets
        self._cache: dict[int, MuestraPixel] = {}

        dirs = sorted(d for d in self.raiz.glob("sample_*") if (d / "kymograph.tif").exists())
        if fps_permitidos is not None:
            dirs = [d for d in dirs if self._leer_fps(d) in fps_permitidos]
        self.dirs = dirs[:limite] if limite else dirs
        if not self.dirs:
            raise FileNotFoundError(f"Sin muestras en {self.raiz} (fps_permitidos={fps_permitidos})")

    @staticmethod
    def _leer_fps(dir_muestra: Path) -> float:
        with open(dir_muestra / "config.yaml") as f:
            return float(yaml.safe_load(f)["general"]["fps"])

    def fps(self) -> list[float]:
        """fps de cada muestra, en el orden del dataset (para `agrupar_por_tokens`
        y para el ablation de SS5.3)."""
        return [self._leer_fps(d) for d in self.dirs]

    def formas(self) -> list[tuple[int, int]]:
        """`(T, L)` de cada muestra sin cargar los pixeles -- lo usa el bucketing."""
        import tifffile

        return [tuple(tifffile.TiffFile(d / "kymograph.tif").pages[0].shape[:2]) for d in self.dirs]

    def __len__(self) -> int:
        return len(self.dirs)

    def _ruta_cache(self, i: int) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{self.dirs[i].name}.npz"

    def _construir(self, i: int) -> dict:
        """Arrays de una muestra: lo caro (rasterizar + dilatar ~30 trazas, ~0.064 s)
        y lo unico que vale la pena cachear a disco."""
        import tifffile

        d = self.dirs[i]
        kymo = normalizar_kymografo(tifffile.imread(d / "kymograph.tif"))
        tg = construir_targets(
            pd.read_csv(d / "positions.csv"),
            kymo.shape,
            pixel_scale_um=self.pixel_scale_um,
            min_desplazamiento_px=self.min_desplazamiento_px,
        )
        return {
            "kymo": kymo,
            "trackness": tg.trackness,
            "instancias": tg.instancias,
            "theta": tg.theta,
            "mask_pull": tg.mask_pull,
            "mask_theta": tg.mask_theta,
            "n_instancias": np.int32(tg.n_instancias),
            "dt_segundos": np.float32(1.0 / self._leer_fps(d)),
        }

    def precalcular(self, verboso: bool = True) -> None:
        """Llena el cache de disco de una. Sin esto la primera epoca paga el costo
        (y con `num_workers>0` lo pagan los workers, en paralelo pero igual)."""
        if self.cache_dir is None:
            raise ValueError("precalcular() necesita cache_dir")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        faltan = [i for i in range(len(self)) if not self._ruta_cache(i).exists()]
        for k, i in enumerate(faltan):
            np.savez(self._ruta_cache(i), **self._construir(i))
            if verboso and (k + 1) % 100 == 0:
                print(f"  cache {k + 1}/{len(faltan)}")
        if verboso:
            print(f"cache listo: {len(self)} muestras en {self.cache_dir}")

    def __getitem__(self, i: int) -> MuestraPixel:
        if i in self._cache:
            return self._cache[i]

        d = self.dirs[i]
        ruta = self._ruta_cache(i)
        if ruta is not None and ruta.exists():
            with np.load(ruta) as z:
                a = {k: z[k] for k in z.files}
        else:
            a = self._construir(i)
            if ruta is not None:
                ruta.parent.mkdir(parents=True, exist_ok=True)
                np.savez(ruta, **a)

        kymo = a["kymo"]
        tg = TargetsPixel(
            trackness=a["trackness"], instancias=a["instancias"], theta=a["theta"],
            mask_pull=a["mask_pull"], mask_theta=a["mask_theta"],
            n_instancias=int(a["n_instancias"]),
        )
        muestra = MuestraPixel(
            nombre=d.name,
            # dtypes COMPACTOS: eran int64 (8 B/px) para etiquetas en {0,1,2} y
            # 0..K, y float32 para theta. A 30 B/px una muestra grande pesaba 34 MB y
            # la cola del DataLoader (workers x prefetch) multiplicaba eso. Ahora son
            # 13 B/px. Las perdidas castean donde hace falta, sobre los pixeles
            # seleccionados, no sobre el lienzo entero.
            kymo=torch.from_numpy(kymo).unsqueeze(0),
            trackness=torch.from_numpy(tg.trackness),  # int8
            instancias=torch.from_numpy(tg.instancias),  # int16
            theta=torch.from_numpy(tg.theta).permute(2, 0, 1).contiguous().half(),
            mask_pull=torch.from_numpy(tg.mask_pull),
            mask_theta=torch.from_numpy(tg.mask_theta),
            n_instancias=tg.n_instancias,
            dt_segundos=float(a["dt_segundos"]),
            dx_um=self.pixel_scale_um,
        )
        if self.cache_targets:
            self._cache[i] = muestra
        return muestra


@dataclass
class LoteEmpaquetado:
    """Un lote de muestras de tamano variable, SIN padding (SS2.3).

    Es una lista, no un tensor apilado: el stem corre por muestra y recien despues
    los tokens se apilan paddeados adentro del modelo, que es como NaViT/Pixtral
    manejan resoluciones mixtas. `tokens_estimados` es el costo del lote para el
    bucketing."""

    muestras: list[MuestraPixel]
    tokens_estimados: int

    def __len__(self) -> int:
        return len(self.muestras)

    def a(self, device: torch.device | str) -> LoteEmpaquetado:
        """Mueve los tensores al dispositivo. `non_blocking` no aplica en MPS
        (memoria unificada), asi que no se usa."""
        movidas = []
        for m in self.muestras:
            movidas.append(
                MuestraPixel(
                    nombre=m.nombre,
                    kymo=m.kymo.to(device),
                    trackness=m.trackness.to(device),
                    instancias=m.instancias.to(device),
                    theta=m.theta.to(device),
                    mask_pull=m.mask_pull.to(device),
                    mask_theta=m.mask_theta.to(device),
                    n_instancias=m.n_instancias,
                    dt_segundos=m.dt_segundos,
                    dx_um=m.dx_um,
                )
            )
        return LoteEmpaquetado(movidas, self.tokens_estimados)


def tokens_de_forma(shape: tuple[int, int], patch: int = 16) -> int:
    """Tokens que produce el stem para una forma `(T, L)` dada."""
    alto, ancho = shape
    return int(np.ceil(alto / patch) * np.ceil(ancho / patch))


def collate_empaquetado(muestras: list[MuestraPixel], patch: int = 16) -> LoteEmpaquetado:
    """`collate_fn` del DataLoader. No apila nada -- ver `LoteEmpaquetado`."""
    total = sum(tokens_de_forma(m.shape, patch) for m in muestras)
    return LoteEmpaquetado(muestras, total)


def agrupar_por_tokens(
    formas: list[tuple[int, int]],
    *,
    patch: int = 16,
    max_tokens: int = 8192,
    max_muestras: int = 8,
    barajar: bool = True,
    semilla: int = 0,
) -> list[list[int]]:
    """Bucketing: indices agrupados en lotes acotados por CONTEO DE TOKENS, no por
    numero de muestras (`batch_sampler` del DataLoader).

El tope acota el costo de la atencion. Con el lote paddeado el costo es
    `B * N_max^2`, asi que lo que conviene es que las muestras de un lote tengan
    tamanos parecidos -- de ahi que se ordene por costo antes de cortar: el padding
    desperdiciado es la diferencia contra `N_max`.

    **El tope esta en tokens pero el stem y el decoder cuestan por PIXEL**, y un
    token cubre `patch^2` px. Si se cambia `patch` hay que mover el tope en la misma
    proporcion o el lote se llena de pixeles (ver `entrenamiento.MAX_TOKENS_LOTE`).
    El peor caso de UNA muestra (463x2722) son 4959 tokens a patch 16 y 1274 a patch
    32, asi que el tope nunca deja una muestra afuera.

    Ordena por tokens para que cada lote sea homogeneo (menos desperdicio), y baraja
    el ORDEN DE LOS LOTES, no su contenido -- asi el bucketing se mantiene y el
    entrenamiento igual ve los lotes en orden aleatorio."""
    costos = [(i, tokens_de_forma(f, patch)) for i, f in enumerate(formas)]
    costos.sort(key=lambda t: t[1])

    lotes: list[list[int]] = []
    actual: list[int] = []
    acum = 0
    for i, c in costos:
        if actual and (acum + c > max_tokens or len(actual) >= max_muestras):
            lotes.append(actual)
            actual, acum = [], 0
        actual.append(i)
        acum += c
    if actual:
        lotes.append(actual)

    if barajar:
        import random

        random.Random(semilla).shuffle(lotes)
    return lotes
