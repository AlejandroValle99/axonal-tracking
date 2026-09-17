"""Asociacion de trayectorias sobre esqueletos de kymografo (notebook 11).

Capa nueva para `plan/association-transformer-guide.md`: de un mapa de trackness
(float) a una lista de `Segmento`s (tramos de esqueleto entre juncionaes), su
asignacion a ground truth, y el enlace oraculo que mide el techo de la
representacion (SS3, Gate A). No vive en `etiquetas_deteccion.py` (exportacion de
etiquetas) ni en `evaluacion.py` (metricas sobre polilineas ya armadas): esto es la
capa intermedia -- segmentos, asignaciones, cadenas -- que ninguno de los dos
modulos existentes modela.

**Por que reusar el esqueletizador de KymoButler (`esqueleto_kymobutler`) en vez de
escribir uno nuevo**: la fila 1 (DecNet) y las filas 3/3b (atencion) del ablation
matrix de SS6 tienen que partir del MISMO esqueleto para que la comparacion aisle
la decision de enlace y nada mas (SS0/SS3.1). `esqueletizar` (skimage, sin el
post-procesado de KymoButler) es SOLO para la fuente 3 (umbral clasico, SS3.3), que
no tiene una prediccion de KymoButler que darle a `process_segmentation_bi`.

**El merge de fragmentos paralelos es deliberadamente sin ground truth.** Una
version anterior de la guia (SS4.1) proponia fusionar dos segmentos cuando
comparten `particle_id` asignado. Eso no puede sostenerse: en inferencia (Stage 1-2
y produccion) no hay GT, asi que un predicado basado en el le daria al oraculo un
conjunto de segmentos mas prolijo que el que cualquier modelo real puede ver, y el
techo de Gate A dejaria de ser alcanzable por construccion -- el oraculo tiene que
operar sobre EXACTAMENTE los mismos segmentos que Stage 1/2 van a recibir.
`fusionar_fragmentos_paralelos` usa en cambio un predicado geometrico (solape
temporal + cercania espacial): el mismo criterio para el oraculo y para cualquier
modelo aguas abajo.
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from kymobutler.morphology import process_segmentation_bi
from scipy.ndimage import convolve, label
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

from axonal_tracking import etiquetas_deteccion as ed
from axonal_tracking import evaluacion as ev

__all__ = [
    "AsignacionGT",
    "CaracteristicasSegmento",
    "Segmento",
    "asignar_gt",
    "caracteristicas_segmentos",
    "costo_asociacion",
    "detectar_junciones",
    "enlaces_verdaderos",
    "enlazar_oracle",
    "enlazar_oracle_duplicado",
    "enlazar_por_costo",
    "enlazar_por_costo_greedy",
    "esqueletizar",
    "esqueleto_kymobutler",
    "fusionar_fragmentos_paralelos",
    "geom_feats",
    "pares_candidatos",
    "partir_en_segmentos",
    "polilinea_desde_puntos",
    "polilineas_desde_cadenas",
    "trackness_gt_moviles",
]

# Vecindad-8: mismo criterio de conectividad que `_filter_components`/`_prune_branches`
# de KymoButler (`structure=np.ones((3,3))`) -- los tracks son diagonales, 4-conectividad
# partiria una linea diagonal continua en pixeles sueltos.
_VECINDAD_8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
_VECINDAD_8_SIN_CENTRO = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


@dataclass
class Segmento:
    """Un tramo de esqueleto entre juncionaes: `puntos` (t, x) ordenados por tiempo,
    mas los extremos cacheados (SS3.1) -- Stage 1/2 los consultan por cada par
    candidato y no tiene sentido recomputarlos cada vez."""

    puntos: np.ndarray  # (n, 2) float, columnas (t, x), ordenado por t
    t_ini: float = field(init=False)
    t_fin: float = field(init=False)
    x_ini: float = field(init=False)
    x_fin: float = field(init=False)

    def __post_init__(self) -> None:
        self.puntos = np.asarray(self.puntos, dtype=float)
        if self.puntos.ndim != 2 or self.puntos.shape[1] != 2 or len(self.puntos) == 0:
            raise ValueError(f"puntos debe ser (n, 2) con n>=1, recibido {self.puntos.shape}")
        self.t_ini = float(self.puntos[0, 0])
        self.t_fin = float(self.puntos[-1, 0])
        self.x_ini = float(self.puntos[0, 1])
        self.x_fin = float(self.puntos[-1, 1])

    def __len__(self) -> int:
        return len(self.puntos)


@dataclass
class AsignacionGT:
    """Resultado de `asignar_gt` para UN segmento. `particle_id=-1` = sin asignar
    (ninguna particula GT gana la mayoria de filas). `pureza` es la fraccion de
    filas que votan al ganador -- un segmento con `pureza < 0.9` pisa dos
    particulas (SS3.2): es un "segmento compartido", el diagnostico central de
    Gate A."""

    particle_id: int
    pureza: float
    n_filas_votantes: int


# --------------------------------------------------------------------------- #
# SS3.1 -- de un mapa de trackness a una lista de Segmento
# --------------------------------------------------------------------------- #
def esqueleto_kymobutler(
    prediction: np.ndarray,
    shape: tuple[int, int],
    threshold: float = 0.2,
    min_size: int = 10,
    min_frames: int = 10,
) -> np.ndarray:
    """Wrapper fino sobre `process_segmentation_bi` (fase 1 de
    `kymobutler.tracking.track_bidirectional`) -- SS3.1: usar el esqueletizador de
    KymoButler, no uno nuevo, para que las filas 1 y 3/3b del ablation matrix (SS6)
    consuman el MISMO esqueleto (`smooth x2 -> thin -> _prune_branches(3) ->
    _filter_components(min_size, min_frames)`). Los defaults `0.2/10/10` son los de
    `track_bidirectional`; no cambiarlos rompe esa igualdad (SS3.1/SS12.4)."""
    return process_segmentation_bi(prediction, shape, threshold, min_size, min_frames)


def esqueletizar(mask: np.ndarray) -> np.ndarray:
    """`skimage.morphology.skeletonize`, SOLO para la fuente 3 de SS3.3 (umbral
    clasico), que no tiene una prediccion de KymoButler que darle a
    `esqueleto_kymobutler`. No usar para las fuentes 1/2 (SS3.1)."""
    return skeletonize(np.asarray(mask, dtype=bool))


def detectar_junciones(skel: np.ndarray) -> np.ndarray:
    """Pixeles del esqueleto con >= 3 vecinos en la vecindad-8 -- removerlos parte
    el esqueleto en componentes conexas = segmentos (SS3.1 diagrama)."""
    skel = np.asarray(skel, dtype=bool)
    vecinos = convolve(skel.astype(np.int32), _VECINDAD_8_SIN_CENTRO, mode="constant", cval=0)
    return skel & (vecinos >= 3)


def partir_en_segmentos(skel: np.ndarray, min_filas: int = 5) -> list[Segmento]:
    """Esqueleto binario (T, L) -> lista de `Segmento`, cada uno una componente
    conexa (8-conectada, igual que `_filter_components` de KymoButler) tras remover
    los pixeles de juncion. Se descartan componentes con menos de `min_filas`
    frames DISTINTOS (no pixeles: un tramo de esqueleto de 2px de ancho no debe
    contar el doble) -- son resto de juncion, no trazas (SS3.1).

    No fusiona fragmentos paralelos -- ver `fusionar_fragmentos_paralelos`, un paso
    separado que hay que correr siempre a continuacion (SS4.1)."""
    skel = np.asarray(skel, dtype=bool)
    junciones = detectar_junciones(skel)
    resto = skel & ~junciones
    etiquetas, n = label(resto, structure=_VECINDAD_8)
    if n == 0:
        return []
    # Agrupar por label en una sola pasada (ordenar por label + searchsorted) en vez
    # de `np.where(etiquetas == i)` por componente -- eso es O(n_componentes x
    # tamano_imagen) y con la fuente clasica (sin post-procesado, cientos de
    # componentes sobre kymografos anchos) se vuelve el cuello de botella real.
    filas_todas, cols_todas = np.where(etiquetas > 0)
    labels_todas = etiquetas[filas_todas, cols_todas]
    orden = np.argsort(labels_todas, kind="stable")
    filas_todas, cols_todas, labels_todas = filas_todas[orden], cols_todas[orden], labels_todas[orden]
    limites = np.searchsorted(labels_todas, np.arange(1, n + 2))

    segmentos = []
    for i in range(n):
        ini, fin = limites[i], limites[i + 1]
        if fin <= ini:
            continue
        filas_i, cols_i = filas_todas[ini:fin], cols_todas[ini:fin]
        if len(np.unique(filas_i)) < min_filas:
            continue
        orden_t = np.argsort(filas_i, kind="stable")
        puntos = np.stack([filas_i[orden_t], cols_i[orden_t]], axis=1).astype(float)
        segmentos.append(Segmento(puntos))
    return segmentos


def _x_medio_por_t(segmento: Segmento) -> dict[int, float]:
    """Colapsa puntos que comparten el mismo frame `t` (esqueleto de 2px de ancho) a
    su columna media -- unidad de "fila" consistente en todo el modulo."""
    por_t: dict[int, list[float]] = {}
    for t, x in segmento.puntos:
        por_t.setdefault(round(t), []).append(float(x))
    return {t: float(np.mean(xs)) for t, xs in por_t.items()}


def fusionar_fragmentos_paralelos(
    segmentos: list[Segmento],
    *,
    min_filas: int = 5,
    max_dx_px: float = 3.0,
) -> list[Segmento]:
    """Fusiona pares de segmentos que se solapan en tiempo por mas de `min_filas`
    filas Y corren a `max_dx_px` uno del otro en esas filas compartidas -- son los
    dos "rieles" en que remover pixeles de juncion partio un tramo de esqueleto de
    2px de ancho, no dos particulas distintas cruzando (SS4.1, SS9 trampa 1).

    Predicado puramente geometrico -- ver el docstring del modulo para por que NO
    usa `particle_id`. Correr siempre inmediatamente despues de
    `partir_en_segmentos`, sobre CUALQUIER fuente (oraculo o modelo real): es lo que
    mantiene "segmentos de una particula son disjuntos en tiempo", el invariante
    que SS4.1/SS5.4 asumen.
    """
    activos = list(segmentos)
    cambiado = True
    while cambiado:
        cambiado = False
        xs_por_t = [_x_medio_por_t(s) for s in activos]
        for i in range(len(activos)):
            for j in range(i + 1, len(activos)):
                a, b = activos[i], activos[j]
                t0, t1 = max(a.t_ini, b.t_ini), min(a.t_fin, b.t_fin)
                if t1 - t0 + 1 <= min_filas:
                    continue
                comunes = sorted(set(xs_por_t[i]) & set(xs_por_t[j]))
                if not comunes:
                    continue
                dx = float(np.mean([abs(xs_por_t[i][t] - xs_por_t[j][t]) for t in comunes]))
                if dx <= max_dx_px:
                    fusion_puntos = np.concatenate([a.puntos, b.puntos], axis=0)
                    fusion_puntos = fusion_puntos[np.argsort(fusion_puntos[:, 0], kind="stable")]
                    activos[i] = Segmento(fusion_puntos)
                    del activos[j]
                    cambiado = True
                    break
            if cambiado:
                break
    return activos


def polilinea_desde_puntos(
    puntos: np.ndarray, shape: tuple[int, int], kymo: np.ndarray
) -> pd.DataFrame:
    """Rasteriza puntos (t, x) -- de un segmento o de una cadena de segmentos ya
    concatenada -- a una mascara fina (T, L) y refina con `ev.extraer_subpixel`:
    el UNICO camino de centroide en todo el repo (SS3.1, no escribir un segundo)."""
    T, L = shape
    mask = np.zeros((T, L), dtype=bool)
    filas = np.clip(np.round(puntos[:, 0]).astype(int), 0, T - 1)
    cols = np.clip(np.round(puntos[:, 1]).astype(int), 0, L - 1)
    mask[filas, cols] = True
    return ev.extraer_subpixel(mask, kymo)


# --------------------------------------------------------------------------- #
# SS3.2 -- asignacion de cada segmento a su particula GT mas cercana
# --------------------------------------------------------------------------- #
def trackness_gt_moviles(
    positions: pd.DataFrame,
    pixel_scale_um: float,
    shape: tuple[int, int],
    *,
    min_desplazamiento_px: float = ed.MIN_DESPLAZAMIENTO_PX_MOVIL,
    dilatar: bool = True,
) -> np.ndarray:
    """Fuente 1 de SS3.3: union de `ed.mascara_traza` de cada movil, como mapa de
    trackness float (0/1) -- mismo formato de entrada que `prediction` de
    KymoButler, para pasar por el mismo `esqueleto_kymobutler` que la fuente 2 y
    asi validar el extractor en aislamiento del segmentador.

    `dilatar=False` -- diagnostico de Gate A (SS12.9 confound): con `dilatar=True`
    (default), dos trazas GT que se cruzan a angulo bajo se funden en un blob ancho
    que `thin()` puede colapsar a una sola linea central, borrando la juncion en vez
    de representarla -- eso infla `frac_pureza_baja` por un artefacto de dilatacion,
    no por una falla real de la representacion de segmentos. `dilatar=False` aisla
    esa variable."""
    ids_mov = ed.ids_que_se_mueven(positions, pixel_scale_um, min_desplazamiento_px)
    T, L = shape
    union = np.zeros((T, L), dtype=bool)
    for pid in ids_mov:
        union |= ed.mascara_traza(positions, int(pid), pixel_scale_um, shape, dilatar=dilatar)
    return union.astype(np.float32)


def _columnas_gt_por_frame(
    positions: pd.DataFrame,
    pixel_scale_um: float,
    L: int,
    min_desplazamiento_px: float,
) -> dict[int, list[tuple[int, float]]]:
    """Mismas columnas GT que `ev.evaluar_trayectorias_polilineas` construye
    internamente (`gt_por_frame`): solo moviles, espejo L-R, SIN filtro `visible`.
    `asignar_gt` tiene que usar esta version -- no `ed.mascara_traza` (que si filtra
    `visible` por default) -- para que la asignacion de Gate A use la MISMA nocion
    de GT que el harness que despues evalua las polilineas resultantes."""
    ids_mov = ed.ids_que_se_mueven(positions, pixel_scale_um, min_desplazamiento_px)
    movs = positions[positions["particle_id"].isin(ids_mov)].copy()
    movs["col"] = (L - 1) - movs["pos_um"].to_numpy() / pixel_scale_um
    por_frame: dict[int, list[tuple[int, float]]] = {}
    for row in movs.itertuples():
        por_frame.setdefault(int(row.frame), []).append((int(row.particle_id), float(row.col)))
    return por_frame


def _etiquetas_por_fila(
    x_por_t: dict[int, float],
    gt_por_frame: dict[int, list[tuple[int, float]]],
    thr_px: float,
) -> list[int]:
    etiquetas = []
    for t, x in x_por_t.items():
        mejor, dmin = -1, thr_px
        for pid, gx in gt_por_frame.get(t, []):
            d = abs(x - gx)
            if d <= dmin:
                mejor, dmin = pid, d
        etiquetas.append(mejor)
    return etiquetas


def asignar_gt(
    segmentos: list[Segmento],
    positions: pd.DataFrame,
    pixel_scale_um: float,
    L: int,
    *,
    thr_px: float = ev.THR_PX_TRACK,
    min_desplazamiento_px: float = ed.MIN_DESPLAZAMIENTO_PX_MOVIL,
) -> list[AsignacionGT]:
    """Para cada segmento, para cada fila (frame unico), la particula GT movil mas
    cercana en columna dentro de `thr_px` (o -1 si ninguna). Voto por mayoria sobre
    esas etiquetas de fila -> `particle_id` ganador; `pureza` = fraccion de filas
    que lo votaron (SS3.2). Un `AsignacionGT` por segmento, mismo orden que
    `segmentos` -- devolver un `list[int]` desnudo perderia `pureza`, de la que
    depende Gate A."""
    gt_por_frame = _columnas_gt_por_frame(positions, pixel_scale_um, L, min_desplazamiento_px)
    asignaciones = []
    for seg in segmentos:
        etiquetas = _etiquetas_por_fila(_x_medio_por_t(seg), gt_por_frame, thr_px)
        ganador, n_votos = Counter(etiquetas).most_common(1)[0]
        asignaciones.append(AsignacionGT(
            particle_id=int(ganador),
            pureza=round(n_votos / len(etiquetas), 4),
            n_filas_votantes=int(n_votos),
        ))
    return asignaciones


# --------------------------------------------------------------------------- #
# SS3.3 -- el enlazador oraculo (dos variantes: exclusiva y duplicada)
# --------------------------------------------------------------------------- #
def _polilineas_desde_grupos(
    grupos: dict[int, list[np.ndarray]], kymo: np.ndarray
) -> list[pd.DataFrame]:
    shape = kymo.shape[:2]
    polilineas = []
    for listas_puntos in grupos.values():
        puntos = np.concatenate(listas_puntos, axis=0)
        puntos = puntos[np.argsort(puntos[:, 0], kind="stable")]
        poli = polilinea_desde_puntos(puntos, shape, kymo)
        if len(poli):
            polilineas.append(poli)
    return polilineas


def enlazar_oracle(
    segmentos: list[Segmento],
    asignaciones: list[AsignacionGT],
    kymo: np.ndarray,
) -> list[pd.DataFrame]:
    """Oraculo EXCLUSIVO (SS3.3, la variante default): agrupa segmentos por su
    `particle_id` ganador, concatena sus puntos ordenados por t y arma una
    polilinea por particula. Cada segmento compartido cede TODAS sus filas a su
    ganador -- la particula perdedora las pierde. Es el techo honesto para
    cualquier modelo que deba asignar cada segmento una sola vez (Stage 1/2)."""
    grupos: dict[int, list[np.ndarray]] = {}
    for seg, asign in zip(segmentos, asignaciones):
        if asign.particle_id == -1:
            continue
        grupos.setdefault(asign.particle_id, []).append(seg.puntos)
    return _polilineas_desde_grupos(grupos, kymo)


def enlazar_oracle_duplicado(
    segmentos: list[Segmento],
    asignaciones: list[AsignacionGT],
    positions: pd.DataFrame,
    pixel_scale_um: float,
    kymo: np.ndarray,
    *,
    thr_px: float = ev.THR_PX_TRACK,
    min_desplazamiento_px: float = ed.MIN_DESPLAZAMIENTO_PX_MOVIL,
    umbral_compartido: float = 0.9,
) -> list[pd.DataFrame]:
    """Oraculo DUPLICADO (SS3.3): un segmento con `pureza < umbral_compartido` se
    agrega a las cadenas de AMBAS particulas que pisa (su ganador y su
    segunda-mas-votada), no solo a la ganadora. Ningun modelo de asignacion unica
    puede alcanzar esto -- separa "la representacion perdio los pixeles" de "la
    representacion forzo una eleccion equivocada". Recomputa las etiquetas de fila
    del segmento (barato: solo para los `pureza<umbral_compartido`, tipicamente
    <=10% de N) para leer la segunda particula, que `AsignacionGT` no guarda."""
    L = kymo.shape[1]
    gt_por_frame = _columnas_gt_por_frame(positions, pixel_scale_um, L, min_desplazamiento_px)
    grupos: dict[int, list[np.ndarray]] = {}
    for seg, asign in zip(segmentos, asignaciones):
        if asign.particle_id == -1:
            continue
        grupos.setdefault(asign.particle_id, []).append(seg.puntos)
        if asign.pureza < umbral_compartido:
            etiquetas = _etiquetas_por_fila(_x_medio_por_t(seg), gt_por_frame, thr_px)
            comunes = Counter(e for e in etiquetas if e != -1).most_common(2)
            if len(comunes) > 1 and comunes[1][0] != asign.particle_id:
                grupos.setdefault(comunes[1][0], []).append(seg.puntos)
    return _polilineas_desde_grupos(grupos, kymo)


# --------------------------------------------------------------------------- #
# SS4 -- Stage 1: costo clasico de asociacion (requisito 5.4, piso clasico)
# --------------------------------------------------------------------------- #
@dataclass
class CaracteristicasSegmento:
    """Features por segmento, precomputadas UNA vez (SS4.2/SS5.2) -- el costo
    clasico y (mas adelante) los tokens de Stage 2 las reusan sin recomputar nada
    por par, y son la garantia de que ambos ven exactamente la misma informacion
    geometrica/fotometrica."""

    pendiente: float  # px/frame, ajuste lineal sobre TODOS los puntos (SS5.2, token de Stage 2)
    pendiente_salida: float  # px/frame, ajuste local sobre las ULTIMAS ~10 filas (SS4.2, termino dv)
    pendiente_entrada: float  # px/frame, ajuste local sobre las PRIMERAS ~10 filas (SS4.2, termino dv)
    intensidad_media: float
    intensidad_std: float


_VENTANA_PENDIENTE_LOCAL = 10  # filas, ver docstring de _pendiente_ventana


def _pendiente_segmento(segmento: Segmento) -> float:
    x_por_t = _x_medio_por_t(segmento)
    ts = np.array(sorted(x_por_t))
    xs = np.array([x_por_t[t] for t in ts])
    return ev.velocidad_px_frame(ts, xs)


def _pendiente_ventana(segmento: Segmento, extremo: str, n_filas: int = _VENTANA_PENDIENTE_LOCAL) -> float:
    """Pendiente ajustada solo sobre las ultimas (`extremo="fin"`) o primeras
    (`extremo="ini"`) `n_filas` filas -- la velocidad LOCAL en el extremo de
    conexion, no el promedio de todo el segmento. `_pendiente_segmento` (para el
    token de Stage 2, SS5.2) promedia un segmento entero; si ese segmento tiene una
    pausa y una corrida, el promedio no coincide con la velocidad real en ninguno de
    los dos extremos -- exactamente el caso que el termino `dv` del costo clasico
    (SS4.2, "velocidad" como senal de continuidad) necesita evitar."""
    x_por_t = _x_medio_por_t(segmento)
    ts = np.array(sorted(x_por_t))
    ts_ventana = ts[-n_filas:] if extremo == "fin" else ts[:n_filas]
    xs_ventana = np.array([x_por_t[t] for t in ts_ventana])
    return ev.velocidad_px_frame(ts_ventana, xs_ventana)


def _intensidad_segmento(segmento: Segmento, kymo: np.ndarray) -> tuple[float, float]:
    T, L = kymo.shape[:2]
    kymo2d = kymo if kymo.ndim == 2 else kymo[..., 0]
    filas = np.clip(np.round(segmento.puntos[:, 0]).astype(int), 0, T - 1)
    cols = np.clip(np.round(segmento.puntos[:, 1]).astype(int), 0, L - 1)
    valores = kymo2d[filas, cols].astype(float)
    return float(valores.mean()), float(valores.std())


def caracteristicas_segmentos(
    segmentos: list[Segmento], kymo: np.ndarray
) -> list[CaracteristicasSegmento]:
    """Una `CaracteristicasSegmento` por segmento, mismo orden que `segmentos`."""
    salida = []
    for seg in segmentos:
        pendiente = _pendiente_segmento(seg)
        p_salida = _pendiente_ventana(seg, "fin")
        p_entrada = _pendiente_ventana(seg, "ini")
        i_media, i_std = _intensidad_segmento(seg, kymo)
        salida.append(CaracteristicasSegmento(pendiente, p_salida, p_entrada, i_media, i_std))
    return salida


def pares_candidatos(
    segmentos: list[Segmento],
    *,
    max_gap_frames: float = 30.0,
    max_salto_px: float = 40.0,
) -> list[tuple[int, int]]:
    """Pares ordenados (i, j) donde el segmento j empieza despues de que i termina,
    dentro de un presupuesto de hueco temporal y espacial (SS4.1):

        0 <= t_ini(j) - t_fin(i) <= max_gap_frames
        |x_ini(j) - x_fin(i)|    <= max_salto_px

    Poda O(N^2) a algo chico y codifica un prior fisico (una particula no
    teletransporta). Requiere que `fusionar_fragmentos_paralelos` ya se haya
    corrido: esta regla asume que los segmentos de una misma particula son
    disjuntos en tiempo (si dos rieles paralelos de la misma particula quedaran sin
    fusionar, se solapan en `t` y esta regla los descarta en vez de encontrarlos)."""
    pares = []
    for i, a in enumerate(segmentos):
        for j, b in enumerate(segmentos):
            if i == j:
                continue
            gap = b.t_ini - a.t_fin
            if 0 <= gap <= max_gap_frames and abs(b.x_ini - a.x_fin) <= max_salto_px:
                pares.append((i, j))
    return pares


def _direccion_unitaria(dt: float, dx: float) -> tuple[float, float]:
    norma = np.hypot(dt, dx)
    return (dt / norma, dx / norma) if norma > 0 else (0.0, 0.0)


def geom_feats(
    a: Segmento, feat_a: CaracteristicasSegmento, b: Segmento, feat_b: CaracteristicasSegmento
) -> tuple[float, float, float, float]:
    """`(d_pos, cos_theta, dv, gap_frames)` -- las mismas cuatro cantidades que
    alimenta el costo clasico (SS4.2) y, sin modificar, el head de enlace de
    Stage 2 (SS5.3): la comparacion entre costo clasico y atencion no le da a esta
    ultima informacion geometrica que la primera no tenga.

    `cos_theta` = coseno entre la direccion de salida LOCAL de `a` (vector unitario
    `(1, pendiente_salida_a)`, ventana de las ultimas filas -- no el promedio de
    todo el segmento) y la direccion de conexion `a.fin -> b.ini`. Si ambos extremos
    coinciden exactamente (`gap=0` y `dx=0`, degenerado) se devuelve 1.0
    (alineacion perfecta) en vez de un vector nulo indefinido. `dv` compara la
    pendiente de salida de `a` contra la de ENTRADA de `b` -- ver
    `_pendiente_ventana`, evita que un segmento con pausa+corrida promedie a una
    velocidad que no coincide con ninguno de sus dos extremos."""
    gap = b.t_ini - a.t_fin
    d_pos = abs(a.x_fin - b.x_ini)
    p_salida = feat_a.pendiente_salida if np.isfinite(feat_a.pendiente_salida) else 0.0
    dir_salida = _direccion_unitaria(1.0, p_salida)
    dir_conexion = _direccion_unitaria(gap, b.x_ini - a.x_fin)
    if dir_conexion == (0.0, 0.0):
        cos_theta = 1.0
    else:
        cos_theta = float(np.dot(dir_salida, dir_conexion))
    if np.isfinite(feat_a.pendiente_salida) and np.isfinite(feat_b.pendiente_entrada):
        dv = abs(feat_a.pendiente_salida - feat_b.pendiente_entrada)
    else:
        dv = 0.0
    return d_pos, cos_theta, dv, gap


def costo_asociacion(
    a: Segmento,
    feat_a: CaracteristicasSegmento,
    b: Segmento,
    feat_b: CaracteristicasSegmento,
    pesos: dict[str, float],
) -> float:
    """`C(i,j) = a*d_pos + b*(1-cos_theta) + g*dv + d*dI + e*gap_frames` (SS4.2).
    `pesos` usa las mismas claves que la formula de la guia: `a, b, g, d, e`."""
    d_pos, cos_theta, dv, gap = geom_feats(a, feat_a, b, feat_b)
    dI = abs(feat_a.intensidad_media - feat_b.intensidad_media)
    return (
        pesos["a"] * d_pos
        + pesos["b"] * (1 - cos_theta)
        + pesos["g"] * dv
        + pesos["d"] * dI
        + pesos["e"] * gap
    )


def enlaces_verdaderos(
    segmentos: list[Segmento], asignaciones: list[AsignacionGT]
) -> set[tuple[int, int]]:
    """GT de enlace (SS5.4, reusado por el ajuste de pesos de SS4.2): `(i, j)` es un
    enlace verdadero si ambos segmentos fueron asignados al MISMO `particle_id`
    real (no -1) y ningun otro segmento de esa particula queda entre ellos en el
    tiempo. Calculado sobre TODOS los segmentos, no solo los que sobrevivieron la
    poda de `pares_candidatos` -- asi se puede medir cuantos se pierden (SS4.1)."""
    por_particula: dict[int, list[int]] = {}
    for idx, asign in enumerate(asignaciones):
        if asign.particle_id != -1:
            por_particula.setdefault(asign.particle_id, []).append(idx)
    verdaderos = set()
    for idxs in por_particula.values():
        idxs_ordenados = sorted(idxs, key=lambda k: segmentos[k].t_ini)
        for i, j in itertools.pairwise(idxs_ordenados):
            verdaderos.add((i, j))
    return verdaderos


def _seguir_cadenas(n: int, sucesor: dict[int, int]) -> list[list[int]]:
    """De un mapeo segmento -> su sucesor (a lo sumo uno) a una lista de cadenas
    (listas de indices de segmento en orden temporal) -- compartido por las dos
    variantes de decodificacion de SS4.3."""
    predecesores = set(sucesor.values())
    cadenas = []
    visitados = set()
    for i in range(n):
        if i in predecesores or i in visitados:
            continue
        cadena = [i]
        visitados.add(i)
        actual = i
        while actual in sucesor:
            actual = sucesor[actual]
            if actual in visitados:  # salvaguarda: no debería pasar (1 sucesor por nodo), pero corta ciclos
                break
            cadena.append(actual)
            visitados.add(actual)
        cadenas.append(cadena)
    return cadenas


def enlazar_por_costo(
    segmentos: list[Segmento],
    caracteristicas: list[CaracteristicasSegmento],
    pares: list[tuple[int, int]],
    pesos: dict[str, float],
    *,
    umbral: float,
) -> list[list[int]]:
    """Decodifica enlaces como matching bipartito GLOBAL (predecesor -> sucesor), a
    lo sumo un predecesor y un sucesor por segmento (SS4.3): matriz de costo N x N
    sobre `pares` (todo lo demas, costo centinela muy alto), `linear_sum_assignment`
    resuelve la asignacion completa, y se descartan los enlaces cuyo costo real
    supera `umbral` (evita forzar un enlace cuando ningun candidato es bueno --
    `linear_sum_assignment` por si solo siempre empareja TODO, aun con pares
    centinela). Ver `enlazar_por_costo_greedy` para la alternativa local que SS4.3
    tambien permite -- el optimo global puede, en principio, sacrificar el mejor
    candidato de un segmento para mejorar la suma total en otro lado; verificar
    contra la variante greedy antes de asumir que no pasa."""
    n = len(segmentos)
    CENTINELA = 1e6
    costos = np.full((n, n), CENTINELA)
    for i, j in pares:
        costos[i, j] = costo_asociacion(
            segmentos[i], caracteristicas[i], segmentos[j], caracteristicas[j], pesos
        )
    filas, cols = linear_sum_assignment(costos)
    sucesor = {int(i): int(j) for i, j in zip(filas, cols) if costos[i, j] <= umbral}
    return _seguir_cadenas(n, sucesor)


def enlazar_por_costo_greedy(
    segmentos: list[Segmento],
    caracteristicas: list[CaracteristicasSegmento],
    pares: list[tuple[int, int]],
    pesos: dict[str, float],
    *,
    umbral: float,
) -> list[list[int]]:
    """Alternativa greedy a `enlazar_por_costo` (SS4.3, ambas permitidas): ordena
    TODOS los pares candidatos por costo ascendente y acepta el primero que deje
    libres tanto el sucesor de `i` como el predecesor de `j`, sin buscar un optimo
    global. Evita por construccion el riesgo de que el matching bipartito global le
    "robe" a un segmento su unico buen candidato para mejorar la suma total en otro
    lado -- irrelevante para una estructura de cadenas casi-disjuntas como esta, y
    mas facil de auditar."""
    costos_pares = []
    for i, j in pares:
        c = costo_asociacion(segmentos[i], caracteristicas[i], segmentos[j], caracteristicas[j], pesos)
        if c <= umbral:
            costos_pares.append((c, i, j))
    costos_pares.sort(key=lambda t: t[0])

    sucesor: dict[int, int] = {}
    tiene_predecesor: set[int] = set()
    for _c, i, j in costos_pares:
        if i in sucesor or j in tiene_predecesor:
            continue
        sucesor[i] = j
        tiene_predecesor.add(j)
    return _seguir_cadenas(len(segmentos), sucesor)


def polilineas_desde_cadenas(
    cadenas: list[list[int]], segmentos: list[Segmento], kymo: np.ndarray
) -> list[pd.DataFrame]:
    """Cadenas de indices de segmento -> polilineas `(frame, col_subpixel)`, el
    formato que `ev.evaluar_trayectorias_polilineas` espera (SS4.3)."""
    shape = kymo.shape[:2]
    polilineas = []
    for cadena in cadenas:
        puntos = np.concatenate([segmentos[k].puntos for k in cadena], axis=0)
        puntos = puntos[np.argsort(puntos[:, 0], kind="stable")]
        poli = polilinea_desde_puntos(puntos, shape, kymo)
        if len(poli):
            polilineas.append(poli)
    return polilineas
