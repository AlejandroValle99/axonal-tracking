"""Generacion de kymografos a partir de un video + ROI del axon.

Un kymografo es una imagen 2D (T x L):
  - Eje X (L): posicion a lo largo del eje del axon
  - Eje Y (T): tiempo (un row por frame)

Se construye proyectando los valores del video sobre la polilinea del ROI.
Vesiculas que se mueven aparecen como **lineas diagonales** (slope = velocidad).
Vesiculas estaticas aparecen como **lineas verticales**.

La gran ventaja vs detectar particulas frame por frame: el kymografo integra
informacion temporal, asi una vesicula que es indistinguible del ruido en un
solo frame se vuelve obvia como traza diagonal coherente.

Equivale al kymografo que el lab genera manualmente con ImageJ (Multi Kymograph
o KymographBuilder). Los archivos `Kymograph_NNN.tif` del dataset son la
referencia ground-truth para validar contra esta implementacion.
"""

from pathlib import Path
from typing import Literal

import numpy as np

from .preprocesamiento import cargar_mascara_roi  # noqa: F401 (re-export friendly)


def _cargar_polilinea_roi(ruta_roi: Path) -> np.ndarray:
    """Lee un .roi de ImageJ y devuelve (N, 2) array [(x, y), ...]."""
    from read_roi import read_roi_file

    rois = read_roi_file(str(ruta_roi))
    roi = next(iter(rois.values()))
    xs = roi.get("x", [])
    ys = roi.get("y", [])
    if not xs or not ys or len(xs) != len(ys):
        raise ValueError(f"ROI invalido o sin puntos: {ruta_roi}")
    return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])


def _interpolar_polilinea(puntos: np.ndarray, paso_px: float = 1.0) -> np.ndarray:
    """Re-muestrea una polilinea para tener puntos espaciados ~paso_px en arc-length.

    El ROI del lab puede tener vertices muy espaciados (10-20 px entre puntos).
    Para un kymografo necesitamos un punto por columna, asi que interpolamos
    linealmente entre vertices.

    Parametros:
        puntos: array (N, 2) con [x, y] de cada vertice.
        paso_px: distancia objetivo entre puntos consecutivos (default 1 px).

    Devuelve:
        Array (M, 2) con la polilinea densamente muestreada.
    """
    if len(puntos) < 2:
        return puntos

    # Distancias acumuladas a lo largo de la polilinea
    diffs = np.diff(puntos, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    largo_total = arc[-1]
    if largo_total <= 0:
        return puntos

    # Re-muestreo uniforme en arc-length
    arc_nuevo = np.arange(0, largo_total + paso_px, paso_px)
    xs_nuevo = np.interp(arc_nuevo, arc, puntos[:, 0])
    ys_nuevo = np.interp(arc_nuevo, arc, puntos[:, 1])
    return np.column_stack([xs_nuevo, ys_nuevo])


def _normales_a_polilinea(polilinea: np.ndarray) -> np.ndarray:
    """Calcula vector normal unitario en cada punto (perpendicular al axon).

    Para cada punto i, la direccion del axon se estima como la tangente entre
    el punto anterior y el siguiente. La normal es la rotacion 90 grados.

    Parametros:
        polilinea: array (M, 2) con [x, y].

    Devuelve:
        Array (M, 2) con vectores normales unitarios.
    """
    M = len(polilinea)
    tangentes = np.zeros_like(polilinea)
    if M < 2:
        return tangentes
    # Diferencia central (forward en bordes)
    tangentes[1:-1] = polilinea[2:] - polilinea[:-2]
    tangentes[0] = polilinea[1] - polilinea[0]
    tangentes[-1] = polilinea[-1] - polilinea[-2]

    # Normalizar
    norm = np.linalg.norm(tangentes, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    tangentes /= norm

    # Rotacion 90: (x, y) -> (-y, x)
    normales = np.column_stack([-tangentes[:, 1], tangentes[:, 0]])
    return normales


def _sampleo_perpendicular(
    frame: np.ndarray,
    polilinea: np.ndarray,
    normales: np.ndarray,
    ancho_perpendicular: int = 3,
    metodo: Literal["mean", "max"] = "max",
) -> np.ndarray:
    """Para cada punto de la polilinea, agrega valores perpendiculares al axon.

    En cada punto (x, y), se samplea una banda de pixeles perpendicular a
    la direccion del axon, de ancho `ancho_perpendicular` (en pixeles).
    Eso compensa el grosor real del axon y reduce ruido.

    Parametros:
        frame: array 2D (H, W).
        polilinea: array (M, 2).
        normales: array (M, 2) con vectores normales unitarios.
        ancho_perpendicular: distancia (en pixeles) a cada lado del eje.
        metodo: "mean" promedia, "max" toma el pico (mejor para particulas brillantes).

    Devuelve:
        Array 1D (M,) con un valor por punto de la polilinea.
    """
    H, W = frame.shape
    M = len(polilinea)
    offsets = np.arange(-ancho_perpendicular, ancho_perpendicular + 1)

    # Generamos coordenadas (M, len(offsets)) muestreando perpendicular
    xs = polilinea[:, 0:1] + normales[:, 0:1] * offsets[None, :]
    ys = polilinea[:, 1:2] + normales[:, 1:2] * offsets[None, :]

    # Clip a los bordes del frame y redondeo a int
    xs_i = np.clip(np.round(xs).astype(int), 0, W - 1)
    ys_i = np.clip(np.round(ys).astype(int), 0, H - 1)

    # Sampleamos: (M, K)
    valores = frame[ys_i, xs_i]

    if metodo == "max":
        return valores.max(axis=1)
    return valores.mean(axis=1)


def generar_kimografo(
    video: np.ndarray,
    ruta_roi: Path,
    *,
    paso_px: float = 1.0,
    ancho_perpendicular: int = 3,
    metodo: Literal["mean", "max"] = "max",
) -> dict:
    """Genera un kymografo de un video usando la polilinea del ROI del axon.

    Parametros:
        video: array 3D (T, H, W). Puede ser uint16 crudo o float ya procesado.
        ruta_roi: path al archivo .roi con la polilinea del axon.
        paso_px: distancia entre puntos muestreados a lo largo del axon
            (default 1 px = maxima resolucion espacial).
        ancho_perpendicular: distancia (px) a cada lado del eje para promediar.
            Mas grande = mas SNR pero menos resolucion lateral. Default 3
            (banda de 7 px de ancho — adecuado para axones tipicos).
        metodo: "max" (recomendado para particulas brillantes) o "mean".

    Devuelve:
        Dict con:
          - 'kimografo':   array 2D (T, L) — la imagen del kymografo
          - 'polilinea':   array (L, 2) interpolada con [(x, y), ...]
          - 'longitud_um': float, longitud total del axon en micrometros
                           (usando PIXEL_SIZE_UM del modulo parametros)
    """
    if video.ndim != 3:
        raise ValueError(f"video debe ser 3D (T, H, W); recibido {video.shape}")

    # 1. Cargar y densificar la polilinea
    puntos_originales = _cargar_polilinea_roi(Path(ruta_roi))
    polilinea = _interpolar_polilinea(puntos_originales, paso_px=paso_px)
    normales = _normales_a_polilinea(polilinea)

    # 2. Sampleo frame por frame
    T = video.shape[0]
    L = len(polilinea)
    kimografo = np.empty((T, L), dtype=np.float64)
    for t in range(T):
        kimografo[t] = _sampleo_perpendicular(
            video[t], polilinea, normales,
            ancho_perpendicular=ancho_perpendicular,
            metodo=metodo,
        )

    # 3. Longitud del axon en micrometros
    from .parametros import PIXEL_SIZE_UM

    diffs = np.diff(polilinea, axis=0)
    largo_px = np.sqrt((diffs ** 2).sum(axis=1)).sum()
    longitud_um = float(largo_px * PIXEL_SIZE_UM)

    return {
        "kimografo": kimografo,
        "polilinea": polilinea,
        "longitud_um": longitud_um,
    }


# ── Rectificacion: "desenrollar" el axon en un strip horizontal ──────────────
#
# Comparte la logica de muestreo perpendicular del kymografo pero CONSERVA la
# dimension perpendicular en lugar de colapsarla. Resultado: por cada frame,
# un strip 2D (alto = banda perpendicular, ancho = longitud del axon).
#
# Util para alimentar a los modelos con SOLO la zona util — eliminando
# practicamente todos los pixeles muertos que quedan tras un simple bbox crop.


def _coordenadas_rectificacion(
    polilinea: np.ndarray,
    normales: np.ndarray,
    ancho_banda: int,
    shape_frame: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula las coordenadas (xs, ys) para muestrear el strip rectificado.

    Devuelve dos arrays (M, K) con K = 2*ancho_banda + 1 (banda perpendicular).
    Las coordenadas estan clipeadas a los bordes del frame.
    """
    H, W = shape_frame
    offsets = np.arange(-ancho_banda, ancho_banda + 1)
    xs = polilinea[:, 0:1] + normales[:, 0:1] * offsets[None, :]
    ys = polilinea[:, 1:2] + normales[:, 1:2] * offsets[None, :]
    xs_i = np.clip(np.round(xs).astype(int), 0, W - 1)
    ys_i = np.clip(np.round(ys).astype(int), 0, H - 1)
    return xs_i, ys_i


def rectificar_frame(
    frame: np.ndarray,
    ruta_roi: Path,
    *,
    ancho_banda: int = 10,
    paso_px: float = 1.0,
) -> dict:
    """Rectifica un frame: lo 'desenrolla' alrededor de la polilinea del axon.

    El frame se muestrea a lo largo de la polilinea, tomando una banda
    perpendicular de `2*ancho_banda + 1` pixeles a cada lado. El resultado es
    un strip 2D donde el axon corre horizontalmente y casi todos los pixeles
    son utiles (no hay zonas muertas).

    Parametros:
        frame: array 2D (H, W) — uint16 o float.
        ruta_roi: path al .roi con la polilinea del axon.
        ancho_banda: pixeles perpendiculares a cada lado del eje (default 10
            -> strip de 21 px de alto).
        paso_px: densidad de muestreo a lo largo del eje (default 1 px).

    Devuelve:
        dict con:
          - 'strip': array 2D (2*ancho_banda+1, L) con el axon horizontal
          - 'polilinea': (L, 2) puntos del eje en coords del frame original
          - 'normales':  (L, 2) vectores normales por punto
          - 'ancho_banda': el ancho usado (para mapear coords si hace falta)
    """
    if frame.ndim != 2:
        raise ValueError(f"frame debe ser 2D; recibido {frame.shape}")

    polilinea_orig = _cargar_polilinea_roi(Path(ruta_roi))
    polilinea = _interpolar_polilinea(polilinea_orig, paso_px=paso_px)
    normales = _normales_a_polilinea(polilinea)
    xs_i, ys_i = _coordenadas_rectificacion(polilinea, normales, ancho_banda, frame.shape)

    # Muestreamos: el resultado es (M, K). Transponemos a (K, M) para que el
    # axon corra horizontalmente y la banda perpendicular sea el eje vertical.
    strip = frame[ys_i, xs_i].T

    return {
        "strip": strip,
        "polilinea": polilinea,
        "normales": normales,
        "ancho_banda": ancho_banda,
    }


def rectificar_video(
    video: np.ndarray,
    ruta_roi: Path,
    *,
    ancho_banda: int = 10,
    paso_px: float = 1.0,
) -> dict:
    """Rectifica cada frame de un video 3D (T, H, W) al strip del axon.

    Mucho mas eficiente que llamar `rectificar_frame` por cada frame —
    calcula las coordenadas de muestreo UNA sola vez y las reutiliza.

    Parametros:
        video: array 3D (T, H, W) — uint16 o float.
        ruta_roi: path al .roi con la polilinea del axon.
        ancho_banda: pixeles perpendiculares a cada lado del eje.
        paso_px: densidad de muestreo a lo largo del eje.

    Devuelve:
        dict con:
          - 'video_strip': array 3D (T, 2*ancho_banda+1, L)
          - 'polilinea': (L, 2) puntos del eje
          - 'normales':  (L, 2) normales unitarias
          - 'ancho_banda': el ancho usado
    """
    if video.ndim != 3:
        raise ValueError(f"video debe ser 3D (T, H, W); recibido {video.shape}")

    polilinea_orig = _cargar_polilinea_roi(Path(ruta_roi))
    polilinea = _interpolar_polilinea(polilinea_orig, paso_px=paso_px)
    normales = _normales_a_polilinea(polilinea)
    xs_i, ys_i = _coordenadas_rectificacion(
        polilinea, normales, ancho_banda, (video.shape[1], video.shape[2])
    )

    T = video.shape[0]
    L = len(polilinea)
    K = 2 * ancho_banda + 1
    salida = np.empty((T, K, L), dtype=video.dtype)
    for t in range(T):
        salida[t] = video[t][ys_i, xs_i].T

    return {
        "video_strip": salida,
        "polilinea": polilinea,
        "normales": normales,
        "ancho_banda": ancho_banda,
    }


def mapear_coords_strip_a_original(
    coords_strip: np.ndarray,
    polilinea: np.ndarray,
    normales: np.ndarray,
    ancho_banda: int,
) -> np.ndarray:
    """Mapea coordenadas (x_strip, y_strip) del strip rectificado al frame original.

    En el strip:
      - x_strip = posicion a lo largo del axon (indice en la polilinea)
      - y_strip = perpendicular al axon (0..2*ancho_banda)

    Coordenadas en el frame original:
      pos_original = polilinea[x_strip] + (y_strip - ancho_banda) * normales[x_strip]

    Parametros:
        coords_strip: array (N, 2) con [(x_strip, y_strip), ...].
        polilinea, normales, ancho_banda: salida de `rectificar_frame/video`.

    Devuelve:
        Array (N, 2) con [(x_orig, y_orig), ...].
    """
    coords_strip = np.asarray(coords_strip, dtype=float)
    if coords_strip.ndim != 2 or coords_strip.shape[1] != 2:
        raise ValueError(f"coords_strip debe ser (N, 2); recibido {coords_strip.shape}")

    xs_strip = np.clip(coords_strip[:, 0].astype(int), 0, len(polilinea) - 1)
    ys_strip = coords_strip[:, 1]
    offset_perpendicular = ys_strip - ancho_banda  # signed offset

    base = polilinea[xs_strip]           # (N, 2)
    shift = normales[xs_strip] * offset_perpendicular[:, None]  # (N, 2)
    return base + shift
