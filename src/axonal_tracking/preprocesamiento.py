"""Funciones de preprocesamiento de frames de microscopia.

Funciones puras: entrada array(s), salida array(s). Ninguna hace plot.

Pipeline tipico:
    1. uint16 -> RGB uint8 (para modelos visuales).
    2. Sustraccion de fondo espacial (apaga soma / fluorescencia difusa).
    3. Reduccion de ruido (shot-noise del sensor 16-bit).
    4. Opcional: sustraccion de fondo temporal (elimina objetos estaticos).
    5. Opcional: mascara del ROI del axon (filtra fuera del axon).

Las funciones se exportan individualmente y tambien via `pipeline_frame()`
que las encadena devolviendo todos los pasos intermedios.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


# ── Conversion 16-bit -> RGB uint8 ───────────────────────────────────────────

def frame_a_rgb_uint8(
    frame_u16: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    ignorar_ceros: bool = False,
) -> np.ndarray:
    """Convierte un frame uint16 monocanal a RGB uint8 con stretch de percentiles.

    Los modelos de vision (SAM, Grounding DINO, etc.) esperan uint8 RGB.
    El stretch evita perder contraste por outliers (pixeles saturados o muertos).

    Parametros:
        frame_u16: array 2D uint16 (alto, ancho).
        p_low, p_high: percentiles para el stretch (default 1-99).
        ignorar_ceros: si True, calcula los percentiles solo sobre pixeles != 0.
            Usar cuando el frame viene enmascarado (la mayoria de pixeles son 0):
            si se incluyen, el stretch se rompe y el contenido visible queda
            saturado en blanco uniforme.

    Devuelve:
        Array 3D uint8 (alto, ancho, 3).
    """
    if ignorar_ceros:
        muestra = frame_u16[frame_u16 > 0]
        if muestra.size == 0:
            return np.zeros((*frame_u16.shape, 3), dtype=np.uint8)
        lo, hi = np.percentile(muestra, [p_low, p_high])
    else:
        lo, hi = np.percentile(frame_u16, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1
    frame_8 = np.clip((frame_u16 - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return np.stack([frame_8] * 3, axis=-1)


# ── Sustraccion de fondo espacial ────────────────────────────────────────────

def sustraer_fondo_espacial(frame: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Resta el fondo "ancho" estimado con un gaussian filter de sigma grande.

    Idea: convolucionar el frame con un kernel gaussiano muy ancho da una
    estimacion del brillo "lento" (soma, fluorescencia difusa, gradientes
    de iluminacion). Restando esa estimacion, sobreviven solo los objetos
    pequenios y puntuales (vesiculas, debris).

    Parametros:
        frame: array 2D (uint16 o float).
        sigma: sigma del gaussian (en pixeles). Mas grande -> fondo mas "ancho"
            se conserva. Para vesiculas de ~3 px, sigma=20 funciona bien.

    Devuelve:
        Array float64 con el frame limpio (no se permite valores negativos).
    """
    f = frame.astype(np.float64)
    fondo = gaussian_filter(f, sigma=sigma)
    return np.clip(f - fondo, 0.0, None)


# ── Reduccion de ruido ───────────────────────────────────────────────────────

def reducir_ruido(
    frame: np.ndarray,
    metodo: Literal["gaussian", "mediana"] = "gaussian",
    sigma: float = 0.8,
    tamano_mediana: int = 3,
) -> np.ndarray:
    """Suaviza el ruido shot-noise del sensor 16-bit.

    Dos opciones:
    - **gaussian**: kernel gaussiano leve (sigma~0.8). Suaviza ruido sin
      borrar blobs reales. Adecuado para ruido aproximadamente gaussiano.
    - **mediana**: filtro de mediana 3x3. Especialmente bueno para "sal y
      pimienta" (pixeles aislados muy brillantes/oscuros).

    Parametros:
        frame: array 2D.
        metodo: "gaussian" o "mediana".
        sigma: sigma para metodo gaussian.
        tamano_mediana: tamanio del kernel para metodo mediana (debe ser impar).

    Devuelve:
        Array del mismo shape y dtype.
    """
    if metodo == "gaussian":
        return gaussian_filter(frame, sigma=sigma)
    if metodo == "mediana":
        return median_filter(frame, size=tamano_mediana)
    raise ValueError(f"metodo desconocido: {metodo!r}; usar 'gaussian' o 'mediana'")


# ── Sustraccion de fondo temporal ────────────────────────────────────────────

def calcular_fondo_temporal(
    video: np.ndarray,
    metodo: Literal["mediana", "media"] = "mediana",
) -> np.ndarray:
    """Calcula el fondo estatico de un video colapsando el eje temporal.

    El resultado es una imagen 2D que representa lo que esta "siempre" en cada
    pixel — el soma, agregados pegados, fluorescencia difusa. Es lo que despues
    se resta de cada frame para dejar solo lo que se mueve.

    Util cuando queres calcular el fondo UNA VEZ y aplicarlo a varios frames
    del mismo video (mas eficiente que llamar a `sustraer_fondo_temporal`
    en cada frame).

    Parametros:
        video: array 3D (T, H, W).
        metodo: "mediana" (default, mas robusta) o "media".

    Devuelve:
        Array 2D (H, W) float64 con el fondo estatico.
    """
    if video.ndim != 3:
        raise ValueError(f"video debe ser 3D (T, H, W); recibido shape {video.shape}")
    v = video.astype(np.float64)
    if metodo == "mediana":
        return np.median(v, axis=0)
    if metodo == "media":
        return v.mean(axis=0)
    raise ValueError(f"metodo desconocido: {metodo!r}; usar 'mediana' o 'media'")


def sustraer_fondo_temporal(
    video: np.ndarray,
    metodo: Literal["mediana", "media"] = "mediana",
) -> np.ndarray:
    """Estima el fondo como agregado temporal y lo resta de cada frame.

    Es la version "todo en uno" de `calcular_fondo_temporal` + resta. Devuelve
    un video 3D limpio (cada frame sin el fondo estatico).

    Las vesiculas se MUEVEN entre frames, asi que al colapsar el video con una
    mediana temporal las vesiculas casi desaparecen y queda solo lo estatico
    (soma, agregados pegados, fluorescencia de fondo). Restando esa estimacion,
    sobreviven los objetos en movimiento.

    Es el complemento de `sustraer_fondo_espacial`: la espacial mata el soma y
    los gradientes; la temporal mata los agregados estaticos.

    Parametros:
        video: array 3D (T, H, W).
        metodo: "mediana" (default, mas robusta) o "media".

    Devuelve:
        Array 3D float64 (T, H, W) con cada frame limpio.
    """
    fondo = calcular_fondo_temporal(video, metodo=metodo)
    v = video.astype(np.float64)
    return np.clip(v - fondo[None], 0.0, None)


# ── Mascara del ROI del axon ─────────────────────────────────────────────────

def cargar_mascara_roi(
    ruta_roi: Path,
    shape: tuple[int, int],
    ancho_px: int = 8,
) -> np.ndarray | None:
    """Lee un archivo .roi de ImageJ y construye una mascara binaria.

    Los .roi del lab contienen una polilinea (lista de puntos x, y) que marca
    el eje del axon. Aca la "engordamos" cierto ancho en pixeles para que la
    mascara cubra el axon entero (no solo la linea infinitesimal).

    Parametros:
        ruta_roi: path al archivo .roi.
        shape: (alto, ancho) del frame al que aplicar la mascara.
        ancho_px: cuanto engordar la polilinea (radio en pixeles).

    Devuelve:
        Array 2D bool (alto, ancho) con True donde esta el axon, False afuera.
        Devuelve None si el archivo no existe.
    """
    ruta_roi = Path(ruta_roi)
    if not ruta_roi.exists():
        return None

    from read_roi import read_roi_file
    from skimage.draw import line, disk

    rois = read_roi_file(str(ruta_roi))
    # read_roi devuelve dict {nombre: {...metadata...}}. Usamos el primero.
    roi = next(iter(rois.values()))

    xs = roi.get("x", [])
    ys = roi.get("y", [])
    if not xs or not ys or len(xs) != len(ys):
        return None

    mascara = np.zeros(shape, dtype=bool)
    # Dibujamos cada segmento de la polilinea y engordamos con un disco.
    for i in range(len(xs) - 1):
        rr, cc = line(int(ys[i]), int(xs[i]), int(ys[i + 1]), int(xs[i + 1]))
        for r, c in zip(rr, cc):
            dr, dc = disk((r, c), ancho_px, shape=shape)
            mascara[dr, dc] = True
    return mascara


def aplicar_mascara(frame: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Pone a 0 todos los pixeles del frame que estan fuera de la mascara.

    Parametros:
        frame: array 2D o 3D (con canales al final).
        mascara: array 2D bool del mismo (alto, ancho).

    Devuelve:
        Array del mismo shape y dtype, con el fondo en 0.
    """
    if frame.ndim == 2:
        return np.where(mascara, frame, 0)
    # 3D (H, W, C): broadcast mascara
    return np.where(mascara[..., None], frame, 0)


# ── Pipeline integrado ───────────────────────────────────────────────────────

def pipeline_frame(
    frame_u16: np.ndarray,
    *,
    fondo_temporal: np.ndarray | None = None,
    sigma_fondo: float = 20.0,
    sigma_ruido: float = 0.8,
    ruta_roi: Path | None = None,
    ancho_roi_px: int = 8,
) -> dict:
    """Aplica el pipeline completo a un frame y devuelve todos los pasos intermedios.

    Pasos en orden:
      1. crudo                  -> frame uint16 original
      2. sin_fondo_temporal     -> tras restar el fondo temporal (None si no se paso)
      3. sin_fondo              -> tras sustraer fondo espacial
      4. sin_ruido              -> tras filtro gaussiano leve
      5. enmascarado            -> tras aplicar ROI (igual a sin_ruido si no hay ROI)
      6. rgb_uint8              -> conversion final para SAM/detectores
      7. mascara_roi            -> la mascara binaria usada (o None)

    Parametros:
        frame_u16: frame crudo (2D uint16).
        fondo_temporal: array 2D (H, W) con el fondo estatico del video, calculado
            con `calcular_fondo_temporal()`. Si se pasa, se resta antes de la
            sustraccion espacial — elimina objetos estaticos (agregados, debris).
            Pasalo si vas a procesar varios frames del mismo video (se calcula
            una vez y se reutiliza).
        sigma_fondo: sigma para gaussian de sustraccion de fondo espacial.
        sigma_ruido: sigma para gaussian de reduccion de ruido.
        ruta_roi: path al .roi del axon (opcional). Si es None, no se enmascara.
        ancho_roi_px: ancho de la mascara del axon en pixeles.

    Devuelve:
        Dict con todos los pasos intermedios.
    """
    crudo = frame_u16

    # 1. Sustraccion temporal (opcional, si se paso fondo_temporal).
    if fondo_temporal is not None:
        sin_fondo_temporal = np.clip(
            crudo.astype(np.float64) - fondo_temporal, 0.0, None
        )
        entrada_espacial = sin_fondo_temporal
    else:
        sin_fondo_temporal = None
        entrada_espacial = crudo

    # 2. Sustraccion espacial.
    sin_fondo = sustraer_fondo_espacial(entrada_espacial, sigma=sigma_fondo)

    # 3. Ruido.
    sin_ruido = reducir_ruido(sin_fondo, metodo="gaussian", sigma=sigma_ruido)

    # 4. Mascara ROI (opcional).
    mascara = None
    enmascarado = sin_ruido
    if ruta_roi is not None:
        mascara = cargar_mascara_roi(ruta_roi, shape=crudo.shape, ancho_px=ancho_roi_px)
        if mascara is not None:
            enmascarado = aplicar_mascara(sin_ruido, mascara)

    # 5. RGB uint8 para SAM/detectores. Si hubo enmascarado, ignoramos los ceros
    # del fondo al calcular el stretch — si no, el axon se satura en blanco
    # uniforme y se pierde el contraste interno.
    rgb_uint8 = frame_a_rgb_uint8(enmascarado, ignorar_ceros=(mascara is not None))

    return {
        "crudo": crudo,
        "sin_fondo_temporal": sin_fondo_temporal,
        "sin_fondo": sin_fondo,
        "sin_ruido": sin_ruido,
        "enmascarado": enmascarado,
        "rgb_uint8": rgb_uint8,
        "mascara_roi": mascara,
    }


def bbox_de_mascara(mascara: np.ndarray, padding: int = 20) -> tuple[int, int, int, int]:
    """Calcula el bounding box (y0, y1, x0, x1) de una mascara binaria, con padding.

    Util para recortar el frame al area util (la del axon) y descartar todo el
    fondo enmascarado. Reduce memoria y compute al alimentar a los modelos
    sin perder resolucion de las particulas chicas.

    Parametros:
        mascara: array 2D bool.
        padding: pixeles a expandir el bbox en cada direccion (clipea al frame).

    Devuelve:
        (y0, y1, x0, x1) con y0/x0 inclusivos y y1/x1 exclusivos (estilo numpy).
        Si la mascara esta vacia, devuelve el frame entero.
    """
    H, W = mascara.shape
    if not mascara.any():
        return (0, H, 0, W)
    ys, xs = np.where(mascara)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(H, int(ys.max()) + 1 + padding)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(W, int(xs.max()) + 1 + padding)
    return (y0, y1, x0, x1)


def recortar_a_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Recorta un array (2D, 3D o 4D) a la bbox (y0, y1, x0, x1).

    Soporta:
      - (H, W)       — un frame en escala de grises
      - (H, W, C)    — un frame RGB
      - (T, H, W)    — un video monocanal
      - (T, H, W, C) — un video RGB

    El recorte siempre actua sobre los ejes H y W (los espaciales).
    """
    y0, y1, x0, x1 = bbox
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1]
    if arr.ndim == 3:
        # Distinguir (H, W, C) vs (T, H, W). Asumimos que (H, W, C) tiene el
        # ultimo eje <= 4 (canales RGB/RGBA) y (T, H, W) tiene mas.
        if arr.shape[-1] <= 4:
            return arr[y0:y1, x0:x1, :]
        return arr[:, y0:y1, x0:x1]
    if arr.ndim == 4:
        return arr[:, y0:y1, x0:x1, :]
    raise ValueError(f"shape no soportado: {arr.shape}")


def recortar_pasos_pipeline(
    pasos: dict,
    padding: int = 20,
) -> dict:
    """Recorta todos los arrays del dict de `pipeline_frame()` al bbox del ROI.

    Si no hay mascara en `pasos`, devuelve el dict tal cual. Si la hay, recorta
    cada array al bbox de la mascara con el padding pedido. El bbox queda
    guardado como `pasos["bbox"]` para mapear coordenadas a la imagen original.

    Parametros:
        pasos: dict devuelto por `pipeline_frame()`.
        padding: pixeles de margen alrededor del bbox del ROI.

    Devuelve:
        Dict con los mismos keys, todos los arrays recortados, + `bbox`.
    """
    mascara = pasos.get("mascara_roi")
    if mascara is None:
        return {**pasos, "bbox": None}

    bbox = bbox_de_mascara(mascara, padding=padding)
    out = {}
    for k, v in pasos.items():
        if v is None or not isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = recortar_a_bbox(v, bbox)
    out["bbox"] = bbox
    return out


def pipeline_video(
    video: np.ndarray,
    *,
    fondo_temporal: np.ndarray | None = None,
    sigma_fondo: float = 20.0,
    sigma_ruido: float = 0.8,
    ruta_roi: Path | None = None,
    ancho_roi_px: int = 8,
) -> np.ndarray:
    """Aplica el pipeline completo a cada frame del video.

    Es equivalente a llamar `pipeline_frame()` en cada frame, pero mucho mas
    eficiente: calcula la mascara del ROI y el fondo temporal UNA sola vez y
    los reutiliza para todos los frames.

    Parametros:
        video: array 3D (T, H, W) uint16.
        fondo_temporal: array 2D (H, W) con el fondo estatico. Si es None, se
            calcula desde el mismo video como mediana temporal (recomendado).
        sigma_fondo, sigma_ruido: ver `pipeline_frame`.
        ruta_roi: path al .roi del axon (opcional).
        ancho_roi_px: ancho de la mascara del axon.

    Devuelve:
        Array 4D (T, H, W, 3) uint8 con cada frame procesado y listo para SAM.

    Memoria: 86 frames * 1148 * 1279 * 3 bytes = ~378 MB para los videos del lab.
    """
    if video.ndim != 3:
        raise ValueError(f"video debe ser 3D (T, H, W); recibido {video.shape}")

    # Calcular fondo temporal una sola vez si no se paso.
    if fondo_temporal is None:
        fondo_temporal = calcular_fondo_temporal(video, metodo="mediana")

    # Cargar la mascara del ROI una sola vez si se paso.
    H, W = video.shape[1], video.shape[2]
    mascara = None
    if ruta_roi is not None:
        mascara = cargar_mascara_roi(ruta_roi, shape=(H, W), ancho_px=ancho_roi_px)

    T = video.shape[0]
    salida = np.empty((T, H, W, 3), dtype=np.uint8)

    for t in range(T):
        frame_u16 = video[t]

        # Sustraccion temporal
        sin_fondo_t = np.clip(frame_u16.astype(np.float64) - fondo_temporal, 0.0, None)

        # Espacial + ruido
        sin_fondo = sustraer_fondo_espacial(sin_fondo_t, sigma=sigma_fondo)
        sin_ruido = reducir_ruido(sin_fondo, metodo="gaussian", sigma=sigma_ruido)

        # Mascara (opcional)
        if mascara is not None:
            sin_ruido = aplicar_mascara(sin_ruido, mascara)

        # RGB uint8
        salida[t] = frame_a_rgb_uint8(sin_ruido, ignorar_ceros=(mascara is not None))

    return salida
