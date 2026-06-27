"""Configuracion compartida entre notebooks.

El notebook 01 escribe `data/configuracion_actual.yaml` con la seleccion de
video, rango de frames y modo de procesamiento. Los notebooks 02/03/04 leen
ese archivo y aplican el preprocesamiento indicado via `aplicar_modo_frame` /
`aplicar_modo_video`.

Modos soportados:
    - "rgb_crudo"    : solo conversion uint16 -> RGB uint8, sin restar fondo
                       ni ruido. Para ver que devuelven los modelos sobre la
                       senal cruda.
    - "completo"     : pipeline completo de preprocesamiento (estado original
                       del proyecto): fondo temporal + espacial + ruido + ROI.
    - "configurable" : pipeline con etapas activables/desactivables individualmente
                       (etapas.fondo_temporal, etapas.fondo_espacial, etapas.ruido,
                       etapas.roi).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import preprocesamiento as pp


# Ruta por defecto del YAML, relativa a la raiz del repo (src/axonal_tracking/configuracion.py
# -> parents[2] = raiz del repo).
RUTA_CONFIG_DEFECTO = Path(__file__).resolve().parents[2] / "data" / "configuracion_actual.yaml"


CONFIG_DEFECTO: dict = {
    "video": {
        "ruta": "/Users/alejandrovalle/Desktop/Videos-Kymos-experimental data/Ex/Movie_674.vsi",
        "roi": "/Users/alejandrovalle/Desktop/Videos-Kymos-experimental data/Ex/674.roi",
    },
    "frames": {
        "inicio": 0,
        "fin": None,
        "demo": 0,
    },
    "procesamiento": {
        "modo": "completo",
        "etapas": {
            "fondo_temporal": True,
            "fondo_espacial": True,
            "ruido": True,
            "roi": True,
        },
        "parametros": {
            "sigma_fondo": 20.0,
            "sigma_ruido": 1.0,
            # Como se limpia el fondo espacial + ruido:
            #   "dog"         -> pasa-banda Diferencia de Gaussianas (1 pasada,
            #                    menos grano; recomendado).
            #   "resta_gauss" -> flujo clasico en dos pasos (resta gaussiana +
            #                    filtro de ruido). Conservado para comparacion.
            "metodo_espacial": "dog",
            "ancho_roi_px": 8,
            "n_frames_temporal": 10,
        },
        # Recorta el output al bounding box del ROI (con padding) — los modelos
        # reciben solo la zona util del axon en lugar de un frame 1148x1279 con
        # ~95% fondo negro. NO reescala, solo recorta: los pixeles utiles
        # mantienen 1:1 para no perder resolucion de particulas chicas.
        "recorte_roi": {
            "habilitado": True,
            "padding_px": 20,
        },
        # Rectifica el axon en un strip horizontal: "desenrolla" la polilinea
        # del ROI y reduce a ~2% del original sin pixeles muertos. Si esta
        # habilitada, sobrescribe `recorte_roi` (no tiene sentido aplicar
        # ambos). El strip es de (2*ancho_banda+1) x L_axon pixeles.
        "rectificacion": {
            "habilitada": False,
            "ancho_banda_px": 10,  # perpendicular a cada lado del eje
            "paso_px": 1.0,        # densidad a lo largo del eje
        },
    },
}


def _convertir_paths(config: dict) -> dict:
    """Convierte los campos de ruta del dict a `Path`. Mutates in place y devuelve."""
    video = config.get("video", {})
    if video.get("ruta") is not None:
        video["ruta"] = Path(video["ruta"])
    if video.get("roi") is not None:
        video["roi"] = Path(video["roi"])
    return config


def cargar_configuracion(ruta: Path | None = None) -> dict:
    """Carga la configuracion desde YAML.

    Si el archivo no existe, escribe la configuracion por defecto (`CONFIG_DEFECTO`)
    en `ruta` y la devuelve. Asi los notebooks downstream funcionan en seco
    aunque el notebook 01 nunca se haya ejecutado.

    Las rutas (`video.ruta`, `video.roi`) vienen como `Path`.
    """
    ruta = Path(ruta) if ruta is not None else RUTA_CONFIG_DEFECTO

    if not ruta.exists():
        guardar_configuracion(CONFIG_DEFECTO, ruta)
        return _convertir_paths({
            "video": dict(CONFIG_DEFECTO["video"]),
            "frames": dict(CONFIG_DEFECTO["frames"]),
            "procesamiento": {
                "modo": CONFIG_DEFECTO["procesamiento"]["modo"],
                "etapas": dict(CONFIG_DEFECTO["procesamiento"]["etapas"]),
                "parametros": dict(CONFIG_DEFECTO["procesamiento"]["parametros"]),
            },
        })

    with ruta.open("r") as f:
        config = yaml.safe_load(f) or {}
    return _convertir_paths(config)


def guardar_configuracion(config: dict, ruta: Path | None = None) -> Path:
    """Guarda la configuracion como YAML legible. Devuelve la ruta usada."""
    ruta = Path(ruta) if ruta is not None else RUTA_CONFIG_DEFECTO
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Path -> str para serializacion limpia.
    proc_in = config["procesamiento"]
    serializable_proc = {
        "modo": proc_in["modo"],
        "etapas": dict(proc_in["etapas"]),
        "parametros": dict(proc_in["parametros"]),
    }
    if "recorte_roi" in proc_in:
        serializable_proc["recorte_roi"] = dict(proc_in["recorte_roi"])
    if "rectificacion" in proc_in:
        serializable_proc["rectificacion"] = dict(proc_in["rectificacion"])
    serializable = {
        "video": {
            "ruta": str(config["video"]["ruta"]) if config["video"].get("ruta") is not None else None,
            "roi": str(config["video"]["roi"]) if config["video"].get("roi") is not None else None,
        },
        "frames": dict(config["frames"]),
        "procesamiento": serializable_proc,
    }
    with ruta.open("w") as f:
        yaml.safe_dump(serializable, f, sort_keys=False, allow_unicode=True)
    return ruta


# ── Aplicacion del modo a frame/video ────────────────────────────────────────


def _ruta_roi_si_activada(config: dict) -> Path | None:
    """Devuelve la ruta del ROI si el modo lo usa, o None."""
    proc = config["procesamiento"]
    modo = proc["modo"]
    if modo == "rgb_crudo":
        return None
    if modo == "configurable" and not proc["etapas"].get("roi", True):
        return None
    roi = config["video"].get("roi")
    if roi is None:
        return None
    roi = Path(roi)
    return roi if roi.exists() else None


def _metodo_espacial_config(config: dict) -> str:
    """Lee `procesamiento.parametros.metodo_espacial` ('dog' | 'resta_gauss').

    Default "dog" (recomendado). Backward compat: configs viejos sin la clave
    tambien usan "dog".
    """
    params = config.get("procesamiento", {}).get("parametros", {})
    return str(params.get("metodo_espacial", "dog"))


def _recorte_roi_config(config: dict) -> tuple[bool, int]:
    """Lee (habilitado, padding_px) de la seccion `recorte_roi` del config.

    Si la seccion no existe (configs viejos), devuelve (False, 0) para
    mantener backward compat.
    """
    proc = config.get("procesamiento", {})
    recorte = proc.get("recorte_roi") or {}
    return bool(recorte.get("habilitado", False)), int(recorte.get("padding_px", 20))


def _rectificacion_config(config: dict) -> tuple[bool, int, float]:
    """Lee (habilitada, ancho_banda_px, paso_px) de `rectificacion`.

    Si la seccion no existe (configs viejos), devuelve (False, 10, 1.0) para
    backward compat.
    """
    proc = config.get("procesamiento", {})
    rect = proc.get("rectificacion") or {}
    return (
        bool(rect.get("habilitada", False)),
        int(rect.get("ancho_banda_px", 10)),
        float(rect.get("paso_px", 1.0)),
    )


def aplicar_modo_frame(
    frame_u16: np.ndarray,
    config: dict,
    *,
    fondo_temporal: np.ndarray | None = None,
) -> dict:
    """Aplica el preprocesamiento indicado por `config` a un frame uint16.

    Devuelve el mismo dict que `pp.pipeline_frame` (con claves `crudo`,
    `sin_fondo_temporal`, `sin_fondo`, `sin_ruido`, `enmascarado`, `rgb_uint8`,
    `mascara_roi`) para que el codigo downstream no cambie.

    En modo `rgb_crudo` todos los pasos intermedios apuntan al frame crudo
    (sin transformar), excepto `rgb_uint8` que sale de la conversion directa.

    `fondo_temporal`: si se pasa, se usa para el paso de sustraccion temporal.
    En modo `configurable` con `etapas.fondo_temporal=False`, se ignora.

    Si `procesamiento.recorte_roi.habilitado=True`, recorta todos los arrays
    del dict al bbox del ROI con padding configurable. Agrega `bbox` al dict.
    """
    proc = config["procesamiento"]
    modo = proc["modo"]
    params = proc["parametros"]
    metodo_espacial = _metodo_espacial_config(config)

    if modo == "rgb_crudo":
        rgb = pp.frame_a_rgb_uint8(frame_u16)
        pasos = {
            "crudo": frame_u16,
            "sin_fondo_temporal": None,
            "sin_fondo": frame_u16,
            "sin_ruido": frame_u16,
            "enmascarado": frame_u16,
            "rgb_uint8": rgb,
            "mascara_roi": None,
        }
    elif modo == "completo":
        pasos = pp.pipeline_frame(
            frame_u16,
            fondo_temporal=fondo_temporal,
            sigma_fondo=params["sigma_fondo"],
            sigma_ruido=params["sigma_ruido"],
            metodo_espacial=metodo_espacial,
            ruta_roi=_ruta_roi_si_activada(config),
            ancho_roi_px=params["ancho_roi_px"],
        )
    elif modo == "configurable":
        etapas = proc["etapas"]
        ft = fondo_temporal if etapas.get("fondo_temporal", True) else None
        pasos = _pipeline_configurable(
            frame_u16,
            fondo_temporal=ft,
            usar_fondo_espacial=etapas.get("fondo_espacial", True),
            usar_ruido=etapas.get("ruido", True),
            metodo_espacial=metodo_espacial,
            ruta_roi=_ruta_roi_si_activada(config),
            sigma_fondo=params["sigma_fondo"],
            sigma_ruido=params["sigma_ruido"],
            ancho_roi_px=params["ancho_roi_px"],
        )
    else:
        raise ValueError(f"modo desconocido: {modo!r}; usar 'rgb_crudo', 'completo' o 'configurable'")

    # Postproceso espacial: rectificacion (prioridad alta) o bbox crop.
    rect_habilitada, ancho_banda, paso_px = _rectificacion_config(config)
    recorte_habilitado, padding_px = _recorte_roi_config(config)
    ruta_roi = _ruta_roi_si_activada(config)

    pasos["bbox"] = None
    pasos["rectificacion"] = None

    if rect_habilitada and ruta_roi is not None:
        # Rectificacion: 'desenrolla' el axon en un strip horizontal.
        # Sobreescribe los arrays del dict con sus versiones rectificadas.
        from . import kimografo as km

        pasos = _rectificar_pasos(
            pasos, ruta_roi, ancho_banda=ancho_banda, paso_px=paso_px
        )
    elif recorte_habilitado and pasos.get("mascara_roi") is not None:
        pasos = pp.recortar_pasos_pipeline(pasos, padding=padding_px)
        pasos["rectificacion"] = None

    return pasos


def _rectificar_pasos(
    pasos: dict,
    ruta_roi: Path,
    *,
    ancho_banda: int,
    paso_px: float,
) -> dict:
    """Aplica rectificacion a cada array 2D del dict pasos.

    Reemplaza cada array (H, W) por su strip rectificado (2*ancho_banda+1, L).
    El RGB se reconvierte desde el strip ya limpio (sin ceros que estiren).
    """
    from . import kimografo as km

    # Tomamos la polilinea y normales una sola vez, las pasamos por argumento
    # a los siguientes muestreos para no recalcularlas.
    crudo = pasos["crudo"]
    H, W = crudo.shape
    polilinea = km._interpolar_polilinea(km._cargar_polilinea_roi(ruta_roi), paso_px=paso_px)
    normales = km._normales_a_polilinea(polilinea)
    xs_i, ys_i = km._coordenadas_rectificacion(polilinea, normales, ancho_banda, (H, W))

    def _strip(arr):
        if arr is None:
            return None
        return arr[ys_i, xs_i].T

    crudo_strip = _strip(crudo)
    sin_fondo_strip = _strip(pasos.get("sin_fondo"))
    sin_ruido_strip = _strip(pasos.get("sin_ruido"))
    enmascarado_strip = _strip(pasos.get("enmascarado"))
    sin_fondo_temporal_strip = _strip(pasos.get("sin_fondo_temporal"))

    # RGB final: lo recomputamos desde el strip limpio. Como el strip ya es
    # 100% util (no hay ceros del fondo), no necesitamos `ignorar_ceros`.
    fuente_rgb = enmascarado_strip if enmascarado_strip is not None else sin_ruido_strip
    if fuente_rgb is None:
        fuente_rgb = crudo_strip
    rgb_uint8 = pp.frame_a_rgb_uint8(fuente_rgb)

    return {
        "crudo": crudo_strip,
        "sin_fondo_temporal": sin_fondo_temporal_strip,
        "sin_fondo": sin_fondo_strip,
        "sin_ruido": sin_ruido_strip,
        "enmascarado": enmascarado_strip,
        "rgb_uint8": rgb_uint8,
        "mascara_roi": pasos.get("mascara_roi"),  # mantenemos referencia a la original
        "bbox": None,
        "rectificacion": {
            "polilinea": polilinea,
            "normales": normales,
            "ancho_banda": ancho_banda,
            "shape_original": (H, W),
        },
    }


def aplicar_modo_video(video_u16: np.ndarray, config: dict):
    """Aplica el preprocesamiento indicado por `config` a un video 3D uint16.

    Devuelve:
        Si `procesamiento.recorte_roi.habilitado=False` (o no hay ROI):
            array 4D (T, H, W, 3) uint8 — el video procesado completo.
        Si `recorte_roi.habilitado=True` y hay ROI:
            tuple (video_recortado, bbox) donde:
              - video_recortado: array 4D (T, h, w, 3) uint8 al tamano del bbox del ROI
              - bbox: tupla (y0, y1, x0, x1) para mapear coordenadas al frame original

    En modo `rgb_crudo` cada frame se convierte a RGB uint8 sin restar nada.
    En modo `completo` se delega a `pp.pipeline_video`. En modo `configurable`
    se reusa el camino frame-a-frame para respetar los flags `etapas.*`.
    """
    proc = config["procesamiento"]
    modo = proc["modo"]
    params = proc["parametros"]
    metodo_espacial = _metodo_espacial_config(config)

    if modo == "rgb_crudo":
        T, H, W = video_u16.shape
        salida = np.empty((T, H, W, 3), dtype=np.uint8)
        for t in range(T):
            salida[t] = pp.frame_a_rgb_uint8(video_u16[t])
        mascara = None
    elif modo == "completo":
        salida = pp.pipeline_video(
            video_u16,
            sigma_fondo=params["sigma_fondo"],
            sigma_ruido=params["sigma_ruido"],
            metodo_espacial=metodo_espacial,
            ruta_roi=_ruta_roi_si_activada(config),
            ancho_roi_px=params["ancho_roi_px"],
        )
        ruta_roi = _ruta_roi_si_activada(config)
        mascara = pp.cargar_mascara_roi(
            ruta_roi, shape=video_u16.shape[1:], ancho_px=params["ancho_roi_px"]
        ) if ruta_roi is not None else None
    elif modo == "configurable":
        etapas = proc["etapas"]
        if etapas.get("fondo_temporal", True):
            fondo_temporal = pp.calcular_fondo_temporal(video_u16, metodo="mediana")
        else:
            fondo_temporal = None
        T, H, W = video_u16.shape
        salida = np.empty((T, H, W, 3), dtype=np.uint8)
        mascara_compartida = None
        for t in range(T):
            pasos = _pipeline_configurable(
                video_u16[t],
                fondo_temporal=fondo_temporal,
                usar_fondo_espacial=etapas.get("fondo_espacial", True),
                usar_ruido=etapas.get("ruido", True),
                metodo_espacial=metodo_espacial,
                ruta_roi=_ruta_roi_si_activada(config),
                sigma_fondo=params["sigma_fondo"],
                sigma_ruido=params["sigma_ruido"],
                ancho_roi_px=params["ancho_roi_px"],
            )
            salida[t] = pasos["rgb_uint8"]
            if mascara_compartida is None:
                mascara_compartida = pasos.get("mascara_roi")
        mascara = mascara_compartida
    else:
        raise ValueError(f"modo desconocido: {modo!r}; usar 'rgb_crudo', 'completo' o 'configurable'")

    # Postproceso espacial: rectificacion (prioridad alta) o bbox crop.
    rect_habilitada, ancho_banda, paso_px = _rectificacion_config(config)
    recorte_habilitado, padding_px = _recorte_roi_config(config)
    ruta_roi = _ruta_roi_si_activada(config)

    if rect_habilitada and ruta_roi is not None:
        # Rectificamos cada frame al strip. Reusamos el calculo de coords.
        from . import kimografo as km
        H, W = video_u16.shape[1], video_u16.shape[2]
        polilinea = km._interpolar_polilinea(
            km._cargar_polilinea_roi(ruta_roi), paso_px=paso_px
        )
        normales = km._normales_a_polilinea(polilinea)
        xs_i, ys_i = km._coordenadas_rectificacion(polilinea, normales, ancho_banda, (H, W))

        # `salida` ya es (T, H, W, 3) RGB uint8 — rectificamos cada canal.
        T = salida.shape[0]
        K = 2 * ancho_banda + 1
        L = len(polilinea)
        rectificado = np.empty((T, K, L, 3), dtype=np.uint8)
        for t in range(T):
            # (M, K, 3) -> transpose a (K, M, 3) para que el axon corra horizontal
            rectificado[t] = salida[t][ys_i, xs_i].transpose(1, 0, 2)
        info_rect = {
            "polilinea": polilinea,
            "normales": normales,
            "ancho_banda": ancho_banda,
            "shape_original": (H, W),
        }
        return rectificado, info_rect

    if recorte_habilitado and mascara is not None:
        bbox = pp.bbox_de_mascara(mascara, padding=padding_px)
        return pp.recortar_a_bbox(salida, bbox), bbox

    return salida


def recortar_video(video: np.ndarray, config: dict) -> np.ndarray:
    """Recorta un video 3D segun `config['frames']['inicio']` y `config['frames']['fin']`.

    `fin` es exclusivo. `None` = hasta el final.
    """
    f = config["frames"]
    inicio = f.get("inicio", 0) or 0
    fin = f.get("fin")
    return video[inicio:fin] if fin is not None else video[inicio:]


# ── Pipeline configurable por etapas ─────────────────────────────────────────


def _pipeline_configurable(
    frame_u16: np.ndarray,
    *,
    fondo_temporal: np.ndarray | None,
    usar_fondo_espacial: bool,
    usar_ruido: bool,
    ruta_roi: Path | None,
    sigma_fondo: float,
    sigma_ruido: float,
    ancho_roi_px: int,
    metodo_espacial: str = "dog",
) -> dict:
    """Aplica el pipeline saltando etapas segun flags. Devuelve el mismo
    diccionario de pasos que `pp.pipeline_frame` para mantener la API estable.

    `metodo_espacial`:
      - "dog": la etapa espacial es un pasa-banda DoG (fondo + ruido en una
        pasada). `usar_fondo_espacial` la prende/apaga; el flag `usar_ruido`
        no aplica (el DoG ya suaviza).
      - "resta_gauss": flujo clasico — `usar_fondo_espacial` (resta gaussiana)
        y `usar_ruido` (filtro gaussiano) son dos toggles independientes.
    """
    crudo = frame_u16

    if fondo_temporal is not None:
        sin_fondo_temporal = np.clip(crudo.astype(np.float64) - fondo_temporal, 0.0, None)
        entrada = sin_fondo_temporal
    else:
        sin_fondo_temporal = None
        entrada = crudo

    if metodo_espacial == "dog":
        if usar_fondo_espacial:
            sin_fondo = pp.banda_pasante_dog(
                entrada, sigma_senal=sigma_ruido, sigma_fondo=sigma_fondo
            )
        else:
            sin_fondo = entrada.astype(np.float64) if entrada.dtype != np.float64 else entrada
        sin_ruido = sin_fondo  # el DoG ya hace denoise
    elif metodo_espacial == "resta_gauss":
        if usar_fondo_espacial:
            sin_fondo = pp.sustraer_fondo_espacial(entrada, sigma=sigma_fondo)
        else:
            sin_fondo = entrada.astype(np.float64) if entrada.dtype != np.float64 else entrada

        if usar_ruido:
            sin_ruido = pp.reducir_ruido(sin_fondo, metodo="gaussian", sigma=sigma_ruido)
        else:
            sin_ruido = sin_fondo
    else:
        raise ValueError(
            f"metodo_espacial desconocido: {metodo_espacial!r}; usar 'dog' o 'resta_gauss'"
        )

    mascara = None
    enmascarado = sin_ruido
    if ruta_roi is not None:
        mascara = pp.cargar_mascara_roi(ruta_roi, shape=crudo.shape, ancho_px=ancho_roi_px)
        if mascara is not None:
            enmascarado = pp.aplicar_mascara(sin_ruido, mascara)

    rgb_uint8 = pp.frame_a_rgb_uint8(enmascarado, ignorar_ceros=(mascara is not None))

    return {
        "crudo": crudo,
        "sin_fondo_temporal": sin_fondo_temporal,
        "sin_fondo": sin_fondo,
        "sin_ruido": sin_ruido,
        "enmascarado": enmascarado,
        "rgb_uint8": rgb_uint8,
        "mascara_roi": mascara,
    }
