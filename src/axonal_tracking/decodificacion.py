"""Decodificacion de KymoRoPE: mapas por-pixel -> polilineas (trayectorias).

KymoRoPE no emite trayectorias: emite tres mapas por pixel (trackness, embedding 8-D,
orientacion). Este modulo los convierte en polilineas `(frame, col_subpixel)`, el formato
que `evaluacion.evaluar_trayectorias_polilineas` consume -- el mismo harness que ya midio
a KymoButler, Mask2Former y YOLO+SAM3, asi que las cifras son comparables.

    p(movil) >= umbral            mascara movil
      -> componentes conexas       conectividad PRIMERO
      -> mean-shift del embedding  dentro de cada componente
      -> seguir en cadenas         una posicion por fila, por prediccion de pendiente
      -> extraer_subpixel          el unico centroide del repo
      -> filtro de movilidad       mismo criterio que el GT del harness (>= 8 px)

Tres decisiones, cada una por un motivo medido o por una restriccion del harness:

1. **Conectividad antes que embedding.** En val, dos trazas LEJANAS pueden tener medias de
   embedding cercanas (NB12 SS9: el 76% de las muestras con >= 2 moviles tiene un par a
   menos de la separacion que pide el entrenamiento). Un mean-shift global las fundiria;
   agrupando dentro de cada componente conexa, el embedding solo decide donde hace falta:
   trazas que se tocan o se cruzan. Una traza aislada no pasa por el embedding.
2. **Seguir en cadenas.** `extraer_subpixel` centroida la fila ENTERA de la mascara: si un
   cluster tiene dos corridas en la misma fila, el centroide cae en el medio y la
   polilinea es basura. `cortar_en_cadenas` arma trayectorias de una posicion por fila
   siguiendo cada una por prediccion de pendiente, a traves de los cruces. (Una primera
   version cortaba ante cualquier ambiguedad; con trackness e identidad GT fragmentaba
   1.37 trayectorias por particula en val, casi todo en cruces -- ver su docstring.)
3. **Filtro de movilidad del harness.** El GT del harness es "movil" con rango >= 8 px
   (`ed.MIN_DESPLAZAMIENTO_PX_MOVIL`, umbral del laboratorio), pero los targets de KymoRoPE
   usan 4 px (`datos_pixel.MIN_DESPLAZAMIENTO_PX`): el modelo marca moviles particulas que el
   harness no cuenta (2.3% de las moviles de val). Mismo filtro que la variante
   `solo_moviles` de KymoButler, del lado de la prediccion.

Todo sale por `ParametrosDecode`, para barrerlo en val sin tocar el codigo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label
from scipy.signal import find_peaks

from axonal_tracking import evaluacion as ev
from axonal_tracking.etiquetas_deteccion import MIN_DESPLAZAMIENTO_PX_MOVIL
from axonal_tracking.kymorope import SalidaKymoRoPE, dispositivo_preferido

__all__ = [
    "MapasDecode",
    "ParametrosDecode",
    "agrupar_mean_shift",
    "cortar_en_cadenas",
    "decodificar",
    "mapas_desde_salida",
    "suavizar_polilinea",
]

_VECINDAD_8 = np.ones((3, 3), dtype=int)
# ver `cortar_en_cadenas`: con max_hueco_filas <= 2 el tope no cambia nada
_TOPE_CRECIMIENTO_FILAS = 3
_MIN_FILAS_FRAGMENTO = 3  # con enlace activo, fragmentos mas cortos no se consideran
_PROMINENCIA_MIN_PICO = 0.15  # fraccion del rango de intensidad de la corrida (`_centros_por_pico`)
_FILAS_PENDIENTE = 5  # filas del final de un fragmento para estimar su pendiente


@dataclass(frozen=True)
class ParametrosDecode:
    """Perillas del decode. Los defaults son el punto de partida, no el punto de operacion:
    ese se elige en val (`scripts/evaluar_kymorope.py`)."""

    umbral_movil: float = 0.5  # sobre softmax(trackness)[movil]
    usar_embedding: bool = True  # False = una instancia por componente (solo geometria)
    ancho_banda: float = 1.0  # mean-shift, kernel plano, en unidades del embedding
    peso_orientacion: float = 0.0  # 0 = solo embedding; >0 agrega w*(sin, cos) del head
    n_semillas: int = 128  # por ronda de mean-shift
    min_soporte: int = 20  # puntos a < ancho_banda que necesita un modo para sobrevivir
    max_dx_px: float = 2.0  # tolerancia de solape entre corridas de filas consecutivas
    max_hueco_filas: int = 10  # filas vacias que una cadena salta DENTRO de un cluster (cruces)
    absorber_px: float = 20.0  # astillas de la misma particula en un cruce (ver cortar_en_cadenas)
    # enlace de fragmentos ENTRE componentes (`_enlazar_fragmentos`); 0 = apagado
    max_hueco_enlace: int = 0  # filas
    tol_enlace_px: float = 4.0  # error de posicion admitido al retomar
    max_dist_emb_enlace: float = 1.5  # = delta_d de la perdida discriminativa
    # enlace en zonas densas (ver `_enlazar_fragmentos`); margen 0 y denso = max_dist_emb_enlace
    # reproducen el enlace anterior
    margen_enlace: float = 0.25  # costo minimo entre el mejor candidato y el segundo
    max_dist_emb_denso: float = 0.75  # umbral de embedding si otra trayectoria cruza el hueco
    radio_denso_px: float = 10.0  # "cruza el hueco" = pasa a menos de esto del tramo enlazado
    # Posicion de cada fila, tres modos:
    # - `centro_desde_mascara` (default): centro geometrico de la corrida completa de la
    #   mascara; donde la corrida es compartida, el de los pixeles propios;
    # - los dos en False: centro de los pixeles del cluster (primera version). En val da
    #   exactamente lo mismo que el default;
    # - `centro_por_pico`: pico de intensidad del kimografo (`_centros_por_pico`).
    #   **Apagado: resultado negativo** (val, 2026-09-29). Sin guardas subio el id-switch
    #   de 0.094 a 0.153; con guardas empeoro la posicion (0.047 -> 0.049 um, y con
    #   trackness e identidad GT 0.034 -> 0.040): en kimografos reales el pixel mas
    #   brillante de UNA fila ruidosa es peor estimador que el centro de la mascara, que
    #   promedia sobre el ancho de la traza. Queda para no repetir la prueba.
    centro_por_pico: bool = False
    max_desplazamiento_pico_px: float = 3.0
    centro_desde_mascara: bool = True
    # Suavizado temporal de la posicion a lo largo de cada trayectoria (`suavizar_polilinea`):
    # None (default), "mediana", "media" o "sg" (Savitzky-Golay cuadratico), con ventana en
    # filas. Se elige en val.
    suavizado: str | None = None
    ventana_suavizado: int = 5
    min_filas: int = 10  # igual a `min_frames` de KymoButler; el punto de operacion usa 30
    min_desplazamiento_px: float = MIN_DESPLAZAMIENTO_PX_MOVIL  # filtro del harness


@dataclass
class MapasDecode:
    """Lo que consume `decodificar`, en numpy. Puede venir del modelo
    (`mapas_desde_salida`) o armarse desde el GT para las filas diagnosticas: con
    `ids_oraculo` el agrupamiento es la identidad GT (embedding perfecto)."""

    p_movil: np.ndarray  # (T, L) probabilidad (o 0/1 si viene del GT)
    embedding: np.ndarray | None = None  # (D, T, L)
    orientacion: np.ndarray | None = None  # (2, T, L), (sin, cos)
    ids_oraculo: np.ndarray | None = None  # (T, L) int, 0 = sin instancia


def mapas_desde_salida(salida: SalidaKymoRoPE) -> MapasDecode:
    """`SalidaKymoRoPE` (tensores del modelo) -> `MapasDecode`. Una sola conversion por
    muestra: el barrido decodifica varias veces desde los mismos mapas."""
    return MapasDecode(
        p_movil=salida.trackness.float().softmax(0)[2].cpu().numpy(),
        embedding=salida.embedding.float().cpu().numpy(),
        orientacion=salida.orientacion.float().cpu().numpy(),
    )


# --------------------------------------------------------------------------- #
# Mean-shift del embedding
# --------------------------------------------------------------------------- #
def agrupar_mean_shift(
    X: np.ndarray,
    ancho_banda: float,
    *,
    n_semillas: int = 128,
    min_soporte: int = 20,
    max_iter: int = 30,
    max_rondas: int = 5,
    tol: float = 1e-3,
    device=None,
) -> np.ndarray:
    """Mean-shift con kernel plano sobre `X` (N, D) -> etiqueta por punto (0..k-1).

    Semillas al azar (generador fijo: determinista) no ven clusters chicos -- una traza
    corta dentro de una componente grande quedaria sin modo propio y se asignaria a la
    vecina. Por eso corre por RONDAS: los puntos a mas de `ancho_banda` de todo modo
    encontrado siembran la ronda siguiente.

    El riesgo inverso son las colas: en val los clusters son flojos (radio p90 de 1 a 2,
    NB12 SS9) y los puntos de la periferia de UNA traza sembrarian modos propios que le
    abririan huecos. Por eso un modo necesita `min_soporte` puntos a menos de
    `ancho_banda` para sobrevivir: una traza corta real (decenas de pixeles) lo tiene, una
    cola rala no. Modos a menos de `ancho_banda` entre si se funden (mismo criterio que
    `sklearn.cluster.MeanShift`), quedandose con el de mas soporte. Cada punto va a su
    modo sobreviviente mas cercano."""
    n = len(X)
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    dev = device or dispositivo_preferido()
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    gen = torch.Generator().manual_seed(0)
    h = float(ancho_banda)

    modos = Xt.new_zeros((0, Xt.shape[1]))
    candidatos = torch.arange(n, device=dev)
    for _ in range(max_rondas):
        semillas = Xt[candidatos[torch.randperm(len(candidatos), generator=gen)[:n_semillas].to(dev)]]
        for _ in range(max_iter):
            dentro = (torch.cdist(semillas, Xt) <= h).float()  # (S, N)
            nuevas = (dentro @ Xt) / dentro.sum(1, keepdim=True).clamp_min(1)
            desplazamiento = float((nuevas - semillas).norm(dim=1).max())
            semillas = nuevas
            if desplazamiento < tol:
                break
        modos = torch.cat([modos, semillas])
        candidatos = torch.nonzero(torch.cdist(Xt, modos).min(1).values > h).flatten()
        if len(candidatos) < min_soporte:
            break

    # fusion y poda en numpy: son a lo sumo max_rondas*n_semillas modos, y hacerlo con
    # una operacion de GPU por par costaba ~0.1 s por componente
    soporte = (torch.cdist(modos, Xt) <= h).sum(1).cpu().numpy()
    modos_np = modos.cpu().numpy()
    centros: list[np.ndarray] = []
    for j in np.argsort(-soporte, kind="stable"):
        if centros and soporte[j] < min_soporte:
            break  # orden descendente: el resto tampoco llega
        if not centros or np.linalg.norm(np.asarray(centros) - modos_np[j], axis=1).min() > h:
            centros.append(modos_np[j])
    centros_t = torch.as_tensor(np.stack(centros), device=dev)
    return torch.cdist(Xt, centros_t).argmin(1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Cadenas: seguimiento predictivo de corridas, fila a fila
# --------------------------------------------------------------------------- #
def cortar_en_cadenas(
    filas: np.ndarray,
    cols: np.ndarray,
    *,
    max_dx_px: float = 2.0,
    max_hueco_filas: int = 10,
    absorber_px: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pixeles de un cluster -> cadenas `(filas, cols)`: trayectorias con a lo sumo una
    posicion por fila, continuas en el tiempo.

    Una corrida es un tramo de columnas contiguas en una fila. Cada cadena arranca en la
    corrida libre mas temprana (y mas a la izquierda) y la SIGUE: en cada fila siguiente
    predice su posicion con la pendiente de sus ultimas `_FILAS_PENDIENTE` filas y toma la
    corrida libre que solapa con la prediccion (tolerancia `max_dx_px` por fila
    transcurrida, con tope en `_TOPE_CRECIMIENTO_FILAS`) mas cercana a ella. Salta hasta
    `max_hueco_filas` filas sin corrida, prediciendo a traves del hueco. Lo que una cadena
    no toma queda libre para la siguiente.

    **Por que seguir y no cortar.** La version anterior cerraba la cadena ante cualquier
    bifurcacion o fusion de corridas. Medido en val (400 muestras) con trackness e
    identidad GT, o sea con el decode como UNICA fuente de error: 1.37 trayectorias por
    particula, peor que KymoButler entero (1.27), y 401 de las 408 particulas fragmentadas
    eran particulas que cruzan a otra. En un cruce la otra traza tapa a esta: quedan
    astillas a los dos lados o varias filas vacias, y cortar ahi fragmenta aunque la
    identidad sea perfecta. Seguir por prediccion atraviesa el cruce.

    `absorber_px > 0`: la corrida elegida absorbe las otras corridas libres de la misma
    fila a menos de esa distancia -- las astillas de la MISMA particula a los dos lados
    de la que la cruza, asi el centro de la fila no queda sesgado hacia una. Solo tiene
    sentido si el cluster es de UNA identidad (oraculo o mean-shift del embedding); sin
    embedding una componente mezcla particulas vecinas y absorber las fundiria."""
    if len(filas) == 0:
        return []
    orden = np.lexsort((cols, filas))
    f, c = filas[orden], cols[orden]
    corte = np.flatnonzero((np.diff(f) != 0) | (np.diff(c) > 1)) + 1
    ini, fin = np.r_[0, corte], np.r_[corte, len(f)]
    run_fila, run_c0, run_c1 = f[ini], c[ini].astype(float), c[fin - 1].astype(float)
    run_centro = (run_c0 + run_c1) / 2
    por_fila: dict[int, list[int]] = {}
    for k, r in enumerate(run_fila):
        por_fila.setdefault(int(r), []).append(k)
    usado = np.zeros(len(ini), dtype=bool)

    def tomar(k: int, fila: int) -> tuple[list[int], float, float, float]:
        """Marca la corrida `k` (+ las astillas absorbidas); devuelve corridas, centro
        medio pesado por ancho y extremos de la fila."""
        tomadas = [k]
        if absorber_px > 0:
            tomadas += [
                j for j in por_fila[fila]
                if not usado[j] and j != k and abs(run_centro[j] - run_centro[k]) <= absorber_px
            ]
        usado[tomadas] = True
        ancho = run_c1[tomadas] - run_c0[tomadas] + 1
        centro = float(np.average(run_centro[tomadas], weights=ancho))
        return tomadas, centro, float(run_c0[tomadas].min()), float(run_c1[tomadas].max())

    cadenas: list[list[int]] = []
    for inicio in np.lexsort((run_c0, run_fila)):
        if usado[inicio]:
            continue
        fila = int(run_fila[inicio])
        corridas, centro, c0, c1 = tomar(int(inicio), fila)
        t_hist, x_hist = [fila], [centro]
        r = fila + 1
        while r - t_hist[-1] <= max_hueco_filas + 1:
            libres = [k for k in por_fila.get(r, []) if not usado[k]]
            if libres:
                k_ult = min(len(t_hist), _FILAS_PENDIENTE)
                v = float(np.polyfit(t_hist[-k_ult:], x_hist[-k_ult:], 1)[0]) if k_ult >= 2 else 0.0
                dt = r - t_hist[-1]
                desplaz = v * dt
                tol = max_dx_px * min(dt, _TOPE_CRECIMIENTO_FILAS)
                pred = x_hist[-1] + desplaz
                solapan = [
                    k for k in libres
                    if run_c0[k] <= c1 + desplaz + tol and run_c1[k] >= c0 + desplaz - tol
                ]
                if solapan:
                    mejor = min(solapan, key=lambda k: abs(run_centro[k] - pred))
                    nuevas, centro, c0, c1 = tomar(mejor, r)
                    corridas += nuevas
                    t_hist.append(r)
                    x_hist.append(centro)
            r += 1
        cadenas.append(corridas)

    salida = []
    for corridas in cadenas:
        idx = np.concatenate([np.arange(ini[k], fin[k]) for k in corridas])
        salida.append((f[idx], c[idx]))
    return salida


# --------------------------------------------------------------------------- #
# Decode completo
# --------------------------------------------------------------------------- #
def _etiquetas_componente(
    mapas: MapasDecode, f: np.ndarray, c: np.ndarray, params: ParametrosDecode
) -> np.ndarray:
    """Agrupamiento de los pixeles de una componente: identidad GT si hay oraculo,
    mean-shift del embedding (+ orientacion) si no, o todo junto sin embedding."""
    if mapas.ids_oraculo is not None:
        return mapas.ids_oraculo[f, c].astype(np.int64)
    if not params.usar_embedding or mapas.embedding is None:
        return np.zeros(len(f), dtype=np.int64)
    X = mapas.embedding[:, f, c].T
    if params.peso_orientacion > 0 and mapas.orientacion is not None:
        X = np.concatenate([X, params.peso_orientacion * mapas.orientacion[:, f, c].T], axis=1)
    return agrupar_mean_shift(
        X, params.ancho_banda, n_semillas=params.n_semillas, min_soporte=params.min_soporte
    )


def decodificar(
    mapas: MapasDecode,
    kymo_crudo: np.ndarray,
    params: ParametrosDecode | None = None,
    *,
    devolver_instancias: bool = False,
) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], np.ndarray]:
    """Mapas -> polilineas `(frame, col_subpixel)` listas para el harness.

    `kymo_crudo` es el kimografo SIN normalizar: `extraer_subpixel` centroida con la
    intensidad real, igual que para KymoButler (la version normalizada satura en p99.8 y
    aplana los picos). Con `devolver_instancias=True` devuelve ademas un mapa (T, L) con
    el id de la polilinea de cada pixel (0 = ninguna), para visualizar."""
    params = params or ParametrosDecode()
    alto, ancho = mapas.p_movil.shape
    movil = mapas.p_movil >= params.umbral_movil
    componentes, n = label(movil, structure=_VECINDAD_8)
    instancias = np.zeros((alto, ancho), np.int32) if devolver_instancias else None
    polilineas: list[pd.DataFrame] = []
    if n == 0:
        return (polilineas, instancias) if devolver_instancias else polilineas

    # pixeles agrupados por componente en una pasada (mismo patron que
    # `asociacion.partir_en_segmentos`: `np.where(etiquetas == i)` por componente es
    # O(componentes x imagen))
    filas_t, cols_t = np.nonzero(componentes)
    labs = componentes[filas_t, cols_t]
    orden = np.argsort(labs, kind="stable")
    filas_t, cols_t, labs = filas_t[orden], cols_t[orden], labs[orden]
    limites = np.searchsorted(labs, np.arange(1, n + 2))

    # sin enlace, un fragmento corto no tiene futuro: se descarta ya (mismo resultado que
    # antes de existir el enlace); con enlace se guardan desde _MIN_FILAS_FRAGMENTO porque
    # un tramo corto entre dos pausas puede terminar dentro de una traza larga
    min_fragmento = _MIN_FILAS_FRAGMENTO if params.max_hueco_enlace > 0 else params.min_filas
    # absorber astillas solo si cada cluster es UNA identidad (oraculo o embedding)
    con_identidad = mapas.ids_oraculo is not None or (
        params.usar_embedding and mapas.embedding is not None
    )
    absorber = params.absorber_px if con_identidad else 0.0
    fragmentos: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        f, c = filas_t[limites[i]:limites[i + 1]], cols_t[limites[i]:limites[i + 1]]
        if len(np.unique(f)) < min_fragmento:
            continue
        etiquetas = _etiquetas_componente(mapas, f, c, params)
        for k in np.unique(etiquetas):
            sel = etiquetas == k
            for fc, cc in cortar_en_cadenas(
                f[sel], c[sel], max_dx_px=params.max_dx_px,
                max_hueco_filas=params.max_hueco_filas, absorber_px=absorber,
            ):
                if len(np.unique(fc)) >= min_fragmento:
                    fragmentos.append((fc, cc))
    if params.max_hueco_enlace > 0 and len(fragmentos) > 1:
        fragmentos = _enlazar_fragmentos(fragmentos, mapas, params)

    # dueno de cada pixel movil (1..n, 0 = ninguno): para saber si la corrida completa de
    # una fila es de una sola trayectoria o la comparte con otra en un cruce
    propietario = None
    if params.centro_desde_mascara or params.centro_por_pico:
        propietario = np.zeros((alto, ancho), np.int32)
        for j, (fc, cc) in enumerate(fragmentos, 1):
            propietario[fc, cc] = j

    for j, (fc, cc) in enumerate(fragmentos, 1):
        if len(np.unique(fc)) < params.min_filas:
            continue
        poli = _polilinea_subpixel(
            fc, cc, kymo_crudo,
            mascara=movil if (params.centro_desde_mascara or params.centro_por_pico) else None,
            propietario=propietario, propio=j,
            por_pico=params.centro_por_pico,
            max_desplazamiento_pico_px=params.max_desplazamiento_pico_px,
        )
        if params.suavizado:
            poli = suavizar_polilinea(poli, params.suavizado, params.ventana_suavizado)
        if len(poli) and ev.es_movil_polilinea(poli, params.min_desplazamiento_px):
            polilineas.append(poli)
            if instancias is not None:
                instancias[fc, cc] = len(polilineas)
    return (polilineas, instancias) if devolver_instancias else polilineas


def suavizar_polilinea(poli: pd.DataFrame, metodo: str, ventana: int = 5) -> pd.DataFrame:
    """Suavizado temporal de `col_subpixel` a lo largo de una trayectoria.

    La posicion de cada fila sale de UNA fila ruidosa, pero el transporte axonal se mueve en
    corridas de velocidad casi constante y pausas: entre filas vecinas la posicion cambia
    poco y de forma regular, asi que promediar a lo largo del tiempo baja el ruido por fila
    sin depender de la geometria de la mascara ni de un pico de intensidad.

    Se suaviza por TRAMO de frames consecutivos (una trayectoria enlazada a traves de una
    pausa tiene huecos: no se promedia a traves de ellos); tramos mas cortos que la ventana
    quedan como estan. `metodo`: "mediana" (conserva escalones), "media" (redondea las
    esquinas corrida/pausa) o "sg" (Savitzky-Golay cuadratico: conserva tramos lineales y
    curvas suaves)."""
    from scipy.ndimage import median_filter, uniform_filter1d
    from scipy.signal import savgol_filter

    if len(poli) < ventana or ventana < 2:
        return poli
    orden = np.argsort(poli["frame"].to_numpy(), kind="stable")
    fr = poli["frame"].to_numpy()[orden]
    x = poli["col_subpixel"].to_numpy(dtype=float)[orden]
    salida = x.copy()
    cortes = np.flatnonzero(np.diff(fr) != 1) + 1
    for ini, fin in zip(np.r_[0, cortes], np.r_[cortes, len(fr)]):
        tramo = x[ini:fin]
        if len(tramo) < ventana:
            continue
        if metodo == "mediana":
            salida[ini:fin] = median_filter(tramo, size=ventana, mode="nearest")
        elif metodo == "media":
            salida[ini:fin] = uniform_filter1d(tramo, size=ventana, mode="nearest")
        elif metodo == "sg":
            salida[ini:fin] = savgol_filter(tramo, ventana if ventana % 2 else ventana + 1, 2, mode="interp")
        else:
            raise ValueError(f"suavizado desconocido: {metodo!r}")
    return pd.DataFrame({"frame": fr, "col_subpixel": salida})


def _centros_por_fila(filas: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filas distintas (ordenadas) y la columna media de cada una."""
    orden = np.argsort(filas, kind="stable")
    f, c = filas[orden], cols[orden]
    unicas, ini = np.unique(f, return_index=True)
    return unicas, np.add.reduceat(c.astype(float), ini) / np.diff(np.r_[ini, len(c)])


def _firma(
    mapas: MapasDecode, filas: np.ndarray, cols: np.ndarray, params: ParametrosDecode
) -> np.ndarray | None:
    """Identidad de un fragmento para el enlace: embedding medio, o el id GT mayoritario
    con el oraculo. `None` = sin identidad (enlace solo geometrico) -- tambien cuando
    `usar_embedding=False`, para que las filas "sin embedding" de la escalera no lo usen
    por la puerta de atras del enlace."""
    if mapas.ids_oraculo is not None:
        ids = mapas.ids_oraculo[filas, cols]
        ids = ids[ids > 0]
        return np.array([np.bincount(ids).argmax()]) if len(ids) else None
    if params.usar_embedding and mapas.embedding is not None:
        return mapas.embedding[:, filas, cols].mean(1)
    return None


def _distancia_firmas(a: np.ndarray | None, b: np.ndarray | None, oraculo: bool) -> float:
    if a is None or b is None:
        return 0.0
    if oraculo:
        return 0.0 if a[0] == b[0] else np.inf
    return float(np.linalg.norm(a - b))


def _enlazar_fragmentos(
    fragmentos: list[tuple[np.ndarray, np.ndarray]],
    mapas: MapasDecode,
    params: ParametrosDecode,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Une el FIN de un fragmento con el INICIO de otro posterior, entre componentes.

    Una pausa que el head de trackness marca estatica saca esas filas de la mascara movil
    y parte la traza en dos componentes; `cortar_en_cadenas` trabaja dentro de una
    componente y no puede verlo (medido: estirar su tolerancia de hueco a 30 filas no
    movio `fragmentos_por_gt`). Un enlace A -> B es candidato si:

    - B empieza entre 1 y `max_hueco_enlace` filas despues de que A termina;
    - B retoma a <= `tol_enlace_px` de donde A tiene que estar: quieta (pausa: misma
      columna) o siguiendo con la pendiente de sus ultimas filas (hueco en movimiento),
      la mas cercana de las dos hipotesis;
    - los embeddings medios estan a <= `max_dist_emb_enlace` (con el oraculo: mismo id GT).

    Costo = error de posicion + distancia de embedding (cada uno sobre su tolerancia), y
    asignacion 1 a 1 (Hungaro): cada fin se une a lo sumo con un inicio y viceversa.

    **Zonas densas.** En val el enlace bajaba la fragmentacion (1.461 -> 1.336 trayectorias
    por particula) pero subia el id-switch (0.100 -> 0.116), y en las superposiciones de NB13
    los enlaces malos se concentran donde varias trazas se cruzan. Dos reglas, las dos solo
    endurecen (con `margen_enlace=0` y `max_dist_emb_denso=max_dist_emb_enlace` el enlace es
    el de antes):

    - `margen_enlace`: si un fin (o un inicio) tiene dos candidatos cuyo costo difiere en
      menos de eso, no se enlaza. Varias continuaciones igual de buenas = no se sabe cual
      es; mejor un fragmento de mas que un id-switch.
    - `max_dist_emb_denso`: si otra trayectoria pasa a <= `radio_denso_px` del tramo que
      el enlace cruzaria, el embedding tiene que coincidir mas (umbral mas chico). Ahi es
      donde la continuacion correcta pudo haber quedado en la otra trayectoria y el unico
      candidato que sobra es el equivocado. No aplica al oraculo (su distancia es 0/inf)."""
    from scipy.optimize import linear_sum_assignment

    n = len(fragmentos)
    oraculo = mapas.ids_oraculo is not None
    centros = [_centros_por_fila(f, c) for f, c in fragmentos]
    info = []
    for (f, c), (t, x) in zip(fragmentos, centros):
        k = min(_FILAS_PENDIENTE, len(t))
        v_fin = float(np.polyfit(t[-k:], x[-k:], 1)[0]) if k >= 2 else 0.0
        info.append((t[0], t[-1], x[0], x[-1], v_fin, _firma(mapas, f, c, params)))

    costo = np.full((n, n), np.inf)
    for a, (_, t1a, _, x1a, v1a, fa) in enumerate(info):
        for b, (t0b, _, x0b, _, _, fb) in enumerate(info):
            hueco = t0b - t1a
            if not 1 <= hueco <= params.max_hueco_enlace:
                continue
            dx = min(abs(x0b - x1a), abs(x0b - (x1a + v1a * hueco)))
            if dx > params.tol_enlace_px:
                continue
            d_id = _distancia_firmas(fa, fb, oraculo)
            if d_id > params.max_dist_emb_enlace:
                continue
            if (
                not oraculo
                and d_id > params.max_dist_emb_denso
                and _hueco_concurrido(centros, a, b, (t1a, x1a), (t0b, x0b), params.radio_denso_px)
            ):
                continue
            costo[a, b] = dx / params.tol_enlace_px + (
                0.0 if oraculo else d_id / params.max_dist_emb_enlace
            )
    if params.margen_enlace > 0:
        costo = _podar_ambiguos(costo, params.margen_enlace)
    validos = np.isfinite(costo)
    if not validos.any():
        return fragmentos
    fila_idx, col_idx = linear_sum_assignment(np.where(validos, costo, 1e9))
    siguiente = {int(a): int(b) for a, b in zip(fila_idx, col_idx) if validos[a, b]}

    destinos = set(siguiente.values())
    unidos = []
    for inicio in (i for i in range(n) if i not in destinos):
        partes, j = [], inicio
        while j is not None:
            partes.append(fragmentos[j])
            j = siguiente.get(j)
        unidos.append((np.concatenate([p[0] for p in partes]), np.concatenate([p[1] for p in partes])))
    return unidos


def _hueco_concurrido(
    centros: list[tuple[np.ndarray, np.ndarray]],
    a: int,
    b: int,
    fin_a: tuple[float, float],
    inicio_b: tuple[float, float],
    radio_px: float,
) -> bool:
    """True si alguna OTRA trayectoria pasa a <= `radio_px` del tramo que el enlace A -> B
    cruzaria: filas del fin de A al inicio de B, posicion interpolada entre los dos
    extremos (con la hipotesis de pausa los extremos casi coinciden y queda constante)."""
    (t1, x1), (t0, x0) = fin_a, inicio_b
    for k, (tk, xk) in enumerate(centros):
        if k in (a, b):
            continue
        sel = (tk >= t1) & (tk <= t0)
        if not sel.any():
            continue
        x_tramo = x1 + (x0 - x1) * (tk[sel] - t1) / max(t0 - t1, 1)
        if np.any(np.abs(xk[sel] - x_tramo) <= radio_px):
            return True
    return False


def _podar_ambiguos(costo: np.ndarray, margen: float) -> np.ndarray:
    """Deja sin enlace a todo fin (fila) o inicio (columna) cuyos dos mejores candidatos
    difieren en menos de `margen`. Se mide sobre la matriz original en los dos ejes, asi
    que el resultado no depende del orden en que se poda."""
    podado = costo.copy()
    for eje, vista in ((0, costo), (1, costo.T)):
        for i, fila in enumerate(vista):
            finitos = np.sort(fila[np.isfinite(fila)])
            if len(finitos) >= 2 and finitos[1] - finitos[0] < margen:
                if eje == 0:
                    podado[i, :] = np.inf
                else:
                    podado[:, i] = np.inf
    return podado


def _polilinea_subpixel(
    filas: np.ndarray,
    cols: np.ndarray,
    kymo_crudo: np.ndarray,
    *,
    mascara: np.ndarray | None = None,
    propietario: np.ndarray | None = None,
    propio: int = 0,
    por_pico: bool = False,
    max_desplazamiento_pico_px: float = 3.0,
) -> pd.DataFrame:
    """Cadena -> linea central de 1 px -> `extraer_subpixel`.

    NO se centroida sobre la mascara de la cadena entera: es la traza DILATADA (6-12 px de
    ancho, asi son los targets) y `extraer_subpixel` suma +/- 2 px de margen, asi que la
    ventana metia fondo y estaticas vecinas. Medido en 20 muestras de val con trackness e
    identidad GT: 0.049 um de error de posicion contra 0.031 del techo oraculo. La linea
    central de 1 px es la misma ruta que `asociacion.polilinea_desde_puntos` y que la
    variante `solo_moviles_subpixel` de KymoButler (rasterizar fino -> `extraer_subpixel`),
    asi que el error de posicion es comparable entre metodos.

    Con `mascara` y `por_pico`, el centro de cada fila es el pico de intensidad propio
    (`_centros_por_pico`); con `mascara` sola, el centro de la corrida completa de la
    mascara movil (`_centros_desde_mascara`); sin ella, el de los pixeles de la cadena."""
    alto, ancho = kymo_crudo.shape[:2]
    unicas, centro = _centros_por_fila(filas, cols)
    if mascara is not None and por_pico and propietario is not None:
        centro = _centros_por_pico(
            filas, cols, unicas, centro, mascara, kymo_crudo,
            propietario, propio, max_desplazamiento_pico_px,
        )
    elif mascara is not None and propietario is not None:
        centro = _centros_desde_mascara(filas, cols, unicas, centro, mascara, propietario, propio)
    fina = np.zeros((alto, ancho), dtype=bool)
    fina[unicas, np.clip(np.round(centro).astype(int), 0, ancho - 1)] = True
    return ev.extraer_subpixel(fina, kymo_crudo)


def _centros_por_pico(
    filas: np.ndarray,
    cols: np.ndarray,
    unicas: np.ndarray,
    centro: np.ndarray,
    mascara: np.ndarray,
    kymo_crudo: np.ndarray,
    propietario: np.ndarray,
    propio: int,
    max_desplazamiento_px: float,
) -> np.ndarray:
    """Centro de cada fila en el PICO de intensidad de la particula, no en la geometria de
    la mascara. El cluster decide de quien es la fila; la posicion la da el kimografo.

    **Solo donde es seguro** (si no, queda el centro de los pixeles propios):

    - la corrida de mascara no tiene pixeles de OTRA trayectoria (`propietario`), y los
      pixeles propios caen en una sola corrida;
    - el pico mueve la posicion a lo sumo `max_desplazamiento_px`.

    Motivo de las dos guardas, medido en val el 2026-09-29: la primera version (pico en
    toda fila) bajo el error de posicion del modelo afinado de 0.047 a 0.042 um pero subio
    el id-switch de 0.094 a 0.153, y con trackness e identidad GT (D0) de 0.028 a 0.153 --
    o sea, lo causaba el decode. Con dos particulas a pocos pixeles sus perfiles se funden
    en UN pico; las dos trayectorias saltaban a el, y como queda mas cerca de una de las
    dos particulas, el harness asignaba las filas de la otra a la particula equivocada.

    Por fila: la ventana es la corrida de mascara movil que contiene los pixeles propios
    (extendida por los dos lados mientras siga habiendo mascara); el perfil de intensidad
    se suaviza con [1, 2, 1]/4, y se toma el maximo local mas cercano al centro de los
    pixeles propios. `extraer_subpixel` refina despues +/- 2 px alrededor de ese pico.

    Motivo, medido en val con el modelo afinado sobre 5k (diagnostico por fila del
    2026-09-29): en filas sin ninguna otra movil a menos de 20 px -- el 84% de las filas --
    el error era 18% mayor que el de KymoButler, y con la mascara GT (D0) era igual al de
    KymoButler. O sea: la extraccion anda bien, lo que falla es que el centro geometrico
    de la mascara PREDICHA no cae en el centro de la traza. Cerca de otra movil el error
    se iba HACIA AFUERA (el cluster vecino se queda con los pixeles del medio y los propios
    quedan de un solo lado). El pico de intensidad no depende de ninguna de las dos cosas;
    es lo que sigue el esqueleto de KymoButler (una cresta de la salida de la U-Net).

    Lo que la guarda deja afuera son justamente las filas de cruce, que siguen con el error
    de antes; el pico arregla las filas aisladas (84% de las filas, donde el modelo afinado
    tenia 18% mas error que KymoButler porque el centro de la mascara PREDICHA no cae en el
    centro de la traza)."""
    orden = np.argsort(filas, kind="stable")
    f, c = filas[orden], cols[orden]
    ini = np.searchsorted(f, unicas)
    fin = np.searchsorted(f, unicas, side="right")
    ancho = mascara.shape[1]
    kymo2d = kymo_crudo if kymo_crudo.ndim == 2 else kymo_crudo[..., 0]
    salida = centro.copy()
    for i, r in enumerate(unicas):
        propias = c[ini[i]:fin[i]]
        a, b = int(propias.min()), int(propias.max())
        fila_mascara = mascara[r]
        if not fila_mascara[a:b + 1].all():
            continue  # pixeles propios en mas de una corrida (astillas de un cruce)
        while a > 0 and fila_mascara[a - 1]:
            a -= 1
        while b < ancho - 1 and fila_mascara[b + 1]:
            b += 1
        duenos = propietario[r, a:b + 1]
        if np.any((duenos != 0) & (duenos != propio)):
            continue  # corrida compartida con otra trayectoria
        perfil = kymo2d[r, a:b + 1].astype(float)
        if len(perfil) >= 3:
            perfil = np.convolve(np.pad(perfil, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        bajo, alto = float(perfil.min()), float(perfil.max())
        if alto <= bajo:
            continue
        # Solo maximos PROMINENTES: el ruido de fondo deja maximitos locales en las colas,
        # y el mas cercano a los pixeles propios puede ser uno de esos (visto en la prueba
        # de juguete: particula en 20.0 -> 17.6). Prominencia relativa y no un umbral de
        # altura: una particula tenue al lado de una brillante conserva su prominencia
        # contra el valle que las separa. Los extremos se rellenan con el minimo para que
        # un pico en el borde de la corrida tambien cuente.
        picos, _ = find_peaks(
            np.r_[bajo, perfil, bajo], prominence=_PROMINENCIA_MIN_PICO * (alto - bajo)
        )
        if len(picos) == 0:
            continue
        picos = picos - 1  # por el relleno
        pico = a + picos[np.argmin(np.abs(a + picos - centro[i]))]
        if abs(pico - centro[i]) <= max_desplazamiento_px:
            salida[i] = pico
    return salida


def _centros_desde_mascara(
    filas: np.ndarray,
    cols: np.ndarray,
    unicas: np.ndarray,
    centro: np.ndarray,
    mascara: np.ndarray,
    propietario: np.ndarray,
    propio: int,
) -> np.ndarray:
    """Centro de cada fila tomado de la corrida COMPLETA de la mascara movil que contiene
    los pixeles de la cadena, no solo de los pixeles que el agrupamiento le dio.

    El cluster decide la IDENTIDAD de la fila; la posicion la da la traza entera. Motivo,
    medido en val (400 muestras): el error de posicion era 0.036 um con la mascara del
    modelo sin agrupar (M1) y subia a ~0.048 al agrupar por embedding (M2); con la mascara
    GT, 0.030 -> 0.048. El mean-shift se queda con parte del ancho de la traza en algunas
    filas y la linea central se corre hacia un borde.

    Dos casos se quedan con el centro de los pixeles propios: la corrida completa tiene
    pixeles de OTRA trayectoria (cruce: su centro caeria entre las dos particulas), o los
    pixeles propios caen en mas de una corrida (astillas absorbidas a los dos lados de la
    que cruza)."""
    orden = np.argsort(filas, kind="stable")
    f, c = filas[orden], cols[orden]
    ini = np.searchsorted(f, unicas)
    fin = np.searchsorted(f, unicas, side="right")
    ancho = mascara.shape[1]
    salida = centro.copy()
    for i, r in enumerate(unicas):
        c0, c1 = int(c[ini[i]:fin[i]].min()), int(c[ini[i]:fin[i]].max())
        fila = mascara[r]
        if not fila[c0:c1 + 1].all():
            continue  # pixeles propios en mas de una corrida
        a, b = c0, c1
        while a > 0 and fila[a - 1]:
            a -= 1
        while b < ancho - 1 and fila[b + 1]:
            b += 1
        duenos = propietario[r, a:b + 1]
        if np.any((duenos != 0) & (duenos != propio)):
            continue  # corrida compartida con otra trayectoria
        salida[i] = (a + b) / 2
    return salida
