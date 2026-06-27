"""Generacion de quimografos sinteticos con ground truth exacto.

Implementa el acuerdo de la reunion del 2026-06-12 (ver
transcripts/2026-06-12 09-03-26_resumen.md, puntos 6-7): antes de pelear con
datos reales, se generan quimografos sinteticos donde se conoce la posicion de
cada particula momento a momento. Eso da un *ground truth* perfecto para medir
cualquier algoritmo de tracking (KymoButler, transformers, etc.).

El modulo trabaja **directo en espacio quimografo** (no genera un video): cada
particula es una trayectoria x(t) que se dibuja como una estela brillante sobre
fondo negro. Convencion de ejes (la misma que `kimografo.generar_kimografo` y la
que espera KymoButler):

    array shape (T, L)
        filas    = tiempo     (frame)
        columnas = posicion    (px a lo largo del axon)

Direccion: anterograda = +1 (la columna crece con el tiempo), retrograda = -1.

Tres tipos de particula segun el punto 7 del acta:
  - "estacionaria": x(t) constante -> linea vertical pura. Es el grupo (a) de
    Tomas (siempre vertical, no interesa para la cinematica).
  - "anterograda" / "retrograda": movimiento estocastico (corridas a velocidad
    ~constante intercaladas con **pausas**). Una pausa es un tramo vertical
    *dentro* de un movimiento -> grupo (b) de Tomas, que si hay que contar.

Nivel de dificultad: por ahora solo el caso ideal (sin ruido). El render expone
`sigma_psf` y `ruido` (default 0) para sumar realismo / nivel 2 mas adelante.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.draw import line as _raster_line

# ── Parametros del modelo de movimiento (px/frame y frames) ──────────────────

VEL_MIN, VEL_MAX = 0.30, 1.10        # magnitud de velocidad en una corrida
DUR_CORRIDA_MIN, DUR_CORRIDA_MAX = 18, 46   # duracion de una corrida (frames)
DUR_PAUSA_MIN, DUR_PAUSA_MAX = 6, 20         # duracion de una pausa (frames)
PROB_PAUSA = 0.55                    # prob. de insertar una pausa tras una corrida

TIPOS = ("estacionaria", "anterograda", "retrograda")
_SIGNO = {"anterograda": +1, "retrograda": -1}


# ── Estructura de datos ──────────────────────────────────────────────────────

@dataclass
class ParticulaSintetica:
    """Una particula sintetica con su trayectoria conocida (ground truth).

    Atributos:
      id:          identificador entero unico.
      tipo:        "estacionaria" | "anterograda" | "retrograda".
      posiciones:  array (T,) con x(t) en px; NaN en los frames donde la
                   particula no existe (fuera de su ventana de vida).
      segmentos:   lista de tramos (t0, t1, vel_px_frame) que componen la
                   trayectoria. vel=0 marca una pausa (tramo vertical).
    """

    id: int
    tipo: str
    posiciones: np.ndarray
    segmentos: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def frames_visibles(self) -> np.ndarray:
        """Indices de frame donde la particula existe (x no es NaN)."""
        return np.where(~np.isnan(self.posiciones))[0]

    @property
    def desplazamiento_px(self) -> float:
        """Desplazamiento neto en columnas (x_fin - x_inicio)."""
        fv = self.frames_visibles
        if len(fv) < 2:
            return 0.0
        return float(self.posiciones[fv[-1]] - self.posiciones[fv[0]])

    @property
    def n_pausas(self) -> int:
        """Cantidad de pausas (tramos verticales dentro de un movimiento).

        Las estacionarias no cuentan: su unica verticalidad es el grupo (a),
        que no interesa para la cinematica.
        """
        if self.tipo == "estacionaria":
            return 0
        return sum(1 for *_, vel in self.segmentos if vel == 0.0)

    @property
    def n_corridas(self) -> int:
        """Cantidad de tramos en movimiento (vel != 0)."""
        return sum(1 for *_, vel in self.segmentos if vel != 0.0)


# ── Generacion de trayectorias ───────────────────────────────────────────────

def generar_trayectoria_estacionaria(
    pid: int, T: int, L: int, *, rng: np.random.Generator, margen: int = 8
) -> ParticulaSintetica:
    """Particula estatica: x(t) constante en todo el video (vertical pura)."""
    x0 = float(rng.uniform(margen, L - 1 - margen))
    posiciones = np.full(T, x0, dtype=float)
    return ParticulaSintetica(
        id=pid,
        tipo="estacionaria",
        posiciones=posiciones,
        segmentos=[(0, T, 0.0)],
    )


def generar_trayectoria_movil(
    pid: int,
    tipo: str,
    T: int,
    L: int,
    *,
    rng: np.random.Generator,
    margen: int = 8,
) -> ParticulaSintetica:
    """Particula movil: corridas a velocidad ~constante intercaladas con pausas.

    El signo neto del movimiento queda fijado por `tipo` (anterograda crece en
    columna, retrograda decrece), de modo que la direccion es siempre inequivoca.
    """
    if tipo not in _SIGNO:
        raise ValueError(f"tipo movil invalido: {tipo!r}")
    s = _SIGNO[tipo]

    # Ventana de vida: arranca en el primer ~12% y termina en el ultimo ~15%.
    t0 = int(rng.integers(0, max(1, int(0.12 * T))))
    t1 = int(rng.integers(int(0.85 * T), T + 1))
    t1 = max(t1, t0 + DUR_CORRIDA_MIN + 1)
    t1 = min(t1, T)

    # Posicion inicial en la mitad del campo que deja espacio para avanzar:
    # las anterogradas arrancan a la izquierda, las retrogradas a la derecha.
    # Asi sus caminos se cruzan en el centro (cruces que pide el acta).
    if s > 0:
        x = float(rng.uniform(margen, 0.40 * L))
    else:
        x = float(rng.uniform(0.60 * L, L - 1 - margen))

    posiciones = np.full(T, np.nan, dtype=float)
    segmentos: list[tuple[int, int, float]] = []

    t = t0
    ultimo_fue_pausa = False
    primero = True
    while t < t1:
        restante = t1 - t
        hacer_pausa = (
            not primero
            and not ultimo_fue_pausa
            and restante > DUR_PAUSA_MIN
            and rng.random() < PROB_PAUSA
        )
        if hacer_pausa:
            dur = int(min(restante, rng.integers(DUR_PAUSA_MIN, DUR_PAUSA_MAX)))
            vel = 0.0
            ultimo_fue_pausa = True
        else:
            dur = int(min(restante, rng.integers(DUR_CORRIDA_MIN, DUR_CORRIDA_MAX)))
            vel = s * float(rng.uniform(VEL_MIN, VEL_MAX))
            ultimo_fue_pausa = False

        for k in range(dur):
            posiciones[t + k] = x
            x = float(np.clip(x + vel, margen, L - 1 - margen))
        segmentos.append((t, t + dur, vel))
        t += dur
        primero = False

    return ParticulaSintetica(
        id=pid, tipo=tipo, posiciones=posiciones, segmentos=segmentos
    )


def generar_particulas(
    T: int,
    L: int,
    *,
    n_estacionarias: int = 5,
    n_anterogradas: int = 5,
    n_retrogradas: int = 5,
    margen: int = 8,
    rng: np.random.Generator | int | None = None,
) -> list[ParticulaSintetica]:
    """Genera el conjunto de particulas segun el spec del acta (5/5/5 por default).

    Las anterogradas y retrogradas se posicionan con rangos espaciales solapados
    para forzar cruces (X-junctions). Devuelve la lista de `ParticulaSintetica`.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    particulas: list[ParticulaSintetica] = []
    pid = 0
    for _ in range(n_estacionarias):
        particulas.append(
            generar_trayectoria_estacionaria(pid, T, L, rng=rng, margen=margen)
        )
        pid += 1
    for _ in range(n_anterogradas):
        particulas.append(
            generar_trayectoria_movil(pid, "anterograda", T, L, rng=rng, margen=margen)
        )
        pid += 1
    for _ in range(n_retrogradas):
        particulas.append(
            generar_trayectoria_movil(pid, "retrograda", T, L, rng=rng, margen=margen)
        )
        pid += 1
    return particulas


# ── Render del quimografo ────────────────────────────────────────────────────

def renderizar_kimografo(
    particulas: list[ParticulaSintetica],
    T: int,
    L: int,
    *,
    grosor: int = 1,
    intensidad: float = 1.0,
    sigma_psf: float = 0.0,
    ruido: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Dibuja las trayectorias como estelas brillantes sobre fondo negro.

    Cada trayectoria se rasteriza conectando puntos consecutivos (t, x(t)) con
    segmentos de recta, de modo que la estela queda continua (8-conexa) aunque
    la velocidad sea > 1 px/frame.

    Args:
      grosor:     ancho de la estela en px (dilatacion). 1 = linea de 1 px.
      intensidad: valor de la estela en [0, 1].
      sigma_psf:  si > 0, desenfoque gaussiano (point spread function).
      ruido:      si > 0, sigma de ruido gaussiano aditivo (nivel 2).

    Returns:
      array (T, L) float en [0, 1]. Fondo negro (0), estelas brillantes.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    lienzo = np.zeros((T, L), dtype=float)
    for p in particulas:
        fv = p.frames_visibles
        if len(fv) == 0:
            continue
        if len(fv) == 1:
            lienzo[fv[0], int(round(p.posiciones[fv[0]]))] = intensidad
            continue
        # Conectar frames consecutivos (la ventana de vida es contigua).
        for a, b in zip(fv[:-1], fv[1:]):
            if b - a != 1:
                continue  # hueco temporal: no conectar a traves del vacio
            r0, c0 = int(a), int(round(p.posiciones[a]))
            r1, c1 = int(b), int(round(p.posiciones[b]))
            rr, cc = _raster_line(r0, c0, r1, c1)
            lienzo[rr, cc] = intensidad

    if grosor > 1:
        iteraciones = grosor - 1
        mascara = binary_dilation(lienzo > 0, iterations=iteraciones)
        lienzo = np.where(mascara, intensidad, 0.0)

    if sigma_psf > 0:
        lienzo = gaussian_filter(lienzo, sigma=sigma_psf)
        if lienzo.max() > 0:
            lienzo = lienzo / lienzo.max() * intensidad

    if ruido > 0:
        lienzo = lienzo + rng.normal(0.0, ruido, size=lienzo.shape)

    return np.clip(lienzo, 0.0, 1.0)


# ── Ground truth ─────────────────────────────────────────────────────────────

def ground_truth_dataframe(particulas: list[ParticulaSintetica]) -> pd.DataFrame:
    """Ground truth en formato largo: una fila por (particula, frame).

    Columnas: particula_id, tipo, frame, x_px.
    """
    filas = []
    for p in particulas:
        for t in p.frames_visibles:
            filas.append(
                {
                    "particula_id": p.id,
                    "tipo": p.tipo,
                    "frame": int(t),
                    "x_px": float(p.posiciones[t]),
                }
            )
    return pd.DataFrame(filas, columns=["particula_id", "tipo", "frame", "x_px"])


def resumen_particulas(particulas: list[ParticulaSintetica]) -> pd.DataFrame:
    """Resumen por particula: ventana, desplazamiento, pausas y corridas."""
    filas = []
    for p in particulas:
        fv = p.frames_visibles
        filas.append(
            {
                "particula_id": p.id,
                "tipo": p.tipo,
                "t_inicio": int(fv[0]) if len(fv) else -1,
                "t_fin": int(fv[-1]) if len(fv) else -1,
                "x_inicio": float(p.posiciones[fv[0]]) if len(fv) else np.nan,
                "x_fin": float(p.posiciones[fv[-1]]) if len(fv) else np.nan,
                "desplazamiento_px": p.desplazamiento_px,
                "n_corridas": p.n_corridas,
                "n_pausas": p.n_pausas,
            }
        )
    return pd.DataFrame(filas)


# ── Persistencia (handoff al notebook 05) ────────────────────────────────────

def guardar_sintetico(
    kimo: np.ndarray,
    particulas: list[ParticulaSintetica],
    dir_salida: Path | str,
    nombre: str = "sintetico",
) -> dict[str, Path]:
    """Guarda el quimografo (PNG, fondo negro) y el ground truth (CSV).

    El PNG es la entrada que consume KymoButler en el notebook 05; los CSV son
    el ground truth contra el cual se mide el tracking.

    Returns:
      dict con las rutas escritas: {"png", "gt", "resumen"}.
    """
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    ruta_png = dir_salida / f"{nombre}.png"
    img_u8 = np.clip(kimo * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(ruta_png)

    ruta_gt = dir_salida / f"{nombre}_gt.csv"
    ground_truth_dataframe(particulas).to_csv(ruta_gt, index=False)

    ruta_resumen = dir_salida / f"{nombre}_resumen.csv"
    resumen_particulas(particulas).to_csv(ruta_resumen, index=False)

    return {"png": ruta_png, "gt": ruta_gt, "resumen": ruta_resumen}
