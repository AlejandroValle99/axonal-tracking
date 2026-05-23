"""Helpers de visualizacion para frames, mascaras, blobs y cajas.

Todas las funciones reciben `ax` opcional para integrarse a layouts mayores.
Si no se pasa `ax`, crean su propia figura.
"""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# ── Mostrar un frame ─────────────────────────────────────────────────────────

def mostrar_frame(
    frame: np.ndarray,
    titulo: str = "",
    ax=None,
    cmap: str = "gray",
    percentiles: tuple[float, float] = (1.0, 99.0),
    ignorar_ceros: bool = False,
):
    """Muestra un frame con contraste ajustado por percentiles.

    Funciona con uint8, uint16, float. Si es RGB (3 canales), ignora cmap.

    Parametros:
        frame: array 2D (escala de grises) o 3D (RGB).
        titulo: titulo del subplot.
        ax: eje matplotlib donde dibujar; si es None, crea uno nuevo.
        cmap: colormap (solo aplica a frames 2D).
        percentiles: rango (low, high) para el stretch de contraste.
        ignorar_ceros: si True, calcula los percentiles solo sobre pixeles != 0.
            Usar para frames enmascarados, donde la mayoria de pixeles son 0 y
            distorsionarian el stretch (el contenido visible queda comprimido
            en un rango chico). Solo aplica a frames 2D.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    if frame.ndim == 3:
        ax.imshow(frame)
    else:
        if ignorar_ceros:
            no_cero = frame[frame > 0]
            if no_cero.size > 0:
                lo, hi = np.percentile(no_cero, list(percentiles))
            else:
                lo, hi = 0, 1
        else:
            lo, hi = np.percentile(frame, list(percentiles))
        ax.imshow(frame, cmap=cmap, vmin=lo, vmax=hi)
    if titulo:
        ax.set_title(titulo)
    ax.axis("off")


def comparar_frames(
    *pares_frame_titulo: tuple[np.ndarray, str],
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Side-by-side de N frames con sus titulos.

    Uso:
        comparar_frames(
            (frame_crudo,  "Original"),
            (frame_limpio, "Sin fondo"),
            (frame_rgb,    "Listo para SAM"),
        )

    Parametros:
        *pares_frame_titulo: tuplas (frame, titulo).
        ncols: cuantas columnas. Si es None, una fila con todos.
        figsize: tamanio de la figura; si None, escala con la cantidad.
    """
    n = len(pares_frame_titulo)
    if n == 0:
        return
    if ncols is None:
        ncols = n
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for ax, (frame, titulo) in zip(axes, pares_frame_titulo):
        mostrar_frame(frame, titulo=titulo, ax=ax)
    # Apagar ejes sobrantes
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()


# ── Overlays sobre frames ────────────────────────────────────────────────────

def overlay_mascaras(
    frame_rgb: np.ndarray,
    masks,
    ax,
    alpha: float = 0.5,
    seed: int = 0,
):
    """Superpone N mascaras con colores aleatorios sobre un frame RGB.

    Acepta:
    - Lista/iterable de mascaras 2D bool/0-1.
    - Tensor torch de shape (N, H, W) o (N, K, H, W) — toma la primera hipotesis.
    - Array numpy con los mismos shapes.

    Parametros:
        frame_rgb: array (H, W, 3) uint8.
        masks: ver arriba.
        ax: eje matplotlib donde dibujar.
        alpha: transparencia de las mascaras.
        seed: semilla para los colores (mismo seed -> mismos colores).
    """
    ax.imshow(frame_rgb)

    arr = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
    # Si vienen multiples hipotesis por mascara, quedarse con la primera.
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None]

    rng = np.random.default_rng(seed)
    for m in arr.astype(bool):
        color = rng.random(3)
        overlay = np.zeros((*m.shape, 4))
        overlay[m] = [*color, alpha]
        ax.imshow(overlay)
    ax.axis("off")


def dibujar_blobs(
    frame_rgb: np.ndarray,
    blobs: np.ndarray,
    ax,
    color: str = "red",
    linewidth: float = 0.8,
):
    """Dibuja circulos sobre las posiciones devueltas por skimage.feature.blob_log.

    Parametros:
        frame_rgb: array (H, W, 3) uint8.
        blobs: array (N, 3) con cada fila [y, x, sigma].
        ax: eje matplotlib donde dibujar.
        color: color del borde del circulo.
        linewidth: grosor del borde.
    """
    ax.imshow(frame_rgb)
    for y, x, sigma in blobs:
        r = sigma * np.sqrt(2)
        ax.add_patch(plt.Circle((x, y), max(r, 3), color=color, fill=False, linewidth=linewidth))
    ax.axis("off")


def dibujar_cajas(
    frame_rgb: np.ndarray,
    boxes,
    ax,
    color: str = "lime",
    labels: Iterable[str] | None = None,
    linewidth: float = 1.2,
):
    """Dibuja rectangulos sobre las cajas devueltas por un detector (DINO/YOLO).

    Parametros:
        frame_rgb: array (H, W, 3) uint8.
        boxes: array/lista (N, 4) con [x1, y1, x2, y2] por caja.
        ax: eje matplotlib donde dibujar.
        color: color del borde.
        labels: opcional, una etiqueta de texto por caja.
        linewidth: grosor del borde.
    """
    ax.imshow(frame_rgb)
    boxes = np.asarray(boxes)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor=color, facecolor="none", linewidth=linewidth,
        ))
        if labels is not None:
            ax.text(x1, y1 - 2, list(labels)[i], color=color, fontsize=7)
    ax.axis("off")


def dibujar_puntos(
    frame_rgb: np.ndarray,
    points_xy,
    ax,
    color_borde: str = "red",
    color_relleno: str = "yellow",
    size: float = 8,
):
    """Dibuja puntos sobre coordenadas (x, y) — utiles para prompts visuales de SAM.

    Parametros:
        frame_rgb: array (H, W, 3) uint8.
        points_xy: lista/array de tuplas (x, y).
        ax: eje matplotlib donde dibujar.
        color_borde, color_relleno, size: estilo del punto.
    """
    ax.imshow(frame_rgb)
    for x, y in points_xy:
        ax.plot(x, y, "o", mfc=color_relleno, mec=color_borde, ms=size, mew=1.2)
    ax.axis("off")
