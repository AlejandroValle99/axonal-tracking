"""Exportacion de etiquetas de deteccion/segmentacion a partir de datasets
sinteticos de synthkymo, para entrenar un detector (YOLO) y/o un segmentador
(SAM3-decoder, Mask2Former).

Esta es la **capa de etiquetado especifica de la tesis** (ver
`plan/detection-segmentation-guide.md`, pasos 3.1-3.3). NO vive en synthkymo a
proposito: synthkymo es un generador general (genera imagen + tabla de ground
truth); convertir esa tabla al formato que espera un modelo concreto (YOLO,
COCO) es una decision de modelado de *esta* tesis. Este modulo solo **lee** la
salida de synthkymo, no la modifica.

Convencion de ejes (la misma que `kimografo_sintetico` y KymoButler):

    kymografo shape (T, L)
        filas    = tiempo    (frame)  -> eje Y de la imagen, alto  = T
        columnas = posicion  (px)     -> eje X de la imagen, ancho = L

    columna = (L - 1) - pos_um / pixel_scale_um   <- OJO con el espejo.
    `pos_um` es arc-length desde el soma, pero synthkymo espeja el kymografo
    izquierda-derecha al extraerlo (`extract.py`: `kymo = kymo[:, ::-1]`, para
    dejar el soma en la columna derecha). Por eso la columna NO es `pos_um/px`
    directo sino su espejo `(L-1) - pos_um/px`. Sin el espejo las cajas salen
    reflejadas y caen sobre fondo vacio (bug real, verificado sobre las trazas).
    NUNCA usar x_px / y_px de positions: esas son coordenadas del *video 2D*
    (donde cae la particula sobre la polilinea del axon), no del kymografo.

Formato YOLO (deteccion): un .txt por imagen, una linea por traza,
    `clase x_center y_center width height`  (todo normalizado a [0,1] por L y T).
Mascara (segmentacion): la polilinea de la traza rasterizada y dilatada por su
    ancho renderizado real (size_um / pixel_scale_um) -> mascara binaria por
    traza, para fine-tuning de decoder / Mask2Former / mask-IoU (paso 3.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.draw import line as _raster_line
from skimage.morphology import disk

__all__ = [
    "CajaTraza",
    "ids_que_se_mueven",
    "cajas_desde_positions",
    "mascara_traza",
    "kymografo_a_rgb_uint8",
    "exportar_muestra",
    "exportar_dataset",
    "escribir_yaml_dataset",
    "plot_muestra_anotada",
]


# Clases YOLO (indice = id de clase). Dos clases para que el modelo distinga
# moviles de estaticas: una caja de un movil puede solaparse con la de un
# estatico, asi que el detector tiene que aprender a diferenciarlas, no ignorar
# las estaticas. Aguas abajo (cinematica) se usan solo las moviles.
CLASE_MOVIL = 0
CLASE_ESTATICO = 1
CLASES = ("movil", "estatico")  # nombres para el data-config de YOLO

# La particula NO es la linea-centro de la trayectoria: se renderiza como un blob
# gaussiano de sigma=size_px hasta ~3 sigma (movie.py: _stamp_gaussian, radius=3*sigma).
# Medido sobre las trazas reales, el medio-ancho visible ~= 3 * size_px. La caja se
# agranda este factor * size_px a cada lado en POSICION para encerrar el blob, no solo
# el centro. (En TIEMPO no se agranda: cada frame es una fila, sin blur temporal.)
FACTOR_ANCHO = 3.0


# --------------------------------------------------------------------------- #
# Geometria: de positions.csv a cajas/mascaras en pixeles del kymografo
# --------------------------------------------------------------------------- #
@dataclass
class CajaTraza:
    """Caja de una traza en pixeles del kymografo (x=columna=posicion, y=fila=frame)."""

    particle_id: int
    tipo: str
    x1: float
    y1: float
    x2: float
    y2: float
    clase: int = CLASE_MOVIL

    def a_yolo(self, ancho_L: int, alto_T: int) -> str:
        """Linea YOLO normalizada: `clase xc yc w h` (todo en [0,1])."""
        xc = (self.x1 + self.x2) / 2 / ancho_L
        yc = (self.y1 + self.y2) / 2 / alto_T
        w = (self.x2 - self.x1) / ancho_L
        h = (self.y2 - self.y1) / alto_T
        return f"{self.clase} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def ids_que_se_mueven(
    positions: pd.DataFrame, pixel_scale_um: float, umbral_px: float
) -> set[int]:
    """`particle_id`s cuya posicion real cambia >= `umbral_px` a lo largo de TODO
    el video (contando frames visibles o no).

    Implementa el criterio confirmado por Tomas: cuenta **toda particula que se
    mueva en algun momento**, aunque haya estado casi todo el video quieta y solo
    se mueva al final (mover-pausar-mover sigue siendo mover); excluye **solo** las
    que nunca se mueven (estaticas puras -- rango de posicion 0). No se filtra por
    la etiqueta `type`: una particula tipo `anterograde` que en esta realizacion
    nunca se movio tambien se excluye, y una que se mueve apenas se incluye. El
    rango se mide sobre la trayectoria completa a proposito: si una particula solo
    es visible un ratito pero se mueve, sigue siendo un mover.
    """
    rango_um = positions.groupby("particle_id")["pos_um"].agg(lambda s: s.max() - s.min())
    return set(rango_um.index[(rango_um / pixel_scale_um) >= umbral_px])


def cajas_desde_positions(
    positions: pd.DataFrame,
    pixel_scale_um: float,
    shape: tuple[int, int],
    *,
    min_desplazamiento_px: float = 1.0,
    etiquetar_estaticos: bool = True,
    tipos: list[str] | None = None,
    solo_visibles: bool = True,
) -> list[CajaTraza]:
    """Una `CajaTraza` por `particle_id`, clasificada movil (clase 0) vs estatico (clase 1).

    - `shape`: (T, L) del kymografo.
    - `min_desplazamiento_px`: umbral que separa movil de estatico -- una particula
      cuenta como movil si se mueve >= este umbral en algun momento del video (ver
      `ids_que_se_mueven`, criterio de Tomas). Default 1.0 px.
    - `etiquetar_estaticos`: si True (default), tambien emite cajas para las
      estaticas como **clase 1** -- asi el modelo aprende a distinguirlas de las
      moviles (clase 0) cuando sus cajas se solapan. Si False, solo moviles (clase 0).
    - `tipos`: filtro opcional por `type` (secundario). None = no filtrar por tipo.
    - `solo_visibles`: la caja abarca solo las filas con `visible==True` (una
      particula no renderizada no deberia reclamar esos pixeles). La caja sigue
      abarcando huecos de ocultamiento internos (min..max) -- la reconexion de
      trayectorias interrumpidas es un paso aparte (guia 4.6).
    """
    T, L = shape
    moviles = ids_que_se_mueven(positions, pixel_scale_um, min_desplazamiento_px)
    df = positions.copy()
    if not etiquetar_estaticos:
        df = df[df["particle_id"].isin(moviles)]
    if tipos is not None:
        df = df[df["type"].isin(tipos)]
    if solo_visibles and "visible" in df.columns:
        df = df[df["visible"].astype(bool)]  # el CSV guarda visible como int 0/1

    cajas: list[CajaTraza] = []
    for pid, sub in df.groupby("particle_id"):
        if len(sub) == 0:
            continue
        clase = CLASE_MOVIL if pid in moviles else CLASE_ESTATICO
        # espejo L-R del kymografo (extract.py: kymo[:, ::-1], soma a la derecha):
        # la columna es (L-1) - pos_um/px, no pos_um/px directo. min/max se re-ordenan solos.
        col = (L - 1) - sub["pos_um"].to_numpy() / pixel_scale_um
        frame = sub["frame"].to_numpy().astype(float)
        size_px = max(float(sub["size_um"].mean()) / pixel_scale_um, 1.0)

        # agrandar por el ancho renderizado del blob (FACTOR_ANCHO * size_px a cada
        # lado en posicion); en tiempo solo media fila (sin blur temporal entre frames).
        margen = FACTOR_ANCHO * size_px
        x1, x2 = float(col.min()) - margen, float(col.max()) + margen
        y1, y2 = float(frame.min()) - 0.5, float(frame.max()) + 0.5
        # clip: ambos extremos a [0, L] / [0, T] (clampeo simetrico, asi x1<=x2 se
        # conserva aunque la particula caiga sobre/pasando el borde -- p.ej. pos_um~0
        # en el soma da col~L, y un clip asimetrico dejaria x1>x2).
        x1, x2 = min(max(x1, 0.0), float(L)), min(max(x2, 0.0), float(L))
        y1, y2 = min(max(y1, 0.0), float(T)), min(max(y2, 0.0), float(T))
        if x2 - x1 < 0.5 or y2 - y1 < 0.5:  # caja degenerada: particula fuera del kymografo
            continue
        cajas.append(CajaTraza(int(pid), str(sub["type"].iloc[0]), x1, y1, x2, y2, clase))
    return cajas


def mascara_traza(
    positions: pd.DataFrame,
    particle_id: int,
    pixel_scale_um: float,
    shape: tuple[int, int],
    *,
    solo_visibles: bool = True,
) -> np.ndarray:
    """Mascara binaria (T, L) de una traza: polilinea rasterizada y dilatada por
    su ancho renderizado real (`size_um / pixel_scale_um`). Mismo criterio que el
    prototipo de `07_validacion_prompts_sam3.ipynb` (paso 3.2 de la guia)."""
    T, L = shape
    sub = positions[positions["particle_id"] == particle_id]
    if solo_visibles and "visible" in sub.columns:
        sub = sub[sub["visible"].astype(bool)]  # el CSV guarda visible como int 0/1
    sub = sub.sort_values("frame")
    mask = np.zeros((T, L), dtype=bool)
    if len(sub) == 0:
        return mask

    rows = sub["frame"].to_numpy()
    cols = (L - 1) - sub["pos_um"].to_numpy() / pixel_scale_um  # espejo L-R (ver cajas_desde_positions)
    for k in range(len(sub) - 1):
        rr, cc = _raster_line(
            int(rows[k]), int(round(cols[k])), int(rows[k + 1]), int(round(cols[k + 1]))
        )
        ok = (rr >= 0) & (rr < T) & (cc >= 0) & (cc < L)
        mask[rr[ok], cc[ok]] = True
    if len(sub) == 1:  # una sola muestra: marcar ese pixel
        r, c = int(rows[0]), int(round(cols[0]))
        if 0 <= r < T and 0 <= c < L:
            mask[r, c] = True

    # dilatar por ~2 sigma para cubrir el nucleo brillante del blob gaussiano (la caja
    # usa ~3 sigma, FACTOR_ANCHO; la mascara queda algo mas ajustada, dentro de la caja).
    ancho_px = max(int(round(2.0 * float(sub["size_um"].mean()) / pixel_scale_um)), 1)
    return binary_dilation(mask, structure=disk(ancho_px))


# --------------------------------------------------------------------------- #
# Imagen: kymografo -> RGB uint8 (los backbones preentrenados esperan 3 canales)
# --------------------------------------------------------------------------- #
def kymografo_a_rgb_uint8(kymo: np.ndarray) -> np.ndarray:
    """(T, L) -> (T, L, 3) uint8 replicando el canal. Si no es uint8, estira por
    percentiles p1-p99.5 (mismo criterio que el resto del proyecto)."""
    a = np.asarray(kymo)
    if a.ndim == 3:  # ya tiene canales
        a = a[..., 0]
    if a.dtype != np.uint8:
        lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
        a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.repeat(a[..., None], 3, axis=2)


def _leer_kymografo_tif(path) -> np.ndarray:
    """Lee kymograph.tif de forma robusta (tifffile si esta, si no PIL)."""
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))
    except Exception:
        return np.asarray(Image.open(path))


# --------------------------------------------------------------------------- #
# Exportacion a disco (formato YOLO)
# --------------------------------------------------------------------------- #
def exportar_muestra(
    kymo: np.ndarray,
    positions: pd.DataFrame,
    pixel_scale_um: float,
    nombre: str,
    dir_images: Path,
    dir_labels: Path,
    *,
    min_desplazamiento_px: float = 1.0,
    etiquetar_estaticos: bool = True,
    tipos: list[str] | None = None,
    solo_visibles: bool = True,
) -> list[CajaTraza]:
    """Escribe `{nombre}.png` (imagen 3-canales) y `{nombre}.txt` (etiquetas YOLO,
    clase 0=movil / 1=estatico) para una escena. Devuelve las cajas."""
    kymo2d = np.asarray(kymo)
    if kymo2d.ndim == 3:
        kymo2d = kymo2d[..., 0]
    T, L = kymo2d.shape
    cajas = cajas_desde_positions(
        positions, pixel_scale_um, (T, L),
        min_desplazamiento_px=min_desplazamiento_px, etiquetar_estaticos=etiquetar_estaticos,
        tipos=tipos, solo_visibles=solo_visibles,
    )

    dir_images, dir_labels = Path(dir_images), Path(dir_labels)
    dir_images.mkdir(parents=True, exist_ok=True)
    dir_labels.mkdir(parents=True, exist_ok=True)

    Image.fromarray(kymografo_a_rgb_uint8(kymo2d)).save(dir_images / f"{nombre}.png")
    lineas = [c.a_yolo(L, T) for c in cajas]
    (dir_labels / f"{nombre}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""))
    return cajas


def exportar_dataset(
    dataset_dir,
    out_dir,
    split: str,
    *,
    min_desplazamiento_px: float = 1.0,
    etiquetar_estaticos: bool = True,
    tipos: list[str] | None = None,
    solo_visibles: bool = True,
) -> int:
    """Lee un dataset de synthkymo (`dataset_dir/sample_*/` con `kymograph.tif`,
    `positions.csv`, `config.yaml`) y escribe el split `split` en formato YOLO:
    `out_dir/images/{split}/*.png` + `out_dir/labels/{split}/*.txt`.

    Devuelve la cantidad de muestras exportadas. El pixel_scale se lee del
    `config.yaml` de cada muestra (por eso conviene el config.yaml por-muestra
    que agrega el generador).
    """
    dataset_dir = Path(dataset_dir)
    dir_images = Path(out_dir) / "images" / split
    dir_labels = Path(out_dir) / "labels" / split

    n = 0
    for sample in sorted(dataset_dir.glob("sample_*")):
        kymo = _leer_kymografo_tif(sample / "kymograph.tif")
        positions = pd.read_csv(sample / "positions.csv")
        cfg = yaml.safe_load((sample / "config.yaml").read_text())
        pixel_scale_um = float(cfg["general"]["pixel_scale_um"])
        exportar_muestra(
            kymo, positions, pixel_scale_um, sample.name, dir_images, dir_labels,
            min_desplazamiento_px=min_desplazamiento_px, etiquetar_estaticos=etiquetar_estaticos,
            tipos=tipos, solo_visibles=solo_visibles,
        )
        n += 1
    return n


def escribir_yaml_dataset(
    out_dir,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    clases: tuple[str, ...] = CLASES,
    nombre: str = "kymograph_detection.yaml",
) -> Path:
    """Escribe el data-config que consume ultralytics YOLO (`data=...`)."""
    out_dir = Path(out_dir)
    data = {"path": str(out_dir.resolve())}
    for s in splits:
        if (out_dir / "images" / s).is_dir():
            data[s] = f"images/{s}"
    data["names"] = {i: c for i, c in enumerate(clases)}
    ruta = out_dir / nombre
    ruta.write_text(yaml.safe_dump(data, sort_keys=False))
    return ruta


# --------------------------------------------------------------------------- #
# Sanity check visual: dibujar las cajas/mascaras derivadas sobre el kymografo
# --------------------------------------------------------------------------- #
def plot_muestra_anotada(
    kymo: np.ndarray,
    positions: pd.DataFrame,
    pixel_scale_um: float,
    ax=None,
    *,
    min_desplazamiento_px: float = 1.0,
    etiquetar_estaticos: bool = True,
    tipos: list[str] | None = None,
    con_mascara: bool = True,
    solo_visibles: bool = True,
):
    """Dibuja el kymografo con las cajas (y opcionalmente mascaras) derivadas de
    `positions`, para confirmar de un vistazo que las etiquetas caen sobre las
    trazas antes de generar el dataset completo (guia paso 5: detectar la trampa
    fila/columna). Devuelve el `ax`."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    kymo2d = np.asarray(kymo)
    if kymo2d.ndim == 3:
        kymo2d = kymo2d[..., 0]
    T, L = kymo2d.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(kymo2d, cmap="gray", aspect="auto",
              vmin=np.percentile(kymo2d, 1), vmax=np.percentile(kymo2d, 99.5))

    cajas = cajas_desde_positions(
        positions, pixel_scale_um, (T, L),
        min_desplazamiento_px=min_desplazamiento_px, etiquetar_estaticos=etiquetar_estaticos,
        tipos=tipos, solo_visibles=solo_visibles,
    )
    color_clase = {CLASE_MOVIL: "lime", CLASE_ESTATICO: "orange"}
    for caja in cajas:
        color = color_clase.get(caja.clase, "cyan")
        if con_mascara and caja.clase == CLASE_MOVIL:
            m = mascara_traza(positions, caja.particle_id, pixel_scale_um, (T, L),
                              solo_visibles=solo_visibles)
            ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.35, aspect="auto")
        ax.add_patch(Rectangle(
            (caja.x1, caja.y1), caja.x2 - caja.x1, caja.y2 - caja.y1,
            edgecolor=color, facecolor="none", linewidth=1.5,
        ))
        ax.text(caja.x1, caja.y1 - 0.5, CLASES[caja.clase], color=color,
                fontsize=7, va="bottom")
    ax.set_xlabel("posición / columna (px)")
    ax.set_ylabel("frame / fila (tiempo)")
    return ax
