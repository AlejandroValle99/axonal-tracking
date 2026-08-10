"""Dibuja las cajas YOLO exportadas sobre cada kymografo, para revisar de un
vistazo que las etiquetas del dataset caen sobre las trazas moviles.

Lee DIRECTAMENTE los archivos exportados (`images/{split}/*.png` +
`labels/{split}/*.txt`), o sea muestra exactamente lo que YOLO va a entrenar --
no re-deriva desde `positions.csv`. Asi, si hubiera un bug en la exportacion,
aca se veria.

El kymografo es una tira finita (~60 px de alto): se agranda por `--escala`
(default 4x) solo para que las cajas y las trazas se vean comodas.

Salida:

  datasets/yolo_detection/previews/{train,val,test}/sample_NNNNN.png
    (el kymografo agrandado con las cajas dibujadas encima)

Uso:

  # exportar previews de TODOS los splits -> abris la carpeta y navegas cualquiera
  uv run python scripts/ver_etiquetas_deteccion.py

  # solo un split
  uv run python scripts/ver_etiquetas_deteccion.py train

  # una sola muestra (por nombre o indice) -> la guarda y la abre en el visor
  uv run python scripts/ver_etiquetas_deteccion.py train sample_00042
  uv run python scripts/ver_etiquetas_deteccion.py train 42

  # variante movers-only (datasets/yolo_detection_movers/), cualquier combinacion de arriba
  uv run python scripts/ver_etiquetas_deteccion.py --movers-only train
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parents[1]
DATASETS = RAIZ / "datasets"
SPLITS = ("train", "val", "test")
ESCALA = 4
# color por clase YOLO: 0=movil (verde), 1=estatico (naranja)
COLOR_CLASE = {0: (60, 255, 60), 1: (255, 150, 40)}


def leer_cajas_yolo(lbl_path: Path, ancho: int, alto: int) -> list[tuple[int, float, float, float, float]]:
    """Lee un .txt YOLO (clase xc yc w h normalizado) y devuelve
    (clase, x1, y1, x2, y2) en pixeles por caja."""
    cajas = []
    if not lbl_path.exists():
        return cajas
    for linea in lbl_path.read_text().splitlines():
        if not linea.strip():
            continue
        cls, xc, yc, w, h = linea.split()
        xc, yc, w, h = float(xc) * ancho, float(yc) * alto, float(w) * ancho, float(h) * alto
        cajas.append((int(cls), xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return cajas


def preview(img_path: Path, lbl_path: Path, escala: int = ESCALA) -> tuple[Image.Image, int]:
    """Devuelve (imagen RGB agrandada con las cajas dibujadas por clase, numero de cajas)."""
    img = Image.open(img_path).convert("RGB")
    ancho, alto = img.size
    cajas = leer_cajas_yolo(lbl_path, ancho, alto)
    img = img.resize((ancho * escala, alto * escala), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for cls, x1, y1, x2, y2 in cajas:
        draw.rectangle(
            [x1 * escala, y1 * escala, x2 * escala, y2 * escala],
            outline=COLOR_CLASE.get(cls, (60, 200, 255)), width=2,
        )
    return img, len(cajas)


def exportar_split(yolo_dir: Path, split: str, escala: int = ESCALA) -> Path:
    img_dir, lbl_dir = yolo_dir / "images" / split, yolo_dir / "labels" / split
    out_dir = yolo_dir / "previews" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(img_dir.glob("*.png"))
    total_cajas = 0
    for p in pngs:
        img, n = preview(p, lbl_dir / f"{p.stem}.txt", escala)
        img.save(out_dir / p.name)
        total_cajas += n
    print(f"  {split:5s}: {len(pngs)} previews ({total_cajas} cajas) -> {out_dir}")
    return out_dir


def _resolver_nombre(ident: str) -> str:
    """'42' -> 'sample_00042'; 'sample_00042' -> tal cual."""
    return f"sample_{int(ident):05d}" if ident.isdigit() else ident


def main() -> None:
    args = sys.argv[1:]
    movers_only = "--movers-only" in args
    args = [a for a in args if a != "--movers-only"]
    yolo_dir = DATASETS / ("yolo_detection_movers" if movers_only else "yolo_detection")

    if not yolo_dir.exists():
        sys.exit(f"No existe {yolo_dir}. Genera el dataset YOLO primero "
                 f"(scripts/exportar_etiquetas_deteccion.py{' --movers-only' if movers_only else ''}).")

    if len(args) >= 2:  # una sola muestra: guardar y abrir en el visor
        split, nombre = args[0], _resolver_nombre(args[1])
        img, n = preview(yolo_dir / "images" / split / f"{nombre}.png",
                         yolo_dir / "labels" / split / f"{nombre}.txt")
        out = yolo_dir / "previews" / split / f"{nombre}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"{split}/{nombre}: {n} cajas -> {out}")
        if sys.platform == "darwin":
            subprocess.run(["open", str(out)], check=False)
    else:  # exportar un split, o todos
        splits = args if args else SPLITS
        for split in splits:
            exportar_split(yolo_dir, split)
        print(f"\nAbri la carpeta {yolo_dir / 'previews'} y navega cualquiera.")


if __name__ == "__main__":
    main()
