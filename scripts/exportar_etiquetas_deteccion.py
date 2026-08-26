"""Exporta el dataset de synthkymo (datasets/{train,val,test}/sample_*/) a formato
YOLO en datasets/yolo_detection/ (images + labels + kymograph_detection.yaml).

Correr DESPUES de generar/agregar muestras con la UI de synthkymo, y ANTES de
`ver_etiquetas_deteccion.py`. Es idempotente: se puede re-correr cuando cambian
las muestras o el modulo `etiquetas_deteccion`.

Uso:
  uv run python scripts/exportar_etiquetas_deteccion.py                    # los 3 splits, 2 clases
  uv run python scripts/exportar_etiquetas_deteccion.py train              # solo un split
  uv run python scripts/exportar_etiquetas_deteccion.py --movers-only      # solo clase movil,
                                                                            # escribe en yolo_detection_movers/
  uv run python scripts/exportar_etiquetas_deteccion.py --min-desplazamiento-px 8
                                                                            # umbral movil/estatico
                                                                            # (default: 1.0, ver docs/handover.md SS2.3)
"""
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from axonal_tracking import etiquetas_deteccion as ed

DATASETS = RAIZ / "datasets"
SPLITS = ("train", "val", "test")


def main() -> None:
    args = sys.argv[1:]
    movers_only = "--movers-only" in args
    min_desplazamiento_px = 1.0
    if "--min-desplazamiento-px" in args:
        i = args.index("--min-desplazamiento-px")
        min_desplazamiento_px = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    splits = [a for a in args if a != "--movers-only"] or list(SPLITS)
    out = DATASETS / ("yolo_detection_movers" if movers_only else "yolo_detection")
    print(f"min_desplazamiento_px = {min_desplazamiento_px}")

    for sp in splits:
        src = DATASETS / sp
        if not src.is_dir():
            print(f"  {sp}: no existe {src}, salteado")
            continue
        n = ed.exportar_dataset(
            src, out, sp,
            etiquetar_estaticos=not movers_only,
            min_desplazamiento_px=min_desplazamiento_px,
        )
        c = Counter()
        for t in (out / "labels" / sp).glob("*.txt"):
            for linea in t.read_text().splitlines():
                if linea.strip():
                    c[int(linea.split()[0])] += 1
        print(f"  {sp}: {n} muestras | movil(0)={c[0]}  estatico(1)={c[1]}")
    ed.escribir_yaml_dataset(out)
    print(f"\nYAML: {out / 'kymograph_detection.yaml'}")
    print("Ahora, para ver las cajas dibujadas:")
    print(f"  uv run python scripts/ver_etiquetas_deteccion.py{' --movers-only' if movers_only else ''}")


if __name__ == "__main__":
    main()
