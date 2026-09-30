"""Precalcula el cache de targets de KymoRoPE para un dataset, en paralelo.

`construir_entrenador` lo construye solo si falta, pero en serie y dentro del kernel: con el
dataset de 30 000 muestras son ~30 min de CPU. Este script lo hace antes, en varios
procesos y fuera del notebook, y deja todo listo para que el entrenamiento arranque directo.

El cache va a `results/kymorope/cache/{split}/<firma>/`, donde la firma codifica formato,
umbral de movil, escala de pixel y de que dataset sale (`DatasetKymografos._firma_cache`):
es la misma carpeta que despues busca `construir_entrenador`, y cambiar cualquiera de esas
cosas apunta a otra carpeta en vez de servir targets viejos.

Uso:
    uv run python scripts/precalcular_cache_kymorope.py --raiz "C:/Users/aleja/Desktop/datasets"
    uv run python scripts/precalcular_cache_kymorope.py --raiz ... --splits val --procesos 4 --limite 200
    uv run python scripts/precalcular_cache_kymorope.py --subconjunto 5000   # etapa 1 del fine-tune
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))

from axonal_tracking.datos_pixel import DatasetKymografos


def main(
    raiz: Path, splits: list[str], procesos: int, limite: int | None, subconjunto: int | None
) -> None:
    cache_base = RAIZ / "results" / "kymorope" / "cache"
    for split in splits:
        # el subconjunto es del train (curva de escala de datos); val se usa entero
        sub = subconjunto if split == "train" else None
        ds = DatasetKymografos(
            raiz / split, limite=limite, subconjunto=sub, cache_dir=cache_base / split
        )
        faltan = sum(1 for i in range(len(ds)) if not ds._ruta_cache(i).exists())
        print(f"{split}: {len(ds)} muestras, faltan {faltan} -> {ds.cache_dir}", flush=True)
        if not faltan:
            continue
        t0 = time.time()
        ds.precalcular(verboso=True, procesos=procesos)
        seg = time.time() - t0
        tam = sum(p.stat().st_size for p in ds.cache_dir.glob("*.npz"))
        print(f"{split}: {faltan} muestras en {seg / 60:.1f} min ({seg / faltan * 1000:.0f} ms/muestra "
              f"con {procesos} procesos) · cache {tam / 1e9:.2f} GB "
              f"({tam / len(ds) / 1e6:.2f} MB/muestra)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raiz", type=Path, default=RAIZ / "datasets",
                    help="carpeta con train/ val/ (default: datasets/ del repo)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N muestras")
    ap.add_argument("--subconjunto", type=int, default=None,
                    help="N muestras del train representativas y anidadas (= subconjunto_train)")
    args = ap.parse_args()
    main(args.raiz, args.splits, args.procesos, args.limite, args.subconjunto)
