"""Calienta la cache de predicciones de KymoButler (`results/asociacion/cache/{split}/`)
para un split arbitrario. `scripts/evaluar_gate_a.py` ya calienta `test`; Stage 1
(`plan/association-transformer-guide.md` SS4) necesita `train` para ajustar los pesos
del costo clasico -- nunca en test (SS4.2). Script separado para poder correrlo en
background mientras se escribe el resto de Stage 1 (~15 min para 800 muestras de train).

Uso: uv run python scripts/preparar_cache_kymobutler.py train
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import kymobutler
from evaluar_kymobutler_400 import cargar_escena  # SS12.2 -- no reescribir
from kymobutler.models.weights import load_default_models
from kymobutler.segmentation import segment_bidirectional

MODELOS_DIR = Path(kymobutler.__file__).resolve().parents[2] / "models"
DEVICE = "cpu"


def main(split: str) -> None:
    split_dir = RAIZ / "datasets" / split
    cache_dir = RAIZ / "results" / "asociacion" / "cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    models = load_default_models(model_dir=MODELOS_DIR, device=DEVICE)
    muestras = sorted(split_dir.glob("sample_*"))
    print(f"{len(muestras)} muestras en {split}, cache={cache_dir}", flush=True)

    t0 = time.time()
    n_nuevas = 0
    for i, d in enumerate(muestras):
        cache = cache_dir / f"{d.name}.npz"
        if cache.exists():
            continue
        e = cargar_escena(d)
        _, raw, pre, pred = segment_bidirectional(str(e["png"]), models["binet"], device=DEVICE)
        scale_factor = raw.shape[1] / e["kymo"].shape[1]
        np.savez_compressed(
            cache,
            prediction=pred.astype(np.float32),
            preprocessed=pre.astype(np.float32),
            scale_factor=np.float32(scale_factor),
        )
        n_nuevas += 1
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(muestras)}  {el/60:.1f} min", flush=True)

    print(f"listo: {n_nuevas} nuevas, {len(muestras) - n_nuevas} ya en cache, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
