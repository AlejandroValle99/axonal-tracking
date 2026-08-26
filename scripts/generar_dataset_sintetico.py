"""Genera el dataset sintetico real-scale (perfiles calibrados contra datos reales,
ver docs/handover.md) en datasets/{train,val,test}/.

Para cada uno de los 8 perfiles en ../ingenia-kymograph/examples/dataset_profiles/
genera N muestras por split (100 train / 50 val / 50 test por defecto) y las agrega
(append) a datasets/{train,val,test}/ sin tocar lo que ya haya ahi. No escribe
movie.tif (write_movie=False): el export a YOLO (etiquetas_deteccion.exportar_dataset)
solo lee kymograph.tif + positions.csv + config.yaml, y movie.tif a esta escala pesa
~1.1GB/muestra — no hace falta para entrenamiento.

Cada muestra usa un seed determinista (perfil, split) -> no hay overlap entre
llamadas y las corridas son reproducibles.

Uso:
  # Prueba rapida (5 muestras por perfil y split) en un directorio aparte, para
  # validar que todo corre antes de comprometerse a la corrida completa:
  uv run python scripts/generar_dataset_sintetico.py --smoke --outdir-root /tmp/smoke_ds

  # Corrida completa, al datasets/ real (100/50/50 por perfil):
  uv run python scripts/generar_dataset_sintetico.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

PROFILES_DIR = RAIZ.parent / "ingenia-kymograph" / "examples" / "dataset_profiles"

# (nombre del perfil, seed base propio del perfil, usado para derivar seeds por split)
PROFILES = [
    ("wt_n1", 101),
    ("wt_n2", 102),
    ("wt_n3", 103),
    ("lbd_n1", 201),
    ("lbd_n2", 202),
    ("lbd_n3", 203),
    ("mixed_i3_crowded", 301),
    ("static_dominant_edge", 401),
]

# split -> (nombre de carpeta, offset de seed, n por defecto)
SPLITS = [
    ("train", 0, 100),
    ("val", 2000, 50),
    ("test", 4000, 50),
]


def base_seed_for(profile_seed: int, split_offset: int) -> int:
    """Seed determinista y sin overlap entre (perfil, split): el hueco entre
    offsets (2000) es muy superior al maximo de muestras por llamada (100),
    asi que seed=base+i nunca choca entre perfiles ni entre splits."""
    return profile_seed * 10_000 + split_offset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="5 muestras por perfil y split, en vez de 100/50/50.")
    ap.add_argument("--outdir-root", default=None,
                    help="Donde escribir train/val/test. Default: datasets/ de este repo. "
                         "Usar un directorio aparte para --smoke.")
    args = ap.parse_args()

    from synthkymo.pipeline import generate_dataset, next_dataset_index
    import yaml

    outdir_root = Path(args.outdir_root) if args.outdir_root else (RAIZ / "datasets")
    outdir_root.mkdir(parents=True, exist_ok=True)

    splits = [(name, off, 5 if args.smoke else n) for name, off, n in SPLITS]

    print(f"outdir_root = {outdir_root}")
    print(f"mode = {'SMOKE (5/5/5 per profile)' if args.smoke else 'FULL (100/50/50 per profile)'}")
    print(f"profiles = {[p for p, _ in PROFILES]}")
    print()

    t_start = time.time()
    totals = {name: 0 for name, _, _ in splits}
    for profile_name, profile_seed in PROFILES:
        cfg_path = PROFILES_DIR / f"{profile_name}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing profile config: {cfg_path}")
        config = yaml.safe_load(cfg_path.read_text())

        for split_name, split_offset, n in splits:
            split_dir = outdir_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            base_index = next_dataset_index(split_dir)
            base_seed = base_seed_for(profile_seed, split_offset)

            t0 = time.time()
            manifest = generate_dataset(
                config, n, split_dir,
                base_seed=base_seed, base_index=base_index, append=True,
                write_movie=False,
            )
            dt = time.time() - t0
            totals[split_name] += len(manifest)
            print(f"[{split_name:5s}] {profile_name:22s} n={len(manifest):3d}  "
                  f"seeds {base_seed}-{base_seed + n - 1}  "
                  f"{dt:.1f}s ({dt / max(len(manifest), 1):.2f}s/sample)")

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed / 60:.1f} min. Totals: " +
          ", ".join(f"{k}={v}" for k, v in totals.items()))
    for split_name, _, _ in splits:
        split_dir = outdir_root / split_name
        print(f"  {split_dir} -> {next_dataset_index(split_dir)} samples total (manifest.csv)")


if __name__ == "__main__":
    main()
