"""Corre KymoButler sobre las 400 muestras del split de test y evalua a nivel
trayectoria con el MISMO harness que Mask2Former y YOLO+SAM3
(`ev.evaluar_trayectorias_polilineas`), solo moviles.

Motivo: la fila de KymoButler de NB06 sale de un subconjunto por perfil e incluye
estaticas, asi que no es comparable con las filas de `plan/notebooks-08-09-closeout.md`
(YOLO+SAM3) ni con la de `docs/revision-rumbo-vit.md` (Mask2Former). Sin esta corrida no
hay tabla de tesis. Ver `docs/revision-rumbo-vit.md` SS4.

Salida: results/kymobutler/trayectorias_400.csv + resumen_400.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))

import kymobutler
from kymobutler.models.weights import load_default_models
from kymobutler.segmentation import segment_bidirectional
from kymobutler.tracking import track_bidirectional

from axonal_tracking import etiquetas_deteccion as ed
from axonal_tracking import evaluacion as ev

KYMOBUTLER_DIR = Path(kymobutler.__file__).resolve().parents[2]
MODELOS_DIR = KYMOBUTLER_DIR / "models"
SPLIT_DIR = RAIZ / "datasets" / "test"
SALIDA = RAIZ / "results" / "kymobutler"
DEVICE = "cpu"

# Mismo umbral movil/estatico que NB08/09/10 -- si esto cambia, la comparacion deja de
# ser el mismo criterio (ver docstring de ed.MIN_DESPLAZAMIENTO_PX_MOVIL).
MIN_DESP = ed.MIN_DESPLAZAMIENTO_PX_MOVIL


def cargar_escena(sample_dir: Path) -> dict:
    """Igual que `cargar_sample` de NB06, pero devolviendo el dict que espera el harness."""
    kymo = tifffile.imread(sample_dir / "kymograph.tif")
    if kymo.ndim == 3:
        kymo = kymo[..., 0]
    png = sample_dir / "kymograph.png"
    if not png.exists():
        Image.fromarray(kymo, mode="L").save(png)
    cfg = yaml.safe_load((sample_dir / "config.yaml").read_text())
    return {
        "nombre": sample_dir.name,
        "kymo": kymo,
        "png": png,
        "positions": pd.read_csv(sample_dir / "positions.csv"),
        "pixel_scale_um": float(cfg["general"]["pixel_scale_um"]),
        "fps": float(cfg["general"]["fps"]),
    }


def polilineas_kymobutler(escena: dict, models) -> tuple[list[pd.DataFrame], float]:
    """Tracks de KymoButler -> polilineas (frame, col_subpixel) en pixeles NATIVOS.

    KymoButler redimensiona internamente (`scale_factor`); NB06 compensaba escalando el
    umbral y llevando el GT a su espacio. Aca se hace al reves -- se traen las
    predicciones al espacio nativo -- para que el harness compartido vea las mismas
    unidades que para Mask2Former, sin tocar su logica de asociacion.
    """
    was_negated, raw, pre, pred = segment_bidirectional(
        str(escena["png"]), models["binet"], device=DEVICE
    )
    scale_factor = raw.shape[1] / escena["kymo"].shape[1]
    tracks = track_bidirectional(
        pred, pre, was_negated, vision_net=models["decnet"],
        threshold=0.2, min_size=10, min_frames=10, device=DEVICE,
    )
    polis = []
    for trk in tracks:
        pts = np.asarray(trk.points, dtype=float)  # (n, 2) = (t, x) en espacio KymoButler
        if len(pts) == 0:
            continue
        polis.append(pd.DataFrame({
            "frame": np.round(pts[:, 0] / scale_factor).astype(int),
            "col_subpixel": pts[:, 1] / scale_factor,
        }))
    return polis, scale_factor


def es_movil(poli: pd.DataFrame) -> bool:
    """Analogo, del lado de la PREDICCION, del filtro de clase `movil` que se le aplica a
    Mask2Former: KymoButler no emite clase, asi que se usa el mismo criterio de
    desplazamiento de `ed.ids_que_se_mueven` (rango total >= MIN_DESP px)."""
    c = poli["col_subpixel"].to_numpy()
    return bool(len(c) and (c.max() - c.min()) >= MIN_DESP)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    models = load_default_models(model_dir=MODELOS_DIR, device=DEVICE)
    muestras = sorted(SPLIT_DIR.glob("sample_*"))
    print(f"{len(muestras)} muestras, device={DEVICE}", flush=True)

    escenas, fallos, t0 = {}, [], time.time()
    for i, d in enumerate(muestras):
        try:
            esc = cargar_escena(d)
            polis, sf = polilineas_kymobutler(esc, models)
            escenas[d.name] = {**esc, "polilineas": polis, "scale_factor": sf}
        except Exception as exc:  # noqa: BLE001 -- una muestra rota no debe tumbar 400
            fallos.append({"muestra": d.name, "error": repr(exc)})
            traceback.print_exc()
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(muestras)}  {el/60:.1f} min  "
                  f"(~{el/(i+1)*(len(muestras)-i-1)/60:.1f} min restantes)", flush=True)

    print(f"\n{len(escenas)} ok, {len(fallos)} fallos, "
          f"{time.time()-t0:.0f}s total", flush=True)
    sfs = np.array([e["scale_factor"] for e in escenas.values()])
    print(f"scale_factor: min={sfs.min():.3f} mediana={np.median(sfs):.3f} max={sfs.max():.3f}")

    resultados = {}
    for etiqueta, filtrar in (("todas", False), ("solo_moviles", True)):
        esc_ev = {
            n: {**e, "polilineas": [p for p in e["polilineas"] if not filtrar or es_movil(p)]}
            for n, e in escenas.items()
        }
        tr, res = ev.evaluar_trayectorias_polilineas(esc_ev, min_desplazamiento_px=MIN_DESP)
        resultados[etiqueta] = res
        tr.to_csv(SALIDA / f"trayectorias_400_{etiqueta}.csv", index=False)
        print(f"\n=== KymoButler, 400 muestras, predicciones: {etiqueta} ===")
        for k, v in res.items():
            print(f"  {k}: {v}")

    (SALIDA / "resumen_400.json").write_text(json.dumps(
        {"resultados": resultados, "n_muestras": len(escenas), "fallos": fallos,
         "min_desplazamiento_px": MIN_DESP, "thr_px_track": ev.THR_PX_TRACK}, indent=2))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
