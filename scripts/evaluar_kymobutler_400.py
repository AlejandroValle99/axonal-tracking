"""Corre KymoButler sobre las 400 muestras del split de test y evalua a nivel
trayectoria con el MISMO harness que Mask2Former y YOLO+SAM3
(`ev.evaluar_trayectorias_polilineas`), solo moviles.

Motivo: la fila de KymoButler de NB06 sale de un subconjunto por perfil e incluye
estaticas, asi que no es comparable con las filas de `plan/notebooks-08-09-closeout.md`
(YOLO+SAM3) ni con la de `docs/revision-rumbo-vit.md` (Mask2Former). Sin esta corrida no
hay tabla de tesis. Ver `docs/revision-rumbo-vit.md` SS4.

Salida (split `test`, el default): results/kymobutler/trayectorias_400_*.csv + resumen_400.json.
Otros splits escriben `trayectorias_{split}_*.csv` + `resumen_{split}.json`, asi la cifra
publicada de test no se pisa. Todos guardan ademas `polilineas_{split}_{variante}.pkl`
(polilineas por muestra, ya filtradas por variante) para superponerlas en NB13.

Uso: uv run python scripts/evaluar_kymobutler_400.py [--split val] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
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
SALIDA = RAIZ / "results" / "kymobutler"
DEVICE = "cpu"  # default de `main`; `--device` lo cambia (cpu = el que dio la cifra publicada)

# Mismo umbral movil/estatico que NB08/09/10 -- si esto cambia, la comparacion deja de
# ser el mismo criterio (ver docstring de ed.MIN_DESPLAZAMIENTO_PX_MOVIL).
MIN_DESP = ed.MIN_DESPLAZAMIENTO_PX_MOVIL


def cargar_escena(sample_dir: Path) -> dict:
    """Igual que `cargar_sample` de NB06, pero devolviendo el dict que espera el harness.
    La carga vive en `ev.cargar_escena_sintetica`; aca solo se agrega el PNG que lee
    KymoButler (lo importan evaluar_gate_a, preparar_cache y NB11: no renombrar)."""
    escena = ev.cargar_escena_sintetica(sample_dir)
    png = Path(sample_dir) / "kymograph.png"
    if not png.exists():
        Image.fromarray(escena["kymo"], mode="L").save(png)
    return {**escena, "png": png}


def polilineas_kymobutler(
    escena: dict, models, device: str = DEVICE
) -> tuple[list[pd.DataFrame], float]:
    """Tracks de KymoButler -> polilineas (frame, col_subpixel) en pixeles NATIVOS.

    KymoButler redimensiona internamente (`scale_factor`); NB06 compensaba escalando el
    umbral y llevando el GT a su espacio. Aca se hace al reves -- se traen las
    predicciones al espacio nativo -- para que el harness compartido vea las mismas
    unidades que para Mask2Former, sin tocar su logica de asociacion.
    """
    was_negated, raw, pre, pred = segment_bidirectional(
        str(escena["png"]), models["binet"], device=device
    )
    scale_factor = raw.shape[1] / escena["kymo"].shape[1]
    tracks = track_bidirectional(
        pred, pre, was_negated, vision_net=models["decnet"],
        threshold=0.2, min_size=10, min_frames=10, device=device,
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


def rasterizar(poli: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    """Polilinea -> mascara fina (T, L), un pixel por fila."""
    T, L = shape
    m = np.zeros((T, L), dtype=bool)
    fr = poli["frame"].to_numpy().astype(int)
    co = np.round(poli["col_subpixel"].to_numpy()).astype(int)
    ok = (fr >= 0) & (fr < T) & (co >= 0) & (co < L)
    m[fr[ok], co[ok]] = True
    return m


def via_subpixel(poli: pd.DataFrame, kymo: np.ndarray) -> pd.DataFrame:
    """Re-extrae la polilinea por la MISMA ruta que Mask2Former y que las filas 2/3/3b del
    plan de NB11: rasterizar a mascara fina -> `ev.extraer_subpixel`.

    Motivo: KymoButler emite coordenadas enteras de esqueleto y NUNCA pasa por
    `extraer_subpixel`, mientras los pipelines basados en mascara si. Eso les da pisos de
    `frac_id_switch` muy distintos (0.005 vs 0.132, ver `docs/revision-rumbo-vit.md`), asi
    que comparar los numeros crudos entre rutas es invalido. Esta variante pone a
    KymoButler en la ruta ajena para tener una comparacion sin aritmetica de excesos.
    """
    return ev.extraer_subpixel(rasterizar(poli, kymo.shape), kymo)


def es_movil(poli: pd.DataFrame) -> bool:
    """Analogo, del lado de la PREDICCION, del filtro de clase `movil` que se le aplica a
    Mask2Former: KymoButler no emite clase, asi que se usa el mismo criterio de
    desplazamiento de `ed.ids_que_se_mueven` (rango total >= MIN_DESP px). La logica vive
    en `ev.es_movil_polilinea`; este nombre queda porque cuatro scripts lo importan."""
    return ev.es_movil_polilinea(poli, MIN_DESP)


def main(split: str = "test", device: str = DEVICE, limite: int | None = None) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    split_dir = RAIZ / "datasets" / split
    # test conserva los nombres de archivo de la cifra publicada; el resto lleva el split.
    # Una corrida con `limite` lleva su propio sufijo: nunca pisa un resultado completo.
    sufijo = "400" if split == "test" else split
    etiqueta_split = split
    if limite:
        sufijo, etiqueta_split = f"{sufijo}_lim{limite}", f"{split}_lim{limite}"
    models = load_default_models(model_dir=MODELOS_DIR, device=device)
    muestras = sorted(split_dir.glob("sample_*"))[:limite]
    print(f"{len(muestras)} muestras de {split}, device={device}", flush=True)

    escenas, fallos, t0 = {}, [], time.time()
    for i, d in enumerate(muestras):
        try:
            esc = cargar_escena(d)
            polis, sf = polilineas_kymobutler(esc, models, device)
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
    variantes = (
        ("todas", False, False),
        ("solo_moviles", True, False),
        # misma ruta de extraccion que Mask2Former / filas 2-3b del plan de NB11
        ("solo_moviles_subpixel", True, True),
    )
    for etiqueta, filtrar, subpix in variantes:
        esc_ev = {}
        for n, e in escenas.items():
            polis = [p for p in e["polilineas"] if not filtrar or es_movil(p)]
            if subpix:
                polis = [q for q in (via_subpixel(p, e["kymo"]) for p in polis) if len(q)]
            esc_ev[n] = {**e, "polilineas": polis}
        tr, res = ev.evaluar_trayectorias_polilineas(esc_ev, min_desplazamiento_px=MIN_DESP)
        res = {**res, **ev.resumen_fragmentacion(tr)}
        resultados[etiqueta] = res
        tr.to_csv(SALIDA / f"trayectorias_{sufijo}_{etiqueta}.csv", index=False)
        with open(SALIDA / f"polilineas_{etiqueta_split}_{etiqueta}.pkl", "wb") as f:
            pickle.dump({n: e["polilineas"] for n, e in esc_ev.items()}, f)
        print(f"\n=== KymoButler, {len(escenas)} muestras de {split}, predicciones: {etiqueta} ===")
        for k, v in res.items():
            print(f"  {k}: {v}")

    (SALIDA / f"resumen_{sufijo}.json").write_text(json.dumps(
        {"resultados": resultados, "split": split, "device": device,
         "n_muestras": len(escenas), "fallos": fallos,
         "min_desplazamiento_px": MIN_DESP, "thr_px_track": ev.THR_PX_TRACK}, indent=2))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="test", help="test (default, cifra publicada), val o train")
    ap.add_argument("--device", default=DEVICE, help="cpu (default) o cuda")
    ap.add_argument("--limite", type=int, default=None,
                    help="solo las primeras N muestras (prueba rapida; archivos con sufijo _limN)")
    args = ap.parse_args()
    main(args.split, args.device, args.limite)
