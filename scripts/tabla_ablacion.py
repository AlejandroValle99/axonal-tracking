"""Tabla de ablacion final: filas 1 / 2 / 3 / 3b / 3c + techo del oraculo, agregada y
estratificada por dificultad, desde los CSV por-polilinea ya guardados.

Reemplaza el calculo embebido en `notebooks/11_asociacion_atencion.ipynb` SS6, que tiene
un bug en el denominador del recall: usa `mm.groupby("muestra").gt_id.nunique()` -- las GT
**recuperadas por la prediccion** -- en vez de las GT moviles reales de `positions.csv`.
Con ese denominador el recall es recuperadas/recuperadas y el `track_f1` estratificado sale
~1.0 en todos los buckets por construccion, escondiendo diferencias reales de recall entre
filas (medido: 0.875 para atencion v1 contra 0.950 para DecNet en el bucket sin cruces).

Estratificacion por `ev.flaggear_cruces_gt` (lado GT, no lado prediccion -- ver
`docs/revision-rumbo-vit.md` SS1 para por que el flag de prediccion esta roto).

Uso: uv run python scripts/tabla_ablacion.py
Salida: `results/asociacion/tabla_ablacion.json` + la tabla por stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import tifffile
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))

from axonal_tracking import etiquetas_deteccion as ed
from axonal_tracking import evaluacion as ev

SPLIT = RAIZ / "datasets" / "test"
ASOC = RAIZ / "results" / "asociacion"
ORDEN = ["0", "1-2", "3-5", ">5"]

FILAS = [
    ("1: DecNet", RAIZ / "results" / "kymobutler" / "trayectorias_400_solo_moviles_subpixel.csv"),
    ("2: costo clasico", ASOC / "costo_clasico_fila2.csv"),
    ("3: atencion v1", ASOC / "atencion_v1_fila3.csv"),
    ("3b: atencion v2", ASOC / "atencion_v2_fila3b.csv"),
    ("3c: atencion v3 (dustbin, margen=1.0)", ASOC / "atencion_v3_fila3c_margen_1.0.csv"),
    ("3c': atencion v3 (punto Gate B)", ASOC / "atencion_v3_fila3c_gate_b.csv"),
    ("3d: v3 + estaticos->dustbin (margen=1.0)", ASOC / "atencion_v3_fila3c_margen_1.0_est.csv"),
    ("3d': v3 + estaticos (punto Gate B)", ASOC / "atencion_v3_fila3c_gate_b_est.csv"),
    ("techo: oraculo", ASOC / "gate_a_kymobutler_exclusiva.csv"),
]


def bucket(n: int) -> str:
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else ">5"


def metadatos_test() -> pd.DataFrame:
    """Por muestra: bucket de dificultad y **numero real de GT moviles** (el denominador
    honesto del recall)."""
    filas = []
    for d in sorted(SPLIT.glob("sample_*")):
        positions = pd.read_csv(d / "positions.csv")
        px = float(yaml.safe_load((d / "config.yaml").read_text())["general"]["pixel_scale_um"])
        kymo = tifffile.imread(d / "kymograph.tif")
        if kymo.ndim == 3:
            kymo = kymo[..., 0]
        ambiguos = ev.flaggear_cruces_gt(positions, px, kymo.shape, ed.MIN_DESPLAZAMIENTO_PX_MOVIL)
        n_mov = len(ed.ids_que_se_mueven(positions, px, ed.MIN_DESPLAZAMIENTO_PX_MOVIL))
        filas.append({"muestra": d.name, "n_ambiguos": len(ambiguos),
                      "bucket": bucket(len(ambiguos)), "n_gt_moviles": n_mov})
    return pd.DataFrame(filas)


def metricas(sub: pd.DataFrame, n_gt: int) -> dict:
    if len(sub) == 0 or n_gt == 0:
        return {k: None for k in ("track_f1", "track_recall", "track_precision",
                                  "frac_id_switch", "fragmentos_por_gt", "err_pos_um", "n_polilineas")}
    mm = sub[sub.gt_id != -1]
    frag = mm.groupby(["muestra", "gt_id"]).size()
    recall = len(frag) / n_gt  # GT recuperadas / GT REALES (el bug de NB11 SS6 estaba aca)
    precision = len(mm) / len(sub)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "track_f1": round(f1, 3), "track_recall": round(recall, 3),
        "track_precision": round(precision, 3),
        "frac_id_switch": round(float(sub.id_switch.mean()), 3),
        "fragmentos_por_gt": round(float(frag.mean()), 3) if len(frag) else None,
        "err_pos_um": round(float(mm.err_um.mean()), 3) if len(mm) else None,
        "n_polilineas": len(sub),
    }


def main() -> None:
    meta = metadatos_test()
    print(f"test: {len(meta)} muestras, {meta.n_gt_moviles.sum()} GT moviles")
    print("  por bucket:", meta.groupby("bucket").agg(
        n_muestras=("muestra", "size"), n_gt=("n_gt_moviles", "sum")).reindex(ORDEN).to_dict("index"))

    agregado, estratificado, faltantes = {}, {}, []
    for nombre, path in FILAS:
        if not path.exists():
            faltantes.append(nombre)
            continue
        tr = pd.read_csv(path).merge(meta, on="muestra", how="left")
        agregado[nombre] = metricas(tr, int(meta.n_gt_moviles.sum()))
        estratificado[nombre] = {
            b: metricas(tr[tr.bucket == b], int(meta[meta.bucket == b].n_gt_moviles.sum()))
            for b in ORDEN
        }

    def imprimir(metrica: str) -> None:
        print(f"\n=== {metrica} ===")
        df = pd.DataFrame({n: {b: estratificado[n][b][metrica] for b in ORDEN} for n in estratificado}).T
        df.insert(0, "GLOBAL", [agregado[n][metrica] for n in estratificado])
        print(df.to_string())

    for m in ("frac_id_switch", "fragmentos_por_gt", "track_f1", "track_recall"):
        imprimir(m)
    if faltantes:
        print(f"\n(filas ausentes, CSV no generado aun: {', '.join(faltantes)})")

    (ASOC / "tabla_ablacion.json").write_text(json.dumps(
        {"agregado": agregado, "estratificado": estratificado,
         "n_muestras_por_bucket": meta.groupby("bucket").size().reindex(ORDEN).to_dict(),
         "n_gt_moviles_por_bucket": meta.groupby("bucket").n_gt_moviles.sum().reindex(ORDEN).to_dict(),
         "filas_ausentes": faltantes}, indent=2, default=str))
    print(f"\nGuardado en {ASOC / 'tabla_ablacion.json'}")


if __name__ == "__main__":
    main()
