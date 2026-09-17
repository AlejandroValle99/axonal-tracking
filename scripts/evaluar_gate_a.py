"""Gate A (`plan/association-transformer-guide.md` SS3.4): antes de escribir
ningun modelo de asociacion, mide si la representacion "segmentos de esqueleto +
enlace oraculo" alcanza para resolver trayectorias en kymografos sinteticos.

Corre, sobre las 400 muestras de `datasets/test`:

1. El piso de `frac_id_switch` (SS12.9) en cuatro rutas de extraccion -- referencia
   para saber cuanto de cualquier numero de mas abajo es piso de metrica y no falla
   de la representacion.
2. El enlazador oraculo (exclusivo y duplicado, SS3.3) sobre tres fuentes de
   trackness: mascaras GT, KymoButler, y umbral clasico (NB08 SS1) -- las tres
   pasan por el MISMO `partir_en_segmentos` / `fusionar_fragmentos_paralelos` /
   `asignar_gt` de `axonal_tracking.asociacion`.

Salida: `results/asociacion/piso.json`, `results/asociacion/gate_a_{fuente}*.csv`,
`results/asociacion/gate_a_resumen.json`. La prediccion/preprocesado de KymoButler
se cachea en `results/asociacion/cache/` (float32) porque SS12.8 dice que se va a
recorrer el split varias veces mientras se ajusta `min_filas`, y `preprocessed` es
tambien lo que v2 (SS5.6) va a necesitar mas adelante.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import kymobutler
from evaluar_kymobutler_400 import cargar_escena  # SS12.2 -- no reescribir
from kymobutler.models.weights import load_default_models
from kymobutler.segmentation import segment_bidirectional

from axonal_tracking import asociacion as aso
from axonal_tracking import etiquetas_deteccion as ed
from axonal_tracking import evaluacion as ev

KYMOBUTLER_DIR = Path(kymobutler.__file__).resolve().parents[2]
MODELOS_DIR = KYMOBUTLER_DIR / "models"
SPLIT_DIR = RAIZ / "datasets" / "test"
CACHE_DIR = RAIZ / "results" / "asociacion" / "cache" / "test"  # namespaced por split (Stage 1 tambien cachea "train")
SALIDA = RAIZ / "results" / "asociacion"
DEVICE = "cpu"

MIN_DESP = ed.MIN_DESPLAZAMIENTO_PX_MOVIL  # 8.0 -- no cambiar (SS12.4)
# Identicos a los defaults de `track_bidirectional` (SS3.1/SS12.4) -- rompe la
# igualdad de esqueleto entre filas 1 y 3/3b del ablation matrix si se tocan.
THRESHOLD, MIN_SIZE, MIN_FRAMES = 0.2, 10, 10
MIN_FILAS_SEGMENTO = 5
MAX_DX_FUSION_PX = 3.0
PUREZA_MIN_COMPARTIDO = 0.9


# --------------------------------------------------------------------------- #
# SS12.9 -- piso de frac_id_switch (corre primero: barato, calibra Gate A)
# --------------------------------------------------------------------------- #
def medir_piso() -> dict:
    escenas_dilatada, escenas_1px, escenas_entero, escenas_exacta = {}, {}, {}, {}
    for d in sorted(SPLIT_DIR.glob("sample_*")):
        e = cargar_escena(d)
        T, L = e["kymo"].shape
        px = e["pixel_scale_um"]
        ids = ed.ids_que_se_mueven(e["positions"], px, MIN_DESP)
        masks = [ed.mascara_traza(e["positions"], p, px, (T, L)) for p in ids]
        polis_dilatada = [ev.extraer_subpixel(m, e["kymo"]) for m in masks]
        escenas_dilatada[d.name] = {**e, "polilineas": [q for q in polis_dilatada if len(q)]}

        polis_1px, polis_entero, polis_exacta = [], [], []
        for pid in ids:
            sub = e["positions"][e["positions"]["particle_id"] == pid].sort_values("frame")
            if "visible" in sub.columns:
                sub = sub[sub["visible"].astype(bool)]
            if len(sub) == 0:
                continue
            filas = sub["frame"].to_numpy()
            cols = (L - 1) - sub["pos_um"].to_numpy() / px
            puntos = np.stack([filas, cols], axis=1).astype(float)
            polis_1px.append(aso.polilinea_desde_puntos(puntos, (T, L), e["kymo"]))
            polis_entero.append(pd.DataFrame({
                "frame": filas.astype(int), "col_subpixel": np.round(cols),
            }))
            polis_exacta.append(pd.DataFrame({"frame": filas.astype(int), "col_subpixel": cols}))
        escenas_1px[d.name] = {**e, "polilineas": [q for q in polis_1px if len(q)]}
        escenas_entero[d.name] = {**e, "polilineas": polis_entero}
        escenas_exacta[d.name] = {**e, "polilineas": polis_exacta}

    resultados = {}
    for nombre, escenas in (
        ("dilatada_subpixel", escenas_dilatada),
        ("1px_subpixel", escenas_1px),
        ("1px_entero_sin_centroide", escenas_entero),
        ("exacta", escenas_exacta),
    ):
        _, res = ev.evaluar_trayectorias_polilineas(escenas, min_desplazamiento_px=MIN_DESP)
        resultados[nombre] = res
        print(f"  piso [{nombre}]: track_f1={res['track_f1']}  "
              f"frac_id_switch={res['frac_id_switch']}  err_pos_um={res['err_pos_um_medio']}")
    return resultados


# --------------------------------------------------------------------------- #
# Las tres fuentes de trackness de SS3.3
# --------------------------------------------------------------------------- #
def predecir_kymobutler(escena: dict, models) -> dict:
    """`prediction`/`preprocessed` de KymoButler, cacheados a disco (float32) --
    ver SS12.8: el split se recorre varias veces mientras se ajusta `min_filas`, y
    `preprocessed` es tambien el insumo de los tiles de v2 (SS5.6)."""
    cache = CACHE_DIR / f"{escena['nombre']}.npz"
    if cache.exists():
        d = np.load(cache)
        return {
            "prediction": d["prediction"], "preprocessed": d["preprocessed"],
            "scale_factor": float(d["scale_factor"]),
        }
    _, raw, pre, pred = segment_bidirectional(str(escena["png"]), models["binet"], device=DEVICE)
    scale_factor = raw.shape[1] / escena["kymo"].shape[1]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        prediction=pred.astype(np.float32),
        preprocessed=pre.astype(np.float32),
        scale_factor=np.float32(scale_factor),
    )
    return {"prediction": pred, "preprocessed": pre, "scale_factor": scale_factor}


def mascara_clasica(kymo: np.ndarray, cierre_radio: int = 1) -> np.ndarray:
    """Mismo umbral+cierre que `detectar_clasico` de NB08 SS1, pero devolviendo la
    mascara binaria antes de reducirla a cajas -- eso es lo que `esqueletizar`
    (fuente 3 de SS3.3) necesita, no una lista de cajas."""
    a = np.asarray(kymo, dtype=float)
    return closing(a > threshold_otsu(a), disk(cierre_radio))


# --------------------------------------------------------------------------- #
# Pipeline comun: esqueleto -> segmentos -> asignacion GT -> oraculo
# --------------------------------------------------------------------------- #
def procesar_fuente(skel: np.ndarray, escena: dict) -> dict:
    segmentos = aso.partir_en_segmentos(skel, min_filas=MIN_FILAS_SEGMENTO)
    n_antes = len(segmentos)
    segmentos = aso.fusionar_fragmentos_paralelos(
        segmentos, min_filas=MIN_FILAS_SEGMENTO, max_dx_px=MAX_DX_FUSION_PX
    )
    L = escena["kymo"].shape[1]
    asignaciones = aso.asignar_gt(segmentos, escena["positions"], escena["pixel_scale_um"], L)
    polis_excl = aso.enlazar_oracle(segmentos, asignaciones, escena["kymo"])
    polis_dup = aso.enlazar_oracle_duplicado(
        segmentos, asignaciones, escena["positions"], escena["pixel_scale_um"], escena["kymo"],
        umbral_compartido=PUREZA_MIN_COMPARTIDO,
    )
    n_asignados = sum(1 for a in asignaciones if a.particle_id != -1)
    n_pureza_baja = sum(
        1 for a in asignaciones if a.particle_id != -1 and a.pureza < PUREZA_MIN_COMPARTIDO
    )
    return {
        "n_segmentos": len(segmentos),
        "n_fusiones": n_antes - len(segmentos),
        "n_asignados": n_asignados,
        "n_pureza_baja": n_pureza_baja,
        "polilineas_exclusiva": polis_excl,
        "polilineas_duplicada": polis_dup,
    }


def main() -> None:
    print("=== SS12.9: piso de frac_id_switch (referencia) ===")
    piso = medir_piso()

    print("\n=== Gate A: tres fuentes de trackness, 400 muestras de test ===")
    models = load_default_models(model_dir=MODELOS_DIR, device=DEVICE)
    muestras = sorted(SPLIT_DIR.glob("sample_*"))
    print(f"{len(muestras)} muestras, device={DEVICE}", flush=True)

    fuentes = ("gt_mascaras", "gt_lineas", "kymobutler", "clasica")
    escenas_excl = {f: {} for f in fuentes}
    escenas_dup = {f: {} for f in fuentes}
    diagnosticos = {f: [] for f in fuentes}
    fallos = []
    t0 = time.time()

    for i, d in enumerate(muestras):
        try:
            e = cargar_escena(d)
            T, L = e["kymo"].shape

            trackness_gt = aso.trackness_gt_moviles(e["positions"], e["pixel_scale_um"], (T, L))
            skel_gt = aso.esqueleto_kymobutler(trackness_gt, (T, L), THRESHOLD, MIN_SIZE, MIN_FRAMES)

            # Diagnostico Gate A (advisor, ver docstring de trackness_gt_moviles): la
            # misma fuente sin la dilatacion de `mascara_traza`, para aislar si el
            # solape de blobs dilatados -- no la representacion de segmentos -- es lo
            # que funde particulas en cruces (n_segmentos_medio=12.1 para ~13 moviles
            # era la senal: casi ninguna juncion se estaba formando).
            trackness_gt_lineas = aso.trackness_gt_moviles(
                e["positions"], e["pixel_scale_um"], (T, L), dilatar=False
            )
            skel_gt_lineas = aso.esqueleto_kymobutler(
                trackness_gt_lineas, (T, L), THRESHOLD, MIN_SIZE, MIN_FRAMES
            )

            pred_info = predecir_kymobutler(e, models)
            pre = pred_info["preprocessed"]
            assert pre.shape == e["kymo"].shape, (
                f"{d.name}: preprocessed.shape {pre.shape} != kymo.shape {e['kymo'].shape} "
                f"(scale_factor={pred_info['scale_factor']:.3f}) -- ver SS12.1/SS9.7"
            )
            skel_kb = aso.esqueleto_kymobutler(
                pred_info["prediction"], pre.shape, THRESHOLD, MIN_SIZE, MIN_FRAMES
            )

            skel_clasica = aso.esqueletizar(mascara_clasica(e["kymo"]))

            for fuente, skel in (
                ("gt_mascaras", skel_gt), ("gt_lineas", skel_gt_lineas),
                ("kymobutler", skel_kb), ("clasica", skel_clasica),
            ):
                diag = procesar_fuente(skel, e)
                diag["muestra"] = d.name
                diagnosticos[fuente].append({k: v for k, v in diag.items()
                                              if not k.startswith("polilineas")})
                escenas_excl[fuente][d.name] = {**e, "polilineas": diag["polilineas_exclusiva"]}
                escenas_dup[fuente][d.name] = {**e, "polilineas": diag["polilineas_duplicada"]}
        except Exception as exc:  # noqa: BLE001 -- una muestra rota no debe tumbar 400
            fallos.append({"muestra": d.name, "error": repr(exc)})
            traceback.print_exc()
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(muestras)}  {el/60:.1f} min  "
                  f"(~{el/(i+1)*(len(muestras)-i-1)/60:.1f} min restantes)", flush=True)

    print(f"\n{sum(len(v) for v in escenas_excl.values())//len(fuentes)} ok, "
          f"{len(fallos)} fallos, {time.time()-t0:.0f}s total", flush=True)

    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = {"piso": piso, "fuentes": {}, "n_muestras": len(muestras), "fallos": fallos}
    for fuente in fuentes:
        diags = pd.DataFrame(diagnosticos[fuente])
        n_asignados_total = int(diags["n_asignados"].sum())
        n_pureza_baja_total = int(diags["n_pureza_baja"].sum())

        res_fuente = {}
        for variante, escenas in (("exclusiva", escenas_excl[fuente]), ("duplicada", escenas_dup[fuente])):
            tr, res = ev.evaluar_trayectorias_polilineas(escenas, min_desplazamiento_px=MIN_DESP)
            mm = tr[tr.gt_id != -1]
            frag = mm.groupby(["muestra", "gt_id"]).size()
            res = {**res, "fragmentos_por_gt": round(float(frag.mean()), 3) if len(frag) else None}
            tr.to_csv(SALIDA / f"gate_a_{fuente}_{variante}.csv", index=False)
            res_fuente[variante] = res

        res_fuente["n_segmentos_total"] = int(diags["n_segmentos"].sum())
        res_fuente["n_segmentos_medio"] = round(float(diags["n_segmentos"].mean()), 1)
        res_fuente["n_fusiones_total"] = int(diags["n_fusiones"].sum())
        res_fuente["frac_pureza_baja"] = (
            round(n_pureza_baja_total / n_asignados_total, 4) if n_asignados_total else None
        )
        diags.to_csv(SALIDA / f"gate_a_{fuente}_diagnostico.csv", index=False)
        resumen["fuentes"][fuente] = res_fuente

        print(f"\n=== fuente: {fuente} ===")
        print(f"  n_segmentos_total={res_fuente['n_segmentos_total']}  "
              f"n_segmentos_medio={res_fuente['n_segmentos_medio']}  "
              f"n_fusiones_total={res_fuente['n_fusiones_total']}  "
              f"frac_pureza_baja={res_fuente['frac_pureza_baja']}")
        for variante in ("exclusiva", "duplicada"):
            r = res_fuente[variante]
            print(f"  [{variante}] track_f1={r['track_f1']}  frac_id_switch={r['frac_id_switch']}  "
                  f"err_pos_um={r['err_pos_um_medio']}  fragmentos_por_gt={r['fragmentos_por_gt']}")

    (SALIDA / "gate_a_resumen.json").write_text(json.dumps(resumen, indent=2, default=str))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
