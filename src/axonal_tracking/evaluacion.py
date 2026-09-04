"""Metricas de segmentacion/tracking sobre kymografos, extraidas de
`notebooks/09_segmentacion_transformer.ipynb` SS5-7 para reusarlas en
`notebooks/10_mask2former_kymografo.ipynb` sin duplicar codigo ni tocar el notebook
congelado (ver `plan/notebooks-08-09-closeout.md`).

**La matematica/logica es la misma que NB09**; lo unico que cambia es la forma: las
funciones de NB09 eran closures sobre variables globales del notebook (`escenas`, rutas
a `config.yaml` de `TEST_DIR`) -- aca esas dependencias son parametros explicitos, para
que el modulo sea importable. Cualquier cifra que NB09 ya publico (recall/precision/
leakage/frac_own/switch_rate/track F1 de A_box, B_point, C_seq) sigue siendo valida sin
cambios: no se toco ese notebook.

**Dos piezas nuevas que NB09 no necesitaba** (documentadas donde aparecen, no aca):
Mask2Former no recibe cajas de un detector externo, asi que no hay un "prompt" del que
heredar gratis la identidad `own_id` de cada mascara predicha -- `emparejar_greedy` con
`iou_mascara` es el analogo directo de `emparejar_con_gt` (IoU de cajas) para resolver
esa asignacion prediccion->GT por IoU de mascara. Mismo razonamiento para el flag de
ambiguedad de la seccion 7 de NB09 (que usaba `masks_A_raw`, las mascaras SIN arbitrar
por `apply_non_overlapping_constraints`): el analogo para Mask2Former son las mascaras
de query umbraladas ANTES de `post_process_instance_segmentation` (que hace su propio
merge/arbitraje) -- `flaggear_cruces` es agnostico al origen de las mascaras, solo pide
que sean las "crudas".
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pandas as pd

__all__ = [
    "AREA_FRAC_DEGENERADA",
    "IOU_AMBIGUO",
    "MIN_OVERLAP_FILAS",
    "THR_PX_TRACK",
    "emparejar_con_gt",
    "emparejar_greedy",
    "evaluar_mascara",
    "evaluar_trayectorias",
    "evaluar_trayectorias_polilineas",
    "extraer_subpixel",
    "flaggear_cruces",
    "iou_caja",
    "iou_mascara",
    "nearest_particle_per_row",
    "resumen_identidad",
]

# Mismos umbrales que notebook 09 (SS6, SS6.1, SS7, SS6.3) -- no cambiar sin repetir la
# comparacion sobre A_box/B_point/C_seq, porque dejarian de ser el mismo criterio.
AREA_FRAC_DEGENERADA = 0.30  # mascara que cubre >30% del kymografo es sospechosa (NB07 SS6f)
IOU_AMBIGUO = 0.10  # par de mascaras CRUDAS que se solapan mas que esto -> candidato a cruce
THR_PX_TRACK = 4.0  # asociacion polilinea-GT, mismo thr_px_nativo que comparar_tracks() de NB06
MIN_OVERLAP_FILAS = 10  # voto por mayoria, mismo min_overlap que NB06


# --------------------------------------------------------------------------- #
# SS4/SS6 -- IoU y emparejamiento prediccion <-> GT (greedy, mayor IoU primero)
# --------------------------------------------------------------------------- #
def iou_caja(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU de dos cajas `(x1, y1, x2, y2)`. Verbatim de NB09 SS6."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_mascara(a: np.ndarray, b: np.ndarray) -> float:
    """IoU de dos mascaras binarias (T, L). Analogo de `iou_caja` para Mask2Former:
    sin cajas de un detector externo, el emparejamiento prediccion<->GT tiene que
    hacerse por solape de mascara en vez de solape de caja (ver docstring del modulo)."""
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def emparejar_greedy(preds: list, gts: list, iou_fn, iou_thr: float = 0.3) -> list[int | None]:
    """Empareja cada `preds[i]` con el `gts[j]` de mayor `iou_fn(pred, gt)` (>= `iou_thr`),
    greedy y sin reemplazo -- generalizacion de `emparejar_con_gt` de NB09 (que fijaba
    `iou_fn=iou_caja`) para poder reusar el mismo algoritmo con `iou_mascara`. Devuelve una
    lista de indices en `gts` (o `None`), paralela a `preds`."""
    usados: set[int] = set()
    match: list[int | None] = []
    for p in preds:
        mejor_j, mejor_iou = None, iou_thr
        for j, g in enumerate(gts):
            if j in usados:
                continue
            v = iou_fn(p, g)
            if v >= mejor_iou:
                mejor_j, mejor_iou = j, v
        if mejor_j is not None:
            usados.add(mejor_j)
        match.append(mejor_j)
    return match


def emparejar_con_gt(
    cajas_pred: list[tuple[float, float, float, float]],
    cajas_gt: list[tuple[float, float, float, float]],
    iou_thr: float = 0.3,
) -> list[int | None]:
    """Caso `iou_caja` de `emparejar_greedy` -- firma identica a NB09 SS6 (cajas GT como
    tuplas `(x1,y1,x2,y2)`, no como `CajaTraza`; el caller extrae los campos)."""
    return emparejar_greedy(cajas_pred, cajas_gt, iou_caja, iou_thr)


# --------------------------------------------------------------------------- #
# SS5 -- extraccion sub-pixel de la polilinea de una mascara
# --------------------------------------------------------------------------- #
def extraer_subpixel(mask: np.ndarray, kymo: np.ndarray, margen_px: int = 2) -> pd.DataFrame:
    """Verbatim de NB09 SS5 (Paso 4.5c): por cada fila (frame) con pixeles activos en la
    mascara, centroide pesado por la intensidad real del kymografo sobre esas columnas
    +/- un margen. Devuelve un DataFrame (frame, col_subpixel) -- subpixel, no columna
    entera, para no cuantizar la velocidad aguas abajo (requisito 1.5)."""
    T, L = kymo.shape[:2]
    kymo2d = kymo if kymo.ndim == 2 else kymo[..., 0]
    filas = []
    for f in range(T):
        cols_activas = np.where(mask[f])[0]
        if len(cols_activas) == 0:
            continue
        c_lo = max(int(cols_activas.min()) - margen_px, 0)
        c_hi = min(int(cols_activas.max()) + margen_px + 1, L)
        ventana = kymo2d[f, c_lo:c_hi].astype(np.float64)
        peso = np.clip(ventana - ventana.min(), 0, None)
        if peso.sum() <= 0:
            col_sub = float(cols_activas.mean())
        else:
            col_sub = float(c_lo + np.average(np.arange(len(ventana)), weights=peso))
        filas.append((f, col_sub))
    return pd.DataFrame(filas, columns=["frame", "col_subpixel"])


def _velocidad_px_frame(frames, cols) -> float:
    """Pendiente (px/frame) por ajuste lineal -- necesita >= 2 puntos. Verbatim de NB09 SS6.3."""
    if len(frames) < 2:
        return np.nan
    return float(np.polyfit(np.asarray(frames, dtype=float), np.asarray(cols, dtype=float), 1)[0])


# --------------------------------------------------------------------------- #
# SS6 -- metricas a nivel mascara (recall/precision/leakage) + diagnostico de identidad
# --------------------------------------------------------------------------- #
def nearest_particle_per_row(
    mask: np.ndarray,
    positions_mov: pd.DataFrame,
    n_frames: int,
    pixel_scale_um: float,
    L: int,
) -> dict[int, tuple[int, float]]:
    """Verbatim de NB09 SS6 (generalizado de NB07 SS6b a N particulas): para cada fila con
    pixeles activos, `particle_id` GT (solo moviles) mas cercano en columna, con el
    espejo L-R correcto (igual que `mascara_traza`/`cajas_desde_positions`)."""
    cols_gt = positions_mov.copy()
    cols_gt["col"] = (L - 1) - cols_gt["pos_um"].to_numpy() / pixel_scale_um
    trace = {}
    for f in range(n_frames):
        row_px = np.where(mask[f])[0]
        if len(row_px) == 0:
            continue
        mcol = row_px.mean()
        gf = cols_gt[cols_gt["frame"] == f]
        if len(gf) == 0:
            continue
        d_ = (gf["col"] - mcol).abs()
        trace[f] = (int(gf.loc[d_.idxmin(), "particle_id"]), float(d_.min()))
    return trace


def resumen_identidad(trace: dict[int, tuple[int, float]], own_id: int) -> dict:
    """Verbatim de NB09 SS6: `cambios_identidad` (conteo crudo de cambios fila a fila) y
    `frac_own` (fraccion de filas cuyo GT mas cercano es la propia particula)."""
    ids = [pid for pid, _ in trace.values()]
    if not ids:
        return {"cambios_identidad": 0, "frac_own": 0.0, "n_filas": 0}
    switches = sum(1 for a, b in itertools.pairwise(ids) if a != b)
    frac_own = sum(1 for i in ids if i == own_id) / len(ids)
    return {"cambios_identidad": switches, "frac_own": round(frac_own, 3), "n_filas": len(ids)}


def evaluar_mascara(
    pred_mask: np.ndarray,
    gt_propia: np.ndarray,
    gt_otras: np.ndarray,
    *,
    area_frac_degenerada: float = AREA_FRAC_DEGENERADA,
) -> dict:
    """Verbatim de NB09 SS6: recall/precision contra la mascara GT propia, `leakage`
    contra `gt_otras` (cualquier OTRA particula real -- ver caveat de NB09 SS9 sobre que
    solo cuenta moviles), y la guarda de area degenerada."""
    pred = pred_mask.astype(bool)
    recall = (pred & gt_propia).sum() / max(gt_propia.sum(), 1)
    precision = (pred & gt_propia).sum() / max(pred.sum(), 1)
    leakage = (pred & gt_otras).sum() / max(pred.sum(), 1)
    area_frac = pred.sum() / pred.size
    return {
        "recall": round(float(recall), 3),
        "precision": round(float(precision), 3),
        "leakage": round(float(leakage), 3),
        "area_frac": round(float(area_frac), 4),
        "degenerada": area_frac > area_frac_degenerada,
    }


# --------------------------------------------------------------------------- #
# SS7 -- flag de cruces/ambiguedad por IoU de a pares entre mascaras CRUDAS (sin arbitrar)
# --------------------------------------------------------------------------- #
def flaggear_cruces(
    masks: list[np.ndarray],
    nombre_muestra: str,
    particle_ids: list[int],
    *,
    iou_thr: float = IOU_AMBIGUO,
) -> list[dict]:
    """Verbatim de NB09 SS7 (Paso 4.7, corregido): IoU de a pares entre mascaras CRUDAS
    (sin arbitrar/sin merge posterior) de la MISMA imagen. No resuelve nada, solo marca
    (requisito 1.12). Incluye el INDICE de cada mascara ademas del `particle_id`: dos
    predicciones sin match a GT comparten `particle_id=-1`, ese campo solo no alcanza
    para distinguirlas (bug real que NB09 encontro y corrigio, ver su SS7)."""
    filas = []
    for i, j in itertools.combinations(range(len(masks)), 2):
        iou = iou_mascara(masks[i], masks[j])
        if iou >= iou_thr:
            filas.append({
                "muestra": nombre_muestra,
                "caja_i": i,
                "caja_j": j,
                "particle_id_i": particle_ids[i],
                "particle_id_j": particle_ids[j],
                "ambos_matcheados": particle_ids[i] != -1 and particle_ids[j] != -1,
                "iou_mascaras_crudas": round(float(iou), 3),
            })
    return filas


# --------------------------------------------------------------------------- #
# SS6.3 -- evaluacion a nivel trayectoria (misma unidad que la tabla final de la tesis)
# --------------------------------------------------------------------------- #
def evaluar_trayectorias(
    escenas: dict[str, dict],
    masks_key: str,
    *,
    min_desplazamiento_px: float,
    thr_px_track: float = THR_PX_TRACK,
    min_overlap_filas: int = MIN_OVERLAP_FILAS,
) -> tuple[pd.DataFrame, dict]:
    """Version para modelos que predicen MASCARAS: extrae la polilinea subpixel de cada
    mascara (`extraer_subpixel`) y delega en `evaluar_trayectorias_polilineas`, que tiene
    toda la logica de asociacion. Firma y resultados identicos a antes del refactor.
    """
    escenas_poli = {}
    for nombre, d in escenas.items():
        polis = [extraer_subpixel(m, d["kymo"]) for m in d[masks_key]]
        escenas_poli[nombre] = {**d, "polilineas": [p for p in polis if len(p)]}
    return evaluar_trayectorias_polilineas(
        escenas_poli,
        min_desplazamiento_px=min_desplazamiento_px,
        thr_px_track=thr_px_track,
        min_overlap_filas=min_overlap_filas,
    )


def evaluar_trayectorias_polilineas(
    escenas: dict[str, dict],
    *,
    min_desplazamiento_px: float,
    thr_px_track: float = THR_PX_TRACK,
    min_overlap_filas: int = MIN_OVERLAP_FILAS,
    polilineas_key: str = "polilineas",
) -> tuple[pd.DataFrame, dict]:
    """Metricas de trayectoria, mismo criterio de asociacion que `comparar_tracks()` de
    NB06 (GT movil mas cercano por fila dentro de `thr_px_track`, voto por mayoria con
    `min_overlap_filas`). Logica identica a `evaluar_trayectorias()` de NB09 SS6.3; la
    unica diferencia es que `escenas` y `min_desplazamiento_px` son parametros explicitos
    en vez de closures sobre variables globales del notebook (ver docstring del modulo).

    `escenas`: `{nombre: {"kymo", "positions", "pixel_scale_um", "fps", masks_key: [...]}}`
    -- un dict por muestra con el kymografo (T,L), el `positions.csv` completo, el
    `pixel_scale_um`/`fps` de su `config.yaml`, y la lista de mascaras predichas bajo la
    clave `masks_key` (una mascara binaria (T,L) por instancia predicha).
    """
    from axonal_tracking import etiquetas_deteccion as ed

    filas_tr = []
    n_gt_total = 0
    recuperadas_total = 0
    for nombre, d in escenas.items():
        kymo = d["kymo"]
        L = kymo.shape[1]
        px = d["pixel_scale_um"]
        fps = d["fps"]
        ids_mov = ed.ids_que_se_mueven(d["positions"], px, min_desplazamiento_px)
        movs = d["positions"][d["positions"]["particle_id"].isin(ids_mov)].copy()
        movs["col"] = (L - 1) - movs["pos_um"].to_numpy() / px

        gt_por_frame: dict[int, list[tuple[int, float]]] = {}
        gt_col: dict[tuple[int, int], float] = {}
        for row in movs.itertuples():
            f, pid, col = int(row.frame), int(row.particle_id), float(row.col)
            gt_por_frame.setdefault(f, []).append((pid, col))
            gt_col[(pid, f)] = col
        v_gt_por_pid = {
            int(pid): _velocidad_px_frame(sub["frame"].to_numpy(), sub["col"].to_numpy())
            for pid, sub in movs.groupby("particle_id")
        }
        n_gt_total += len(ids_mov)
        dominantes_escena = set()

        for k, poli in enumerate(d[polilineas_key]):
            if len(poli) == 0:
                continue
            etiquetas = []
            for f, x in zip(poli["frame"], poli["col_subpixel"]):
                mejor, dmin = -1, thr_px_track
                for pid, gx in gt_por_frame.get(int(f), []):
                    dd = abs(x - gx)
                    if dd <= dmin:
                        mejor, dmin = pid, dd
                etiquetas.append(mejor)
            conteo = Counter(l for l in etiquetas if l != -1)
            sustanciales = {pid: c for pid, c in conteo.items() if c >= min_overlap_filas}
            dominante = (
                max(sustanciales, key=sustanciales.get)
                if sustanciales
                else (conteo.most_common(1)[0][0] if conteo else -1)
            )
            errs = [
                abs(x - gt_col[(dominante, int(f))])
                for f, x, l in zip(poli["frame"], poli["col_subpixel"], etiquetas)
                if l == dominante and dominante != -1 and (dominante, int(f)) in gt_col
            ]
            v_pred = _velocidad_px_frame(poli["frame"].to_numpy(), poli["col_subpixel"].to_numpy())
            v_gt = v_gt_por_pid.get(dominante, np.nan)
            err_v_um_s = (
                abs(v_pred - v_gt) * px * fps
                if dominante != -1 and np.isfinite(v_pred) and np.isfinite(v_gt)
                else np.nan
            )
            if dominante != -1:
                dominantes_escena.add(dominante)
            filas_tr.append({
                "muestra": nombre,
                "caja": k,
                "gt_id": dominante,
                "n_filas": len(poli),
                "id_switch": len(sustanciales) > 1,
                "err_px": float(np.mean(errs)) if errs else np.nan,
                "err_um": float(np.mean(errs)) * px if errs else np.nan,
                "err_vel_um_s": err_v_um_s,
            })
        recuperadas_total += len(dominantes_escena)

    tr = pd.DataFrame(filas_tr)
    matched = tr[tr.gt_id != -1] if len(tr) else tr
    precision = len(matched) / len(tr) if len(tr) else np.nan
    recall = recuperadas_total / n_gt_total if n_gt_total else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    resumen = {
        "n_polilineas": len(tr),
        "n_gt_moviles": n_gt_total,
        "track_recall": round(recall, 3) if np.isfinite(recall) else recall,
        "track_precision": round(precision, 3) if np.isfinite(precision) else precision,
        "track_f1": round(f1, 3) if np.isfinite(f1) else f1,
        "err_pos_px_medio": round(float(matched.err_px.mean()), 2) if len(matched) else np.nan,
        "err_pos_um_medio": round(float(matched.err_um.mean()), 3) if len(matched) else np.nan,
        "err_vel_um_s_medio": round(float(matched.err_vel_um_s.mean()), 3) if len(matched) else np.nan,
        "err_vel_um_s_mediana": round(float(matched.err_vel_um_s.median()), 3) if len(matched) else np.nan,
        "frac_id_switch": round(float(tr.id_switch.mean()), 3) if len(tr) else np.nan,
    }
    return tr, resumen
