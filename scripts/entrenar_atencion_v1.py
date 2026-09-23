"""Stage 2 v1 (`plan/association-transformer-guide.md` SS5.1-5.5): transformer
encoder sobre tokens de segmento, geometria+fotometria solamente (sin tiles de
DecNet -- eso es v2, SS5.6, no este script). Entrenado desde cero (sin pesos
preentrenados), decodificado por el MISMO nucleo (`asociacion.decodificar_*`) que
Stage 1 usa, para que la comparacion aisle la funcion de scoring.

Reusa `entrenar_costo_clasico.preparar_muestra` (segmentos + pares candidatos sobre
la trackness real de KymoButler, identico pipeline que Gate A/Stage 1) y le agrega
los tensores de Stage 2 (tokens SS5.2, geom_feats normalizados SS5.3).

Pasos:
1. Preparar train (800) + val (400) -- tokens, pares, geom_feats, etiquetas de
   enlace (SS5.4). `pos_weight` = razon negativos/positivos en train.
2. Smoke test: un modelo nuevo, unos pocos pasos sobre una submuestra de train --
   confirmar que la perdida baja ANTES de correr el entrenamiento completo.
3. Entrenamiento: acumulacion de gradiente (~8 muestras/paso, Adam 1e-3), early
   stopping en F1 de enlace por-par sobre val a umbral de probabilidad 0.5 (metrica
   de paro, reportada aparte del umbral de decodificacion elegido en el paso 4).
4. Barrido de umbral x decodificador (Hungarian vs greedy) sobre val -- MISMA regla
   de eleccion que la fila 2, la de Gate B (SS8: `fragmentos_por_gt <= 1.282` y
   `track_f1 >= 0.968`, los de DecNet, menor `frac_id_switch` entre las que
   cumplen ambas), para que la comparacion entre filas no cambie de criterio de
   eleccion (si cambia, deja de aislar la funcion de scoring).
5. Fila 3: decodificar + evaluar sobre test (400), una sola vez.

Requiere `results/asociacion/cache/{train,val,test}/` tibios.

Salida: `results/asociacion/atencion_v1_pesos.pt`,
`results/asociacion/atencion_v1_resumen.json`, `results/asociacion/atencion_v1_fila3.csv`.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # antes de `import torch` (SS12.8)

import numpy as np
import torch
from torch import nn

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from entrenar_costo_clasico import (
    MAX_GAP_FRAMES,
    MAX_SALTO_PX,
    MIN_DESP,
    elegir_punto_operacion,
    preparar_muestra,
)
from evaluar_kymobutler_400 import es_movil

from axonal_tracking import asociacion as aso
from axonal_tracking import atencion as at
from axonal_tracking import evaluacion as ev

SALIDA = RAIZ / "results" / "asociacion"

SEMILLA = 0
MAX_EPOCHS = 60
PACIENCIA = 15
TAMANO_ACUMULACION = 8
LR = 1e-3
N_MUESTRAS_SMOKE = 40
UMBRALES_BARRIDO = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]  # unidades de -log(prob)


def preparar_muestra_stage2(nombre: str, split: str, *, con_gt: bool, incluir_crudos: bool = False) -> dict:
    """`incluir_crudos=True` (SS5.6, v2 unicamente): agrega `preprocessed`/`skel` al
    dict para extraer tiles de DecNet -- default False porque cargar esos arrays
    para las 1200 muestras de train+val a la vez (varios GB) es peso que v1 no
    necesita y no deberia pagar.

    `mascara_etiquetada` (SS5.4, `plan/notebook-11-review.md` SS2.4 item 4): True
    solo para los pares donde AMBOS segmentos tienen `particle_id != -1`. Un par
    que toca un segmento sin asignar no es "no-enlace" -- es SIN ETIQUETA (el plan
    dice excluirlo de la perdida explicitamente), y la version anterior de este
    script lo trataba como negativo duro por omision, inyectando ruido de etiqueta
    concentrado justo en los cruces (donde `asignar_gt` mas frecuentemente deja
    `particle_id=-1`). `pares`/`geom_pares` en si NO se filtran -- siguen
    representando TODOS los candidatos, porque en inferencia real no hay GT para
    saber cuales excluir; la mascara solo decide que pares entran a la perdida/
    metricas de entrenamiento."""
    segmentos, caract, pares, verdaderos, asignaciones, e, preprocessed, skel = preparar_muestra(nombre, split)
    T, L = e["kymo"].shape
    features = at.construir_features_tokens(segmentos, caract, T, L, e["kymo"])
    t_ini_norm = np.array([s.t_ini / T for s in segmentos], dtype=np.float32)
    x_ini_norm = np.array([s.x_ini / L for s in segmentos], dtype=np.float32)
    escala_v = (L / T) if T > 0 else 1.0
    geom_pares = at.construir_geom_feats_pares(
        segmentos, caract, pares, max_salto_px=MAX_SALTO_PX, max_gap_frames=MAX_GAP_FRAMES,
        escala_v=escala_v,
    )
    etiquetas = None
    mascara_etiquetada = None
    if con_gt:
        ids_asignados = [a.particle_id for a in asignaciones]
        etiquetas = np.array([(i, j) in verdaderos for i, j in pares], dtype=np.float32)
        mascara_etiquetada = np.array(
            [ids_asignados[i] != -1 and ids_asignados[j] != -1 for i, j in pares], dtype=bool
        )
    salida = {
        "nombre": nombre, "segmentos": segmentos, "caracteristicas": caract, "pares": pares,
        "escena": e, "features": features, "t_ini_norm": t_ini_norm, "x_ini_norm": x_ini_norm,
        "geom_pares": geom_pares, "etiquetas": etiquetas, "mascara_etiquetada": mascara_etiquetada,
    }
    if incluir_crudos:
        salida["preprocessed"] = preprocessed
        salida["skel"] = skel
    return salida


def cargar_split(
    nombres: list[str], split: str, *, con_gt: bool, incluir_crudos: bool = False
) -> tuple[list[dict], list[dict]]:
    datos, fallos = [], []
    for nombre in nombres:
        try:
            datos.append(preparar_muestra_stage2(nombre, split, con_gt=con_gt, incluir_crudos=incluir_crudos))
        except Exception as exc:  # noqa: BLE001 -- una muestra rota no debe tumbar el split
            fallos.append({"muestra": nombre, "error": repr(exc)})
    return datos, fallos


def _tensorizar(datos: list[dict]) -> None:
    """Convierte los arrays numpy a tensores UNA vez (no en cada epoca/evaluacion) --
    muta los dicts en el lugar."""
    for d in datos:
        d["features_t"] = torch.from_numpy(d["features"])
        d["t_ini_t"] = torch.from_numpy(d["t_ini_norm"])
        d["x_ini_t"] = torch.from_numpy(d["x_ini_norm"])
        d["geom_t"] = torch.from_numpy(d["geom_pares"])
        d["pares_t"] = (
            torch.tensor(d["pares"], dtype=torch.long) if d["pares"] else torch.zeros((0, 2), dtype=torch.long)
        )
        if d["etiquetas"] is not None:
            d["etiquetas_t"] = torch.from_numpy(d["etiquetas"])
        if d["mascara_etiquetada"] is not None:
            d["mascara_etiquetada_t"] = torch.from_numpy(d["mascara_etiquetada"])


def _f1_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def _forward(modelo: at.ModeloAtencionV1, d: dict) -> torch.Tensor:
    return modelo(d["features_t"], d["t_ini_t"], d["x_ini_t"], d["pares_t"], d["geom_t"])


def epoca_entrenamiento(modelo, datos_train, optimizador, criterio, *, tam_acumulacion) -> float:
    """SS5.4: la perdida se calcula SOLO sobre `mascara_etiquetada` -- los pares que
    tocan un segmento sin asignar (`particle_id == -1`) no tienen etiqueta valida y
    se excluyen, no se tratan como negativos (`plan/notebook-11-review.md` SS2.4
    item 4)."""
    modelo.train()
    random.shuffle(datos_train)
    perdida_acum, n_pares_total = 0.0, 0
    optimizador.zero_grad()
    for i, d in enumerate(datos_train):
        mascara = d["mascara_etiquetada_t"]
        if len(d["pares"]) == 0 or mascara.sum() == 0:
            continue
        logits = _forward(modelo, d)
        perdida = criterio(logits[mascara], d["etiquetas_t"][mascara]) / tam_acumulacion
        perdida.backward()
        n_validos = int(mascara.sum())
        perdida_acum += float(perdida.item()) * tam_acumulacion * n_validos
        n_pares_total += n_validos
        if (i + 1) % tam_acumulacion == 0:
            optimizador.step()
            optimizador.zero_grad()
    optimizador.step()
    optimizador.zero_grad()
    return perdida_acum / max(n_pares_total, 1)


@torch.no_grad()
def evaluar_val_por_par(modelo, datos_val) -> tuple[float, float, float]:
    """Misma exclusion que `epoca_entrenamiento` (SS5.4): la F1 de enlace por-par
    de val tampoco cuenta pares sin etiqueta valida."""
    modelo.eval()
    etiquetas_todas, pred_todas = [], []
    for d in datos_val:
        mascara = d["mascara_etiquetada"]
        if len(d["pares"]) == 0 or mascara.sum() == 0:
            continue
        logits = _forward(modelo, d)
        etiquetas_todas.append(d["etiquetas"][mascara].astype(bool))
        pred_todas.append(logits.numpy()[mascara] > 0)
    y = np.concatenate(etiquetas_todas)
    pred = np.concatenate(pred_todas)
    return _f1_precision_recall(y, pred)


@torch.no_grad()
def decodificar_evaluar(modelo, datos, nombre_decodificador: str, umbral: float):
    modelo.eval()
    escenas = {}
    for d in datos:
        if len(d["pares"]) == 0:
            polis = []
        else:
            logits = _forward(modelo, d)
            costos = at.costos_modelo(logits, d["pares"])
            if nombre_decodificador == "hungarian":
                cadenas = aso.decodificar_bipartito(len(d["segmentos"]), d["pares"], costos, umbral)
            else:
                cadenas = aso.decodificar_greedy(len(d["segmentos"]), costos, umbral)
            polis = aso.polilineas_desde_cadenas(cadenas, d["segmentos"], d["escena"]["kymo"])
            polis = [p for p in polis if es_movil(p)]
        escenas[d["nombre"]] = {**d["escena"], "polilineas": polis}
    tr, res = ev.evaluar_trayectorias_polilineas(escenas, min_desplazamiento_px=MIN_DESP)
    mm = tr[tr.gt_id != -1]
    frag = mm.groupby(["muestra", "gt_id"]).size()
    res = {**res, "fragmentos_por_gt": round(float(frag.mean()), 3) if len(frag) else None}
    return tr, res


def main() -> None:
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("=== Preparando train (800) + val (400): tokens + pares + etiquetas ===", flush=True)
    t0 = time.time()
    nombres_train = [d.name for d in sorted((RAIZ / "datasets" / "train").glob("sample_*"))]
    nombres_val = [d.name for d in sorted((RAIZ / "datasets" / "val").glob("sample_*"))]
    datos_train, fallos_train = cargar_split(nombres_train, "train", con_gt=True)
    datos_val, fallos_val = cargar_split(nombres_val, "val", con_gt=True)
    _tensorizar(datos_train)
    _tensorizar(datos_val)
    print(f"  {time.time() - t0:.0f}s -- train: {len(datos_train)} ok, {len(fallos_train)} fallos  "
          f"val: {len(datos_val)} ok, {len(fallos_val)} fallos", flush=True)

    # SS5.4: pos_weight se calcula SOLO sobre pares con etiqueta valida (mascara_etiquetada) --
    # ver docstring de preparar_muestra_stage2 para por que excluir, no tratar como negativo.
    n_pares_total_train = sum(len(d["pares"]) for d in datos_train)
    etiquetas_train_todas = np.concatenate(
        [d["etiquetas"][d["mascara_etiquetada"]] for d in datos_train if len(d["pares"])]
    )
    n_pos, n_neg = float(etiquetas_train_todas.sum()), float(len(etiquetas_train_todas) - etiquetas_train_todas.sum())
    pos_weight = n_neg / max(n_pos, 1.0)
    n_excluidos = n_pares_total_train - len(etiquetas_train_todas)
    print(f"  pares candidatos train: {n_pares_total_train}  con etiqueta valida: {len(etiquetas_train_todas)}  "
          f"excluidos (tocan segmento sin asignar): {n_excluidos} ({n_excluidos / max(n_pares_total_train,1):.3f})  "
          f"positivos={int(n_pos)}  negativos={int(n_neg)}  pos_weight={pos_weight:.3f}", flush=True)

    print(f"\n=== Smoke test: {N_MUESTRAS_SMOKE} muestras, la perdida debe bajar ===", flush=True)
    modelo_smoke = at.ModeloAtencionV1()
    opt_smoke = torch.optim.Adam(modelo_smoke.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    submuestra = datos_train[:N_MUESTRAS_SMOKE]
    perdida_1 = epoca_entrenamiento(modelo_smoke, submuestra, opt_smoke, crit, tam_acumulacion=4)
    perdida_n = perdida_1
    for _ in range(9):
        perdida_n = epoca_entrenamiento(modelo_smoke, submuestra, opt_smoke, crit, tam_acumulacion=4)
    print(f"  perdida epoca 1: {perdida_1:.4f}  perdida epoca 10 (mismas {N_MUESTRAS_SMOKE} muestras): "
          f"{perdida_n:.4f}", flush=True)
    assert perdida_n < perdida_1, "smoke test: la perdida NO bajo -- no seguir sin resolver esto (SS10 checklist)"
    print("  smoke test OK", flush=True)

    print("\n=== Entrenamiento (early stopping en val, F1 de enlace por-par @ prob=0.5) ===", flush=True)
    modelo = at.ModeloAtencionV1()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    # Coseno hacia un piso bajo (no 0): la primera corrida oscilaba 0.44-0.50 en val
    # sin asentarse en el plateau (epocas 17-55) -- sintoma de LR alta para esa
    # etapa, no de no-convergencia. Un solo reintento con decaimiento (advisor:
    # "cap it at that one run").
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=MAX_EPOCHS, eta_min=LR * 0.05)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

    mejor_f1_val, mejor_estado, mejor_epoca, sin_mejora = -1.0, None, -1, 0
    historial = []
    t0 = time.time()
    for epoca in range(MAX_EPOCHS):
        perdida_train = epoca_entrenamiento(modelo, datos_train, optimizador, criterio,
                                             tam_acumulacion=TAMANO_ACUMULACION)
        programador.step()
        f1_val, precision_val, recall_val = evaluar_val_por_par(modelo, datos_val)
        historial.append({"epoca": epoca, "perdida_train": perdida_train, "f1_val_par": f1_val,
                           "precision_val_par": precision_val, "recall_val_par": recall_val,
                           "lr": optimizador.param_groups[0]["lr"]})
        print(f"  epoca {epoca:3d}  ({(time.time()-t0)/60:.1f} min)  lr={optimizador.param_groups[0]['lr']:.2e}  "
              f"perdida_train={perdida_train:.4f}  f1_val_par={f1_val:.4f}  precision={precision_val:.4f}  "
              f"recall={recall_val:.4f}", flush=True)
        if f1_val > mejor_f1_val:
            mejor_f1_val, mejor_epoca, sin_mejora = f1_val, epoca, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                print(f"  early stopping en epoca {epoca} (sin mejora en {PACIENCIA} epocas)", flush=True)
                break

    modelo.load_state_dict(mejor_estado)
    print(f"\nmejor epoca: {mejor_epoca}  f1_val_por_par (@0.5)={mejor_f1_val:.4f}", flush=True)
    torch.save(mejor_estado, SALIDA / "atencion_v1_pesos.pt")

    print("\n=== Barrido umbral x decodificador sobre val (400 muestras) ===", flush=True)
    filas_barrido = []
    for nombre_decod in ("hungarian", "greedy"):
        for umbral in UMBRALES_BARRIDO:
            _, res = decodificar_evaluar(modelo, datos_val, nombre_decod, umbral)
            filas_barrido.append({"decodificador": nombre_decod, "umbral": umbral, **res})
            print(f"  [{nombre_decod}] umbral={umbral:>4}  track_f1={res['track_f1']}  "
                  f"frac_id_switch={res['frac_id_switch']}  fragmentos_por_gt={res['fragmentos_por_gt']}  "
                  f"precision={res['track_precision']}  recall={res['track_recall']}", flush=True)

    # Regla de seleccion de SS8, identica para las filas 2/3/3b -- ver elegir_punto_operacion.
    # Elegido en VAL, nunca en test. cumple_barras_seleccion_val != veredicto de Gate B (ese
    # se calcula en el notebook SS7, con el frac_id_switch de TEST contra el 0.103 de DecNet).
    elegido = elegir_punto_operacion(filas_barrido)
    print(f"\n  elegido: decodificador={elegido['decodificador']}  umbral={elegido['umbral']}  "
          f"cumple_barras_seleccion_val={elegido['cumple_barras_seleccion_val']}  (val: "
          f"track_f1={elegido['track_f1']}  frac_id_switch={elegido['frac_id_switch']}  "
          f"fragmentos_por_gt={elegido['fragmentos_por_gt']})",
          flush=True)

    print("\n=== Test: decodificando con el punto elegido (400 muestras) ===", flush=True)
    nombres_test = [d.name for d in sorted((RAIZ / "datasets" / "test").glob("sample_*"))]
    datos_test, fallos_test = cargar_split(nombres_test, "test", con_gt=False)
    _tensorizar(datos_test)
    tr, res = decodificar_evaluar(modelo, datos_test, elegido["decodificador"], elegido["umbral"])
    tr.to_csv(SALIDA / "atencion_v1_fila3.csv", index=False)

    print("\n=== Fila 3 (KymoButler trackness + atencion v1), test ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    resumen = {
        "pos_weight": pos_weight, "n_pos_train": int(n_pos), "n_neg_train": int(n_neg),
        "n_pares_total_train": n_pares_total_train, "n_excluidos_sin_asignar_train": n_excluidos,
        "smoke_test": {"perdida_epoca_1": perdida_1, "perdida_epoca_10": perdida_n},
        "mejor_epoca": mejor_epoca, "f1_val_por_par": mejor_f1_val,
        "historial_entrenamiento": historial,
        "barrido_umbral_val": filas_barrido,
        "punto_elegido": elegido,
        "fila_3_test": res,
        "fallos_train": fallos_train, "fallos_val": fallos_val, "fallos_test": fallos_test,
        "hiperparametros": {
            "d": modelo.d, "max_epochs": MAX_EPOCHS, "paciencia": PACIENCIA,
            "tam_acumulacion": TAMANO_ACUMULACION, "lr": LR, "semilla": SEMILLA,
        },
    }
    (SALIDA / "atencion_v1_resumen.json").write_text(json.dumps(resumen, indent=2, default=str))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
