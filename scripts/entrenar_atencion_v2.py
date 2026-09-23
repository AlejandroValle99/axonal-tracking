"""Stage 2b v2 (`plan/association-transformer-guide.md` SS5.6): v1 + tiles de
DecNet -- el mismo recorte 48x48 de tres canales (kymografo, mascara de track,
mascara de estructura) que DecNet consume en cada paso, agregado a cada extremo de
segmento y fusionado por suma en el token. Mismo nucleo de decodificacion que
v1/Stage 1 (SS5.5): la comparacion fila 3 vs 3b aisla si la informacion visual de
DecNet, vista globalmente en vez de local-greedy, agrega algo sobre v1
(solo-geometria) -- y fila 3b vs fila 2 si agrega algo sobre el costo clasico.

**Precondicion**: el chequeo de alineacion de tiles (SS10 checklist / SS5.6) ya
corrio y paso, numerica (desvio 0px respecto del centro esperado en 366 extremos
de prueba) y visualmente (el punto de conexion cae sobre el pixel brillante del
segmento en cada tile revisado) -- no repetido aca, ver notebook SS5.

Requiere `results/asociacion/cache/{train,val,test}/` tibios y
`results/asociacion/costo_clasico_resumen.json` (fila 2, referencia de respaldo
para `elegir_punto_operacion`).

Salida: `results/asociacion/atencion_v2_pesos.pt`,
`results/asociacion/atencion_v2_resumen.json`, `results/asociacion/atencion_v2_fila3b.csv`.
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

from entrenar_atencion_v1 import _f1_precision_recall, cargar_split
from entrenar_costo_clasico import MIN_DESP, elegir_punto_operacion
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
N_MUESTRAS_TIMING = 60  # submuestra para medir CPU vs MPS antes de comprometerse (advisor)
UMBRALES_BARRIDO = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]


def agregar_tiles(d: dict) -> dict:
    """Agrega `tiles_fin`/`tiles_ini` `(N, 3, 48, 48)` numpy a un dict de
    `preparar_muestra_stage2(..., incluir_crudos=True)` (SS5.6). Se dejan como
    numpy, NO tensorizados -- 2*N*3*48*48 floats por muestra (~11MB para N=200)
    hace que tensorizar las 1200 muestras de una vez (como v1 hace con sus
    features chicas) no entre en memoria; se mueven a torch/device recien dentro
    del loop, una muestra a la vez (`_forward`). `preprocessed`/`skel` se
    descartan despues de usarse -- ya no hacen falta y son lo que mas memoria pesa."""
    padkym, allyx_padded, pad_size, kdtree = at.preparar_contexto_tiles(d["preprocessed"], d["skel"])
    tiles_fin, tiles_ini, n_ceros = [], [], 0
    for seg in d["segmentos"]:
        t_fin, _ = at.extraer_tile_segmento(seg, "fin", padkym, allyx_padded, pad_size, kdtree)
        t_ini, _ = at.extraer_tile_segmento(seg, "ini", padkym, allyx_padded, pad_size, kdtree)
        tiles_fin.append(t_fin)
        tiles_ini.append(t_ini)
        if not t_fin.any():
            n_ceros += 1
    dim = at.VISION_MODULE_TILE_SIZE
    d["tiles_fin"] = np.stack(tiles_fin) if tiles_fin else np.zeros((0, 3, dim, dim), dtype=np.float32)
    d["tiles_ini"] = np.stack(tiles_ini) if tiles_ini else np.zeros((0, 3, dim, dim), dtype=np.float32)
    d["n_tiles_ceros"] = n_ceros
    del d["preprocessed"], d["skel"]
    return d


def preparar_split_v2(nombres: list[str], split: str, *, con_gt: bool):
    datos, fallos = cargar_split(nombres, split, con_gt=con_gt, incluir_crudos=True)
    n_ceros_total = 0
    for d in datos:
        agregar_tiles(d)
        n_ceros_total += d["n_tiles_ceros"]
    return datos, fallos, n_ceros_total


def _tensorizar_v2(datos: list[dict], device: str) -> None:
    """Tensoriza y mueve a `device` las piezas CHICAS (features/pares/geom/etiquetas)
    -- viven en `device` todo el entrenamiento. Los tiles quedan como numpy; se
    tensorizan y mueven recien en `_forward`, una muestra a la vez (ver
    `agregar_tiles`)."""
    for d in datos:
        d["features_t"] = torch.from_numpy(d["features"]).to(device)
        d["t_ini_t"] = torch.from_numpy(d["t_ini_norm"]).to(device)
        d["x_ini_t"] = torch.from_numpy(d["x_ini_norm"]).to(device)
        d["geom_t"] = torch.from_numpy(d["geom_pares"]).to(device)
        pares_t = torch.tensor(d["pares"], dtype=torch.long) if d["pares"] else torch.zeros((0, 2), dtype=torch.long)
        d["pares_t"] = pares_t.to(device)
        if d["etiquetas"] is not None:
            d["etiquetas_t"] = torch.from_numpy(d["etiquetas"]).to(device)
        if d["mascara_etiquetada"] is not None:
            d["mascara_etiquetada_t"] = torch.from_numpy(d["mascara_etiquetada"]).to(device)


def _forward(modelo: at.ModeloAtencionV2, d: dict, device: str) -> torch.Tensor:
    tiles_fin = torch.from_numpy(d["tiles_fin"]).to(device)
    tiles_ini = torch.from_numpy(d["tiles_ini"]).to(device)
    return modelo(d["features_t"], d["t_ini_t"], d["x_ini_t"], tiles_fin, tiles_ini,
                   d["pares_t"], d["geom_t"])


def epoca_entrenamiento(modelo, datos_train, optimizador, criterio, device, *, tam_acumulacion) -> float:
    """SS5.4: la perdida excluye los pares que tocan un segmento sin asignar (ver
    docstring de `entrenar_atencion_v1.preparar_muestra_stage2`)."""
    modelo.train()
    random.shuffle(datos_train)
    perdida_acum, n_pares_total = 0.0, 0
    optimizador.zero_grad()
    for i, d in enumerate(datos_train):
        mascara = d["mascara_etiquetada_t"]
        if len(d["pares"]) == 0 or mascara.sum() == 0:
            continue
        logits = _forward(modelo, d, device)
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
def evaluar_val_por_par(modelo, datos_val, device) -> tuple[float, float, float]:
    modelo.eval()
    etiquetas_todas, pred_todas = [], []
    for d in datos_val:
        mascara = d["mascara_etiquetada"]
        if len(d["pares"]) == 0 or mascara.sum() == 0:
            continue
        logits = _forward(modelo, d, device)
        etiquetas_todas.append(d["etiquetas"][mascara].astype(bool))
        pred_todas.append(logits.cpu().numpy()[mascara] > 0)
    y = np.concatenate(etiquetas_todas)
    pred = np.concatenate(pred_todas)
    return _f1_precision_recall(y, pred)


@torch.no_grad()
def decodificar_evaluar(modelo, datos, nombre_decodificador: str, umbral: float, device: str):
    modelo.eval()
    escenas = {}
    for d in datos:
        if len(d["pares"]) == 0:
            polis = []
        else:
            logits = _forward(modelo, d, device)
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


def medir_tiempo_epoca(datos_muestra: list[dict], device: str) -> float:
    """Una epoca de entrenamiento sobre una submuestra chica, con un modelo/optim
    descartables -- solo para comparar velocidad CPU vs MPS antes de comprometerse
    a la corrida completa (advisor: 'verify MPS is actually faster before
    committing; time one epoch each way')."""
    _tensorizar_v2(datos_muestra, device)
    modelo = at.ModeloAtencionV2().to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    criterio = nn.BCEWithLogitsLoss()
    t0 = time.time()
    epoca_entrenamiento(modelo, datos_muestra, optimizador, criterio, device, tam_acumulacion=4)
    return time.time() - t0


def main() -> None:
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("=== Preparando train (800) + val (400): tokens + tiles + etiquetas ===", flush=True)
    t0 = time.time()
    nombres_train = [d.name for d in sorted((RAIZ / "datasets" / "train").glob("sample_*"))]
    nombres_val = [d.name for d in sorted((RAIZ / "datasets" / "val").glob("sample_*"))]
    datos_train, fallos_train, n_ceros_train = preparar_split_v2(nombres_train, "train", con_gt=True)
    datos_val, fallos_val, n_ceros_val = preparar_split_v2(nombres_val, "val", con_gt=True)
    n_tiles_train = sum(2 * len(d["segmentos"]) for d in datos_train)
    n_tiles_val = sum(2 * len(d["segmentos"]) for d in datos_val)
    print(f"  {time.time() - t0:.0f}s -- train: {len(datos_train)} ok, {len(fallos_train)} fallos, "
          f"tiles_en_blanco={n_ceros_train}/{n_tiles_train} ({n_ceros_train / max(n_tiles_train,1):.3f})  "
          f"val: {len(datos_val)} ok, {len(fallos_val)} fallos, "
          f"tiles_en_blanco={n_ceros_val}/{n_tiles_val} ({n_ceros_val / max(n_tiles_val,1):.3f})", flush=True)

    print(f"\n=== Midiendo CPU vs MPS sobre {N_MUESTRAS_TIMING} muestras (1 epoca) ===", flush=True)
    submuestra = datos_train[:N_MUESTRAS_TIMING]
    tiempos = {"cpu": medir_tiempo_epoca(submuestra, "cpu")}
    if torch.backends.mps.is_available():
        tiempos["mps"] = medir_tiempo_epoca(submuestra, "mps")
    print(f"  tiempos: {tiempos}", flush=True)
    DEVICE = min(tiempos, key=tiempos.get)
    print(f"  usando device={DEVICE} (mas rapido medido)", flush=True)

    _tensorizar_v2(datos_train, DEVICE)
    _tensorizar_v2(datos_val, DEVICE)

    # SS5.4: solo pares con etiqueta valida (mascara_etiquetada) -- ver docstring de
    # entrenar_atencion_v1.preparar_muestra_stage2.
    n_pares_total_train = sum(len(d["pares"]) for d in datos_train)
    etiquetas_train_todas = np.concatenate(
        [d["etiquetas"][d["mascara_etiquetada"]] for d in datos_train if len(d["pares"])]
    )
    n_pos = float(etiquetas_train_todas.sum())
    n_neg = float(len(etiquetas_train_todas) - n_pos)
    pos_weight = n_neg / max(n_pos, 1.0)
    n_excluidos = n_pares_total_train - len(etiquetas_train_todas)
    print(f"  pares candidatos train: {n_pares_total_train}  con etiqueta valida: {len(etiquetas_train_todas)}  "
          f"excluidos (tocan segmento sin asignar): {n_excluidos} ({n_excluidos / max(n_pares_total_train,1):.3f})  "
          f"positivos={int(n_pos)}  negativos={int(n_neg)}  pos_weight={pos_weight:.3f}", flush=True)

    print(f"\n=== Smoke test: {N_MUESTRAS_SMOKE} muestras, la perdida debe bajar ===", flush=True)
    modelo_smoke = at.ModeloAtencionV2().to(DEVICE)
    n_params = sum(p.numel() for p in modelo_smoke.parameters())
    print(f"  n_params v2 (con CodificadorTile): {n_params}", flush=True)
    opt_smoke = torch.optim.Adam(modelo_smoke.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE))
    smoke_datos = datos_train[:N_MUESTRAS_SMOKE]
    perdida_1 = epoca_entrenamiento(modelo_smoke, smoke_datos, opt_smoke, crit, DEVICE, tam_acumulacion=4)
    perdida_n = perdida_1
    for _ in range(9):
        perdida_n = epoca_entrenamiento(modelo_smoke, smoke_datos, opt_smoke, crit, DEVICE, tam_acumulacion=4)
    print(f"  perdida epoca 1: {perdida_1:.4f}  perdida epoca 10: {perdida_n:.4f}", flush=True)
    assert perdida_n < perdida_1, "smoke test: la perdida NO bajo -- no seguir sin resolver esto (SS10 checklist)"
    print("  smoke test OK", flush=True)

    print("\n=== Entrenamiento (early stopping en val, F1 de enlace por-par @ prob=0.5) ===", flush=True)
    modelo = at.ModeloAtencionV2().to(DEVICE)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=MAX_EPOCHS, eta_min=LR * 0.05)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE))

    mejor_f1_val, mejor_estado, mejor_epoca, sin_mejora = -1.0, None, -1, 0
    historial = []
    t0 = time.time()
    for epoca in range(MAX_EPOCHS):
        perdida_train = epoca_entrenamiento(modelo, datos_train, optimizador, criterio, DEVICE,
                                             tam_acumulacion=TAMANO_ACUMULACION)
        programador.step()
        f1_val, precision_val, recall_val = evaluar_val_por_par(modelo, datos_val, DEVICE)
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
    torch.save(mejor_estado, SALIDA / "atencion_v2_pesos.pt")

    print("\n=== Barrido umbral x decodificador sobre val (400 muestras) ===", flush=True)
    filas_barrido = []
    for nombre_decod in ("hungarian", "greedy"):
        for umbral in UMBRALES_BARRIDO:
            _, res = decodificar_evaluar(modelo, datos_val, nombre_decod, umbral, DEVICE)
            filas_barrido.append({"decodificador": nombre_decod, "umbral": umbral, **res})
            print(f"  [{nombre_decod}] umbral={umbral:>4}  track_f1={res['track_f1']}  "
                  f"frac_id_switch={res['frac_id_switch']}  fragmentos_por_gt={res['fragmentos_por_gt']}  "
                  f"precision={res['track_precision']}  recall={res['track_recall']}", flush=True)

    # cumple_barras_seleccion_val != veredicto de Gate B (ese se calcula en el notebook SS7,
    # con el frac_id_switch de TEST contra el 0.103 de DecNet -- ver elegir_punto_operacion).
    elegido = elegir_punto_operacion(filas_barrido)
    print(f"\n  elegido: decodificador={elegido['decodificador']}  umbral={elegido['umbral']}  "
          f"cumple_barras_seleccion_val={elegido['cumple_barras_seleccion_val']}  (val: "
          f"track_f1={elegido['track_f1']}  frac_id_switch={elegido['frac_id_switch']}  "
          f"fragmentos_por_gt={elegido['fragmentos_por_gt']})",
          flush=True)

    print("\n=== Test: decodificando con el punto elegido (400 muestras) ===", flush=True)
    nombres_test = [d.name for d in sorted((RAIZ / "datasets" / "test").glob("sample_*"))]
    datos_test, fallos_test, n_ceros_test = preparar_split_v2(nombres_test, "test", con_gt=False)
    n_tiles_test = sum(2 * len(d["segmentos"]) for d in datos_test)
    print(f"  tiles_en_blanco test: {n_ceros_test}/{n_tiles_test} ({n_ceros_test / max(n_tiles_test,1):.3f})",
          flush=True)
    _tensorizar_v2(datos_test, DEVICE)
    tr, res = decodificar_evaluar(modelo, datos_test, elegido["decodificador"], elegido["umbral"], DEVICE)
    tr.to_csv(SALIDA / "atencion_v2_fila3b.csv", index=False)

    print("\n=== Fila 3b (KymoButler trackness + atencion v2, con tiles de DecNet), test ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    resumen = {
        "device": DEVICE, "tiempos_cpu_vs_mps": tiempos, "n_params": n_params,
        "pos_weight": pos_weight, "n_pos_train": int(n_pos), "n_neg_train": int(n_neg),
        "n_pares_total_train": n_pares_total_train, "n_excluidos_sin_asignar_train": n_excluidos,
        "frac_tiles_en_blanco_train": n_ceros_train / max(n_tiles_train, 1),
        "frac_tiles_en_blanco_val": n_ceros_val / max(n_tiles_val, 1),
        "frac_tiles_en_blanco_test": n_ceros_test / max(n_tiles_test, 1),
        "smoke_test": {"perdida_epoca_1": perdida_1, "perdida_epoca_10": perdida_n},
        "mejor_epoca": mejor_epoca, "f1_val_por_par": mejor_f1_val,
        "historial_entrenamiento": historial,
        "barrido_umbral_val": filas_barrido,
        "punto_elegido": elegido,
        "fila_3b_test": res,
        "fallos_train": fallos_train, "fallos_val": fallos_val, "fallos_test": fallos_test,
        "hiperparametros": {
            "d": modelo.d, "max_epochs": MAX_EPOCHS, "paciencia": PACIENCIA,
            "tam_acumulacion": TAMANO_ACUMULACION, "lr": LR, "semilla": SEMILLA,
        },
    }
    (SALIDA / "atencion_v2_resumen.json").write_text(json.dumps(resumen, indent=2, default=str))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
