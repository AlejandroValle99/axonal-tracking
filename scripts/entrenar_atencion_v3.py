"""Stage 2c -- v3: atencion sobre segmentos CON dustbin (rechazo aprendido).

**Por que existe** (`results/asociacion/diagnostico_rechazo.json`, medido con
`scripts/diagnostico_rechazo.py`): v1/v2 no fallan al desambiguar -- aciertan 0.58/0.60
del argmax en grupos ambiguos contra 0.42 de eleccion aleatoria -- sino al RECHAZAR. El
50.7% de los segmentos con candidatos NO tiene sucesor verdadero y v1 los enlaza al
100% igual en su punto de operacion. Cada enlace falso encadena dos particulas en una
polilinea, y eso es un ID-switch.

La causa es estructural: v1/v2 emiten una sigmoide independiente por par y el
decodificador aplica UN umbral global; en el punto que iguala la fragmentacion de DecNet
ese umbral acepta cualquier par por encima de ~1% de probabilidad. v3 cambia SOLO eso:
softmax por segmento sobre [candidatos ; dustbin], asi el rechazo compite localmente
contra los candidatos reales. Tokens, PE, encoder y head de pares identicos a v1 -- la
comparacion fila 3 (v1) vs fila 3c (v3) aisla el rechazo.

Pasos: identicos a `entrenar_atencion_v1.py` salvo (1) el modelo, (2) la perdida
(cross-entropy por grupo en vez de BCE por par -- sin `pos_weight`, el desbalance
desaparece por construccion), y (3) el punto de operacion: `margen=1.0` es
**parameter-free** (decide el dustbin), no hay que elegirlo en val. Se barre igual para
reportar la curva y para poder aplicar la misma regla de Gate B que las filas 2/3/3b.

Salida: `results/asociacion/atencion_v3_pesos.pt`, `..._resumen.json`, `..._fila3c.csv`.
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

SEMILLA = int(os.environ.get("SEMILLA_V3", "0"))
# Variante: incluir segmentos estaticos sin asignar como grupos con objetivo dustbin.
# Es la variante pre-registrada en `plan/rejection-dustbin-guide.md` SS3.1 para el caso
# "la compuerta falla especificamente por rechazo", que es lo que midio el v3 base.
DUSTBIN_ESTATICOS = os.environ.get("V3_DUSTBIN_ESTATICOS", "0") == "1"
MAX_EPOCHS = 60
PACIENCIA = 15
TAMANO_ACUMULACION = 8
LR = 1e-3
N_MUESTRAS_SMOKE = 40
# Submuestra de val para las metricas de trayectoria por epoca (registro, no seleccion).
# Se toma `datos_val[::PASO_VAL_TRAYECTORIA]`, NO `datos_val[:N]`: el split de val esta
# ordenado en 8 bloques de 50 muestras, uno por perfil del generador, asi que un prefijo
# cubre solo los primeros perfiles -- y son los mas faciles (`datasets/val/manifest.csv`:
# n_particles 23.5 en las primeras 100 contra 28.8 en el resto; 1.95 GT moviles por
# muestra contra 5.7 en test). Es el mismo error de muestreo por prefijo que
# `docs/revision-rumbo-vit.md` SS1 documenta para NB10 (`sorted(test)[:40]`) y que
# `plan/notebook-11-review.md` SS1.3 documenta para la fila 2 (`train[:100]`).
PASO_VAL_TRAYECTORIA = 4  # 400 / 4 = 100 muestras, abarcando los 8 perfiles
MARGENES = [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
MARGEN_SIN_PARAMETROS = 1.0
FRAG_MAX_GATE_B = 1.282   # fragmentos/GT de DecNet
F1_MIN_GATE_B = 0.968     # track_f1 de DecNet -- las dos barras secundarias de Gate B (plan SS8)


def preparar_muestra_v3(nombre: str, split: str, *, con_gt: bool) -> dict:
    """Como `entrenar_atencion_v1.preparar_muestra_stage2` pero agregando los grupos
    de softmax por segmento (`at.construir_grupos`). Mismo pipeline de segmentacion
    que Gate A / Stage 1 / v1 / v2: los cuatro comparten `preparar_muestra`."""
    segmentos, caract, pares, verdaderos, asignaciones, e, _pre, _skel = preparar_muestra(nombre, split)
    T, L = e["kymo"].shape
    escala_v = (L / T) if T > 0 else 1.0
    ids_asignados = [a.particle_id for a in asignaciones]
    es_estatico = None
    if con_gt and DUSTBIN_ESTATICOS:
        # mismo criterio que `evaluar_kymobutler_400.es_movil`, sobre el segmento solo
        es_estatico = [
            (seg.puntos[:, 1].max() - seg.puntos[:, 1].min()) < MIN_DESP for seg in segmentos
        ]
    salida = {
        "nombre": nombre,
        "segmentos": segmentos,
        "pares": pares,
        "escena": e,
        "features": at.construir_features_tokens(segmentos, caract, T, L, e["kymo"]),
        "t_ini_norm": np.array([s.t_ini / T for s in segmentos], dtype=np.float32),
        "x_ini_norm": np.array([s.x_ini / L for s in segmentos], dtype=np.float32),
        "geom_pares": at.construir_geom_feats_pares(
            segmentos, caract, pares, max_salto_px=MAX_SALTO_PX,
            max_gap_frames=MAX_GAP_FRAMES, escala_v=escala_v,
        ),
    }
    for lado, clave in (("sucesor", "grupos_suc"), ("predecesor", "grupos_pred")):
        salida[clave] = at.construir_grupos(
            pares, len(segmentos),
            verdaderos if con_gt else None,
            ids_asignados if con_gt else None,
            lado=lado,
            es_estatico=es_estatico,
        )
    return salida


def cargar_split(nombres: list[str], split: str, *, con_gt: bool) -> tuple[list[dict], list[dict]]:
    datos, fallos = [], []
    for nombre in nombres:
        try:
            datos.append(preparar_muestra_v3(nombre, split, con_gt=con_gt))
        except Exception as exc:  # noqa: BLE001 -- una muestra rota no debe tumbar el split
            fallos.append({"muestra": nombre, "error": repr(exc)})
    return datos, fallos


def _tensorizar(datos: list[dict]) -> None:
    for d in datos:
        d["features_t"] = torch.from_numpy(d["features"])
        d["t_ini_t"] = torch.from_numpy(d["t_ini_norm"])
        d["x_ini_t"] = torch.from_numpy(d["x_ini_norm"])
        d["geom_t"] = torch.from_numpy(d["geom_pares"])
        d["pares_t"] = (
            torch.tensor(d["pares"], dtype=torch.long) if d["pares"] else torch.zeros((0, 2), dtype=torch.long)
        )
        for clave in ("grupos_suc", "grupos_pred"):
            g = d[clave]
            if g is None:
                continue
            d[clave] = {
                "idx": torch.from_numpy(g["idx"]),
                "mask": torch.from_numpy(g["mask"]),
                "objetivo": torch.from_numpy(g["objetivo"]),
                "segmentos": torch.from_numpy(g["segmentos"]),
                "en_perdida": torch.from_numpy(g["en_perdida"]),
            }


def _forward(modelo: at.ModeloAtencionV3, d: dict):
    return modelo(d["features_t"], d["t_ini_t"], d["x_ini_t"], d["pares_t"], d["geom_t"])


def epoca_entrenamiento(modelo, datos_train, optimizador, *, tam_acumulacion) -> float:
    modelo.train()
    random.shuffle(datos_train)
    perdida_acum, n_total = 0.0, 0
    optimizador.zero_grad()
    for i, d in enumerate(datos_train):
        if not d["pares"] or (d["grupos_suc"] is None and d["grupos_pred"] is None):
            continue
        logits, dust_s, dust_p = _forward(modelo, d)
        perdida, n_grupos = at.perdida_dustbin(logits, dust_s, dust_p, d["grupos_suc"], d["grupos_pred"])
        if n_grupos == 0:
            continue
        (perdida / tam_acumulacion).backward()
        perdida_acum += float(perdida.item()) * n_grupos
        n_total += n_grupos
        if (i + 1) % tam_acumulacion == 0:
            optimizador.step()
            optimizador.zero_grad()
    optimizador.step()
    optimizador.zero_grad()
    return perdida_acum / max(n_total, 1)


@torch.no_grad()
def exactitud_grupos(modelo, datos) -> float:
    """Fraccion de grupos donde el argmax del softmax (candidatos + dustbin) cae en
    la respuesta correcta. Mide desambiguacion Y rechazo en un solo numero, sin
    umbral ni decodificador -- el analogo honesto del `f1_val_por_par` de v1/v2."""
    modelo.eval()
    ok = n = 0
    for d in datos:
        if not d["pares"]:
            continue
        logits, dust_s, dust_p = _forward(modelo, d)
        for dust, grupos in ((dust_s, d["grupos_suc"]), (dust_p, d["grupos_pred"])):
            if grupos is None:
                continue
            sel = grupos["en_perdida"]
            if not bool(sel.any()):
                continue
            m = at._matriz_grupos(logits, dust, grupos["idx"][sel], grupos["mask"][sel])
            ok += int((m.argmax(dim=1) == grupos["objetivo"][sel]).sum())
            n += m.shape[0]
    return ok / max(n, 1)


@torch.no_grad()
def decodificar_evaluar(modelo, datos, margen: float):
    """Decodifica SIN umbral global: admisibles por comparacion contra el dustbin
    (`at.costos_dustbin`), y `decodificar_bipartito` resuelve un-sucesor/
    un-predecesor sobre ese conjunto ya filtrado. El `umbral` que se le pasa es una
    cota laxa (los costos estan acotados en -log(1e-6) ~= 13.8 por `piso_prob`), no
    un punto de operacion."""
    modelo.eval()
    escenas = {}
    for d in datos:
        polis = []
        if d["pares"]:
            logits, dust_s, dust_p = _forward(modelo, d)
            costos = at.costos_dustbin(
                logits, dust_s, dust_p, d["grupos_suc"], d["grupos_pred"], d["pares"], margen=margen,
            )
            cadenas = aso.decodificar_bipartito(
                len(d["segmentos"]), list(costos), costos, umbral=1e3,
            )
            polis = aso.polilineas_desde_cadenas(cadenas, d["segmentos"], d["escena"]["kymo"])
            polis = [p for p in polis if es_movil(p)]
        escenas[d["nombre"]] = {**d["escena"], "polilineas": polis}
    tr, res = ev.evaluar_trayectorias_polilineas(escenas, min_desplazamiento_px=MIN_DESP)
    mm = tr[tr.gt_id != -1]
    frag = mm.groupby(["muestra", "gt_id"]).size()
    return tr, {**res, "fragmentos_por_gt": round(float(frag.mean()), 3) if len(frag) else None}


def puntaje_early_stopping(res: dict) -> float:
    """Un solo numero, lexicografico y alineado con las TRES barras de Gate B:

        violacion = max(0, frag - 1.282) + max(0, 0.968 - track_f1)
        puntaje   = 100 * violacion + frac_id_switch

    Entre modelos que cumplen las dos barras secundarias (violacion = 0) ordena por
    `frac_id_switch`, la metrica primaria; si alguna barra se viola, la violacion
    domina (frac_id_switch <= 1 siempre).

    **Por que no basta penalizar solo la fragmentacion** (version anterior de esta
    funcion, corregida a mitad de la primera corrida): un dustbin que rechaza de mas
    baja `frac_id_switch` casi a cero SIN sobre-fragmentar -- simplemente pierde
    trayectorias, y eso cae en `track_f1`, no en `fragmentos_por_gt`. La corrida
    descartada mostraba exactamente eso en la epoca 0: switch 0.036 y frag 1.201
    (las dos barras que si se miraban) con track_f1 0.937, por DEBAJO del 0.968 de
    Gate B. Sin el termino de F1, el early stopping habria guardado un checkpoint
    que no puede pasar la compuerta. Es la version simetrica del artefacto de
    fragmentacion que SS7 item 3 ya documenta."""
    if res["fragmentos_por_gt"] is None or res["track_f1"] is None:
        return float("inf")
    violacion = (
        max(0.0, res["fragmentos_por_gt"] - FRAG_MAX_GATE_B)
        + max(0.0, F1_MIN_GATE_B - res["track_f1"])
    )
    return 100.0 * violacion + res["frac_id_switch"]


def main() -> None:
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    sufijo = ("_est" if DUSTBIN_ESTATICOS else "") + ("" if SEMILLA == 0 else f"_s{SEMILLA}")
    print(f"variante: DUSTBIN_ESTATICOS={DUSTBIN_ESTATICOS}  semilla={SEMILLA}  sufijo={sufijo!r}", flush=True)

    print("=== Preparando train (800) + val (400) ===", flush=True)
    t0 = time.time()
    nombres_train = [d.name for d in sorted((RAIZ / "datasets" / "train").glob("sample_*"))]
    nombres_val = [d.name for d in sorted((RAIZ / "datasets" / "val").glob("sample_*"))]
    datos_train, fallos_train = cargar_split(nombres_train, "train", con_gt=True)
    datos_val, fallos_val = cargar_split(nombres_val, "val", con_gt=True)
    _tensorizar(datos_train)
    _tensorizar(datos_val)
    n_grupos_train = sum(
        (0 if d[k] is None else int(d[k]["en_perdida"].sum()))
        for d in datos_train for k in ("grupos_suc", "grupos_pred")
    )
    n_dustbin = sum(
        int(((d[k]["objetivo"] == d[k]["mask"].sum(dim=1) - 1) & d[k]["en_perdida"]).sum())
        for d in datos_train for k in ("grupos_suc", "grupos_pred") if d[k] is not None
    )
    print(f"  {time.time() - t0:.0f}s -- train {len(datos_train)} ok / {len(fallos_train)} fallos, "
          f"val {len(datos_val)} ok / {len(fallos_val)} fallos", flush=True)
    print(f"  grupos de entrenamiento: {n_grupos_train}  de los cuales el objetivo es DUSTBIN: "
          f"{n_dustbin} ({n_dustbin / max(n_grupos_train, 1):.1%}) -- esta es la senal de rechazo "
          f"que v1/v2 no tenian", flush=True)

    datos_val_tray = datos_val[::PASO_VAL_TRAYECTORIA]  # estratificado, ver constante

    print(f"\n=== Smoke test: {N_MUESTRAS_SMOKE} muestras, la perdida debe bajar ===", flush=True)
    modelo_smoke = at.ModeloAtencionV3()
    opt_smoke = torch.optim.Adam(modelo_smoke.parameters(), lr=LR)
    sub = datos_train[:N_MUESTRAS_SMOKE]
    p1 = epoca_entrenamiento(modelo_smoke, sub, opt_smoke, tam_acumulacion=4)
    pn = p1
    for _ in range(9):
        pn = epoca_entrenamiento(modelo_smoke, sub, opt_smoke, tam_acumulacion=4)
    print(f"  perdida epoca 1: {p1:.4f}  epoca 10: {pn:.4f}", flush=True)
    assert pn < p1, "smoke test: la perdida NO bajo -- no seguir sin resolver esto"
    print("  smoke test OK", flush=True)

    print("\n=== Entrenamiento (early stopping en exactitud_grupos de val) ===", flush=True)
    modelo = at.ModeloAtencionV3()
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"  n_params v3: {n_params} (v1: 844289)", flush=True)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=MAX_EPOCHS, eta_min=LR * 0.05)

    # Early stopping en `exactitud_grupos`, NO en la trayectoria a margen fijo.
    # Motivo, medido en la corrida descartada del 2026-09-19 (16 epocas): `track_f1`
    # se queda clavado en 0.931-0.939 y `frac_id_switch` oscila 0.036-0.063 SIN
    # tendencia, mientras `exactitud_grupos` sube monotona 0.528 -> 0.655. Evaluar la
    # trayectoria a un margen FIJO mezcla dos cosas distintas -- la calidad del modelo
    # y el punto de operacion -- y ahi el ruido del punto domina. Se separan: el modelo
    # se elige por `exactitud_grupos` y el punto de operacion por el barrido de margen
    # del final.
    #
    # Esto NO repite el error de NB11 (parar en F1-por-par, desacoplada de la
    # trayectoria): `exactitud_grupos` es el argmax del MISMO softmax que el
    # decodificador consulta a margen=1.0, asi que esta acoplado por construccion, y
    # a diferencia de la F1-por-par pooleada mide desambiguacion Y rechazo juntos.
    # Las metricas de trayectoria se siguen registrando en cada epoca para el archivo.
    mejor, mejor_estado, mejor_epoca, sin_mejora = -1.0, None, -1, 0
    historial = []
    t0 = time.time()
    for epoca in range(MAX_EPOCHS):
        perdida = epoca_entrenamiento(modelo, datos_train, optimizador, tam_acumulacion=TAMANO_ACUMULACION)
        programador.step()
        exact = exactitud_grupos(modelo, datos_val)
        _, res = decodificar_evaluar(modelo, datos_val_tray, MARGEN_SIN_PARAMETROS)
        puntaje = puntaje_early_stopping(res)  # solo registro, ver comentario arriba
        historial.append({
            "epoca": epoca, "perdida_train": perdida, "exactitud_grupos_val": exact,
            "val_track_f1": res["track_f1"], "val_frac_id_switch": res["frac_id_switch"],
            "val_fragmentos_por_gt": res["fragmentos_por_gt"], "puntaje": puntaje,
            "lr": optimizador.param_groups[0]["lr"],
        })
        print(f"  epoca {epoca:3d} ({(time.time()-t0)/60:5.1f} min) perdida={perdida:.4f} "
              f"exact_grupos={exact:.4f} | val f1={res['track_f1']} switch={res['frac_id_switch']} "
              f"frag={res['fragmentos_por_gt']} -> puntaje={puntaje:.4f}", flush=True)
        if exact > mejor:
            mejor, mejor_epoca, sin_mejora = exact, epoca, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                print(f"  early stopping en epoca {epoca}", flush=True)
                break

    modelo.load_state_dict(mejor_estado)
    torch.save(mejor_estado, SALIDA / f"atencion_v3_pesos{sufijo}.pt")
    print(f"\nmejor epoca: {mejor_epoca}  exactitud_grupos={mejor:.4f}", flush=True)

    print("\n=== Barrido de margen sobre val (400 muestras) ===", flush=True)
    barrido = []
    for margen in MARGENES:
        _, res = decodificar_evaluar(modelo, datos_val, margen)
        barrido.append({"decodificador": "dustbin+hungarian", "margen": margen, **res})
        print(f"  margen={margen:>5}  track_f1={res['track_f1']}  frac_id_switch={res['frac_id_switch']}  "
              f"fragmentos_por_gt={res['fragmentos_por_gt']}", flush=True)

    elegido_gate_b = elegir_punto_operacion(barrido)
    sin_param = next(f for f in barrido if f["margen"] == MARGEN_SIN_PARAMETROS)
    print(f"\n  punto SIN PARAMETROS (margen=1.0, decide el dustbin): "
          f"switch={sin_param['frac_id_switch']} frag={sin_param['fragmentos_por_gt']}", flush=True)
    print(f"  punto por la regla de Gate B (comparabilidad con filas 2/3/3b): margen="
          f"{elegido_gate_b['margen']} cumple={elegido_gate_b['cumple_barras_seleccion_val']}", flush=True)

    print("\n=== Test: 400 muestras, UNA vez, en los dos puntos ===", flush=True)
    nombres_test = [d.name for d in sorted((RAIZ / "datasets" / "test").glob("sample_*"))]
    datos_test, fallos_test = cargar_split(nombres_test, "test", con_gt=False)
    _tensorizar(datos_test)
    filas_test = {}
    for etiqueta, margen in (("margen_1.0", MARGEN_SIN_PARAMETROS), ("gate_b", elegido_gate_b["margen"])):
        tr, res = decodificar_evaluar(modelo, datos_test, margen)
        filas_test[etiqueta] = {**res, "margen": margen}
        tr.to_csv(SALIDA / f"atencion_v3_fila3c_{etiqueta}{sufijo}.csv", index=False)
        print(f"  [{etiqueta}] margen={margen}: " + "  ".join(
            f"{k}={res[k]}" for k in ("track_f1", "frac_id_switch", "fragmentos_por_gt", "err_pos_um_medio")
        ), flush=True)

    resumen = {
        "semilla": SEMILLA, "dustbin_estaticos": DUSTBIN_ESTATICOS, "n_params": n_params,
        "n_grupos_train": n_grupos_train, "n_grupos_objetivo_dustbin": n_dustbin,
        "smoke_test": {"perdida_epoca_1": p1, "perdida_epoca_10": pn},
        "mejor_epoca": mejor_epoca, "mejor_exactitud_grupos_val": mejor,
        "criterio_early_stopping": "exactitud_grupos (ver comentario en main)",
        "puntaje_gate_b_en_mejor_epoca": historial[mejor_epoca]["puntaje"],
        "historial_entrenamiento": historial,
        "barrido_margen_val": barrido,
        "punto_sin_parametros": sin_param,
        "punto_gate_b": elegido_gate_b,
        "fila_3c_test": filas_test,
        "fallos_train": fallos_train, "fallos_val": fallos_val, "fallos_test": fallos_test,
        "hiperparametros": {
            "d": modelo.d, "max_epochs": MAX_EPOCHS, "paciencia": PACIENCIA,
            "tam_acumulacion": TAMANO_ACUMULACION, "lr": LR, "semilla": SEMILLA,
            "paso_val_trayectoria": PASO_VAL_TRAYECTORIA,
        },
    }
    (SALIDA / f"atencion_v3_resumen{sufijo}.json").write_text(json.dumps(resumen, indent=2, default=str))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
