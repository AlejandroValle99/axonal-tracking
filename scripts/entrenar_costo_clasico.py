"""Stage 1 (`plan/association-transformer-guide.md` SS4): costo clasico de
asociacion, el piso obligatorio del requisito 5.4 (comparacion clasico-vs-ML) antes
de escribir el modulo de atencion de Stage 2.

Todo sobre la trackness REAL de KymoButler (fuente 2 de Gate A) -- no las fuentes
GT, que solo sirven para validar el extractor. Pasos:

1. Segmentos + pares candidatos sobre **train** (800 muestras) con el presupuesto
   de poda de SS4.1; reporte de perdida por poda (cuantos enlaces verdaderos caen
   fuera del presupuesto).
2. Ajuste de pesos (SS4.2) maximizando F1 de enlace por-par sobre los pares
   candidatos de train, pooleados. Los 5 features se NORMALIZAN antes de ajustar
   (si no, `d_pos` domina por escala -- 0-80px contra `1-cos_theta` en 0-2 -- y el
   optimizador no puede reponderar nada; ver justificacion en el docstring de
   `ajustar_pesos`). Nunca se toca test en este paso.
3. Barrido de umbral x decodificador (Hungarian vs greedy, SS4.3) sobre las 400
   muestras de VAL completas (antes: 100 de train en orden `sorted()[:100]` -- el
   mismo patron `sorted(test)[:40]` que `docs/revision-rumbo-vit.md` SS1 documenta
   como error para NB10; esas 100 resultaron un subconjunto facil, ver
   `plan/notebook-11-review.md` SS1.3) -- el ajuste de pesos optimiza F1 por PAR,
   que no es lo mismo que buen F1/fragmentacion a nivel TRAYECTORIA; este barrido
   elige el punto de operacion honesto (val, nunca test), con la regla de Gate B
   (SS8) aplicada igual a las filas 2/3/3b (`elegir_punto_operacion`).
4. Fila 2 del ablation matrix (SS6): decodificar + evaluar con el punto elegido
   sobre las 400 muestras de **test**, mismo harness que Gate A. Test se toca UNA
   sola vez, aca.

Requiere `results/asociacion/cache/{train,val,test}/` tibios (`scripts/evaluar_gate_a.py`
y `scripts/preparar_cache_kymobutler.py {train,val}`).

Salida: `results/asociacion/costo_clasico_resumen.json`,
`results/asociacion/costo_clasico_fila2.csv`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from evaluar_kymobutler_400 import cargar_escena, es_movil  # SS12.2 -- no reescribir

from axonal_tracking import asociacion as aso
from axonal_tracking import etiquetas_deteccion as ed
from axonal_tracking import evaluacion as ev

CACHE_DIR = RAIZ / "results" / "asociacion" / "cache"
SALIDA = RAIZ / "results" / "asociacion"

MIN_DESP = ed.MIN_DESPLAZAMIENTO_PX_MOVIL
THRESHOLD, MIN_SIZE, MIN_FRAMES = 0.2, 10, 10  # identicos a track_bidirectional (SS3.1/SS12.4)
MIN_FILAS_SEGMENTO = 5
# SS4.1: se probo aflojar a 60/80 (advisor) -- recupero solo 280 de 671 enlaces
# perdidos (5.0% -> 2.9%) mientras DUPLICABA los pares candidatos (83k -> 171k),
# empeorando la precision por-par. Mal cambio: el 5% residual en 30/40 es
# probablemente estructural (el segmento intermedio ya fue descartado por
# `min_filas`, ningun presupuesto de hueco lo recupera). Se documenta el 5% en vez
# de aflojar a ciegas.
MAX_GAP_FRAMES = 30.0
MAX_SALTO_PX = 40.0


def preparar_muestra(nombre: str, split: str):
    """De una muestra cacheada a (segmentos, caracteristicas, pares, verdaderos,
    asignaciones, escena, preprocessed, skel) -- el mismo pipeline de segmentacion
    que Gate A (fuente `kymobutler`), reusado tal cual para que la comparacion
    entre oraculo, costo clasico y atencion (v1/v2) sea sobre el MISMO conjunto de
    segmentos. `preprocessed`/`skel` se devuelven ademas de `segmentos` porque
    Stage 2b (v2, SS5.6) los necesita para extraer los tiles de DecNet -- no
    recomputar el esqueleto una segunda vez en el script de v2. `asignaciones` se
    devuelve para que Stage 2 (v1/v2) pueda excluir del entrenamiento los pares que
    tocan un segmento sin asignar (`particle_id == -1`) -- SS5.4 pide excluirlos de
    la perdida, no tratarlos como negativos (`plan/notebook-11-review.md` SS2.4)."""
    d = RAIZ / "datasets" / split / nombre
    e = cargar_escena(d)
    L = e["kymo"].shape[1]
    cache = np.load(CACHE_DIR / split / f"{nombre}.npz")
    pred, pre = cache["prediction"], cache["preprocessed"]
    assert pre.shape == e["kymo"].shape, (
        f"{nombre} ({split}): preprocessed.shape {pre.shape} != kymo.shape {e['kymo'].shape}"
    )
    skel = aso.esqueleto_kymobutler(pred, pre.shape, THRESHOLD, MIN_SIZE, MIN_FRAMES)
    segmentos = aso.partir_en_segmentos(skel, min_filas=MIN_FILAS_SEGMENTO)
    segmentos = aso.fusionar_fragmentos_paralelos(segmentos, min_filas=MIN_FILAS_SEGMENTO)
    asignaciones = aso.asignar_gt(segmentos, e["positions"], e["pixel_scale_um"], L)
    caract = aso.caracteristicas_segmentos(segmentos, e["kymo"])
    pares = aso.pares_candidatos(segmentos, max_gap_frames=MAX_GAP_FRAMES, max_salto_px=MAX_SALTO_PX)
    verdaderos = aso.enlaces_verdaderos(segmentos, asignaciones)
    return segmentos, caract, pares, verdaderos, asignaciones, e, pre, skel


def features_par(segmentos, caract, i: int, j: int) -> np.ndarray:
    """`(d_pos, 1-cos_theta, dv, dI, gap)` -- las cinco cantidades del costo (SS4.2),
    en el orden de pesos `(a, b, g, d, e)`."""
    d_pos, cos_theta, dv, gap = aso.geom_feats(segmentos[i], caract[i], segmentos[j], caract[j])
    dI = abs(caract[i].intensidad_media - caract[j].intensidad_media)
    return np.array([d_pos, 1 - cos_theta, dv, dI, gap])


def _f1_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def ajustar_pesos(X: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], float, np.ndarray, dict]:
    """Maximiza F1 de enlace por-par sobre pares candidatos pooleados de train
    (SS4.2), variando `(b, g, d, e, umbral)` en ESPACIO NORMALIZADO -- `a=1.0` fijo
    en unidades normalizadas, no crudas.

    **Por que normalizar**: `d_pos` vive en 0-80px (`max_salto_px`) mientras
    `1-cos_theta` vive en 0-2 -- si se fija `a=1.0` en unidades crudas (como hacia
    una version anterior de este script), toda otra columna queda forzada a pesos
    minusculos solo para no ser dominada por la escala de `d_pos`, y el ajuste
    degenera a "enlazar si los extremos estan cerca" sin aprovechar direccion,
    velocidad o intensidad. Normalizando (z-score) las 5 columnas antes de ajustar,
    las 5 quedan en una escala comparable y `a=1.0` es una eleccion de gauge
    inofensiva (el costo es lineal: escalar TODOS los pesos + umbral por la misma
    constante no cambia la decision `costo <= umbral`, ese grado de libertad es
    redundante). Los pesos devueltos estan reconvertidos a unidades CRUDAS (para que
    `costo_asociacion`/`geom_feats`, que trabajan en unidades fisicas, no cambien);
    `escala` se devuelve y se guarda para que Stage 2 (SS5.2, que tambien normaliza
    sus features) pueda usar el mismo criterio documentado."""
    escala = X.std(axis=0)
    escala[escala == 0] = 1.0
    X_norm = X / escala

    def objetivo(params):
        b, g, d, e, umbral = params
        if min(b, g, d, e, umbral) < 0:
            return 1e6  # pesos/umbral negativos no tienen sentido fisico
        pesos_norm = np.array([1.0, b, g, d, e])
        costos = X_norm @ pesos_norm
        f1, _, _ = _f1_precision_recall(y, costos <= umbral)
        return -f1

    arranques = [
        [1.0, 1.0, 1.0, 1.0, 2.0],
        [0.5, 0.5, 0.5, 0.5, 1.0],
        [2.0, 1.0, 0.5, 2.0, 3.0],
        [1.0, 2.0, 2.0, 0.5, 1.5],
    ]
    mejor = None
    for x0 in arranques:
        r = minimize(objetivo, x0=x0, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-7})
        if mejor is None or r.fun < mejor.fun:
            mejor = r
    b, g, d, e, umbral_norm = mejor.x
    pesos_norm_vec = np.array([1.0, b, g, d, e])
    # convertir a unidades crudas: costo = X_norm @ pesos_norm = X_raw @ (pesos_norm/escala),
    # y el umbral (mismo valor numerico del costo) no cambia entre las dos parametrizaciones.
    pesos_raw = pesos_norm_vec / escala
    pesos = {"a": float(pesos_raw[0]), "b": float(pesos_raw[1]), "g": float(pesos_raw[2]),
              "d": float(pesos_raw[3]), "e": float(pesos_raw[4])}

    f1, precision, recall = _f1_precision_recall(y, (X_norm @ pesos_norm_vec) <= umbral_norm)
    diagnostico = {"f1_por_par": f1, "precision": precision, "recall": recall}
    return pesos, float(umbral_norm), escala, diagnostico


def preparar_split(nombres: list[str], split: str, *, con_gt: bool):
    """Prepara un conjunto de muestras y devuelve, por muestra, lo necesario para
    decodificar + evaluar. `con_gt=True` tambien arma `X, y` pooleados (train, para
    ajustar); `con_gt=False` es mas liviano (test, solo decodificacion)."""
    datos = {}
    X_list, y_list = [], []
    n_verdaderos_totales, n_verdaderos_sobreviven = 0, 0
    fallos = []
    for nombre in nombres:
        try:
            segmentos, caract, pares, verdaderos, _asig, e, _pre, _skel = preparar_muestra(nombre, split)
        except Exception as exc:  # noqa: BLE001 -- una muestra rota no debe tumbar el split
            fallos.append({"muestra": nombre, "error": repr(exc)})
            continue
        datos[nombre] = (segmentos, caract, pares, e)
        if con_gt:
            n_verdaderos_totales += len(verdaderos)
            n_verdaderos_sobreviven += len(verdaderos & set(pares))
            for pi, pj in pares:
                X_list.append(features_par(segmentos, caract, pi, pj))
                y_list.append((pi, pj) in verdaderos)
    resultado = {"datos": datos, "fallos": fallos}
    if con_gt:
        resultado["X"] = np.stack(X_list) if X_list else np.zeros((0, 5))
        resultado["y"] = np.array(y_list, dtype=bool)
        resultado["n_verdaderos_totales"] = n_verdaderos_totales
        resultado["n_verdaderos_sobreviven"] = n_verdaderos_sobreviven
    return resultado


FRAG_MAX_GATE_B = 1.282  # fragmentos/GT de DecNet (results/kymobutler/resumen_400.json, solo_moviles_subpixel)
F1_MIN_GATE_B = 0.968  # track_f1 de DecNet -- las dos barras de Gate B (plan SS8)


def elegir_punto_operacion(
    filas_barrido: list[dict], *, frag_max: float = FRAG_MAX_GATE_B, f1_min: float = F1_MIN_GATE_B,
) -> dict:
    """Elige el punto de operacion del barrido umbral x decodificador (SS4.3/SS5.5)
    con la REGLA UNICA de Gate B (plan SS8), la misma para las filas 2/3/3b: entre
    las filas con `fragmentos_por_gt <= frag_max` (el de DecNet, 1.282) Y
    `track_f1 >= f1_min` (el de DecNet, 0.968), la de MENOR `frac_id_switch` -- la
    metrica primaria del plan (SS7 item 1).

    **Historia -- dos versiones anteriores de esta funcion, ambas descartadas**
    (`plan/notebook-11-review.md` SS1.3): la primera elegia por `track_f1` maximo
    entre `fragmentos_por_gt <= 1.5` (una barra inventada en este script, no del
    plan) con una barra de respaldo mas floja si no habia candidatos -- eso hacia
    que las filas se seleccionaran con reglas DISTINTAS (fila 2 desde una barra,
    fila 3 desde la de respaldo referenciando el fragmentos_por_gt de TEST de la
    fila 2 -- una fuga de test hacia la seleccion de otra fila). La segunda elegia
    por menor `frac_id_switch` directamente, que resulto en el artefacto que SS7
    item 3 nombra: `frac_id_switch` 0.073 -> 0.008 en la fila 2 a costa de
    `fragmentos_por_gt` 1.604 -> 2.657 (fragmentar mas abarata el switch-rate
    "gratis"). La solucion no es una barra floja NI optimizar `frac_id_switch` sin
    mirar fragmentacion: es la barra EXACTA que el plan ya define (Gate B), fija e
    igual para las tres filas -- no hay eleccion de parametro que hacer.

    Si NINGUNA fila cumple las dos barras a la vez, esta funcion NO elige a ciegas:
    devuelve el punto de MENOR `fragmentos_por_gt` (desempate por mejor `track_f1`)
    con `cumple_barras_seleccion_val=False` explicito, para que el caller reporte la
    falla en vez de esconderla detras de un numero que parece un exito.

    **`cumple_barras_seleccion_val` NO es el veredicto de Gate B.** Es solo si estas
    dos barras (fragmentacion, F1) tienen algun candidato en el barrido de VAL. El
    veredicto real de Gate B (plan SS8) exige ademas `frac_id_switch` materialmente
    por debajo del 0.103 de DecNet, medido una sola vez en TEST -- eso se calcula en
    el notebook (11_asociacion_atencion.ipynb SS7), no aca. Una fila puede tener
    `cumple_barras_seleccion_val=True` y aun asi fallar Gate B (le paso a la fila 2:
    cumple las barras en val pero su `frac_id_switch` de test, 0.136, queda por
    ENCIMA del 0.103 de DecNet, no por debajo)."""
    candidatos = [f for f in filas_barrido if f["fragmentos_por_gt"] is not None
                  and f["fragmentos_por_gt"] <= frag_max and f["track_f1"] >= f1_min]
    if candidatos:
        elegido = dict(min(candidatos, key=lambda f: f["frac_id_switch"]))
        elegido["cumple_barras_seleccion_val"] = True
        return elegido
    cercano = dict(min(filas_barrido, key=lambda f: (f["fragmentos_por_gt"] or float("inf"), -f["track_f1"])))
    cercano["cumple_barras_seleccion_val"] = False
    return cercano


def evaluar_decodificado(datos: dict, decodificador, pesos, umbral) -> dict:
    """`enlazar_oracle` descarta segmentos con `particle_id == -1` (estaticos o sin
    asignar) gratis -- `asignar_gt` solo asigna moviles. El decodificador clasico no
    tiene ese filtro y emite una polilinea por CADA segmento sin enlazar, estatico
    incluido (las bandas verticales brillantes de las muestras). Sin filtrar, la
    fila 2 queda dominada por singletons estaticos: mismo filtro del lado de la
    PREDICCION que usa `evaluar_kymobutler_400.es_movil` para la fila de
    KymoButler, para que ambas filas sean comparables."""
    escenas = {}
    for nombre, (segmentos, caract, pares, e) in datos.items():
        cadenas = decodificador(segmentos, caract, pares, pesos, umbral=umbral)
        polis = aso.polilineas_desde_cadenas(cadenas, segmentos, e["kymo"])
        polis = [p for p in polis if es_movil(p)]
        escenas[nombre] = {**e, "polilineas": polis}
    tr, res = ev.evaluar_trayectorias_polilineas(escenas, min_desplazamiento_px=MIN_DESP)
    mm = tr[tr.gt_id != -1]
    frag = mm.groupby(["muestra", "gt_id"]).size()
    res = {**res, "fragmentos_por_gt": round(float(frag.mean()), 3) if len(frag) else None}
    return tr, res


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)

    print(f"=== Train: segmentos + pares candidatos (max_gap_frames={MAX_GAP_FRAMES}, "
          f"max_salto_px={MAX_SALTO_PX}), 800 muestras ===", flush=True)
    nombres_train = [d.name for d in sorted((RAIZ / "datasets" / "train").glob("sample_*"))]
    t0 = time.time()
    prep_train = preparar_split(nombres_train, "train", con_gt=True)
    X, y = prep_train["X"], prep_train["y"]
    frac_perdida_poda = (
        1 - prep_train["n_verdaderos_sobreviven"] / prep_train["n_verdaderos_totales"]
        if prep_train["n_verdaderos_totales"] else float("nan")
    )
    print(f"  {time.time() - t0:.0f}s -- {len(prep_train['datos'])} ok, {len(prep_train['fallos'])} fallos, "
          f"{X.shape[0]} pares candidatos, {int(y.sum())} positivos", flush=True)
    print(f"  perdida por poda (SS4.1): {prep_train['n_verdaderos_totales']} enlaces verdaderos, "
          f"{prep_train['n_verdaderos_sobreviven']} sobreviven -> frac_perdida={frac_perdida_poda:.4f}",
          flush=True)

    print("\n=== Ajustando pesos (Nelder-Mead multi-arranque, features normalizadas) ===", flush=True)
    pesos, umbral_base, escala, diag_ajuste = ajustar_pesos(X, y)
    print(f"  pesos (unidades crudas): {pesos}", flush=True)
    print(f"  escala (std por feature, d_pos/1-cos/dv/dI/gap): {escala.tolist()}", flush=True)
    print(f"  umbral (espacio normalizado)={umbral_base:.4f}  F1 por-par={diag_ajuste['f1_por_par']:.4f}  "
          f"precision={diag_ajuste['precision']:.4f}  recall={diag_ajuste['recall']:.4f}", flush=True)

    print("\n=== Barrido umbral x decodificador sobre val (400 muestras) ===", flush=True)
    nombres_val = [d.name for d in sorted((RAIZ / "datasets" / "val").glob("sample_*"))]
    prep_val = preparar_split(nombres_val, "val", con_gt=False)
    factores = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    filas_barrido = []
    for decod_nombre, decodificador in (
        ("hungarian", aso.enlazar_por_costo), ("greedy", aso.enlazar_por_costo_greedy),
    ):
        for factor in factores:
            umbral = umbral_base * factor
            _, res = evaluar_decodificado(prep_val["datos"], decodificador, pesos, umbral)
            filas_barrido.append({"decodificador": decod_nombre, "factor_umbral": factor,
                                    "umbral": umbral, **res})
            print(f"  [{decod_nombre}] factor={factor:>4}  track_f1={res['track_f1']}  "
                  f"frac_id_switch={res['frac_id_switch']}  fragmentos_por_gt={res['fragmentos_por_gt']}  "
                  f"precision={res['track_precision']}  recall={res['track_recall']}", flush=True)

    # Elegido en VAL con la regla de seleccion de SS8, nunca en test -- ver elegir_punto_operacion.
    # cumple_barras_seleccion_val != veredicto de Gate B (ese se calcula en el notebook SS7,
    # con el frac_id_switch de TEST contra el 0.103 de DecNet -- ver docstring de la funcion).
    elegido = elegir_punto_operacion(filas_barrido)
    print(f"\n  elegido: decodificador={elegido['decodificador']}  factor={elegido['factor_umbral']}  "
          f"umbral={elegido['umbral']:.4f}  cumple_barras_seleccion_val={elegido['cumple_barras_seleccion_val']}  "
          f"(val: track_f1={elegido['track_f1']}  frac_id_switch={elegido['frac_id_switch']}  "
          f"fragmentos_por_gt={elegido['fragmentos_por_gt']})", flush=True)

    decodificador_elegido = (
        aso.enlazar_por_costo if elegido["decodificador"] == "hungarian" else aso.enlazar_por_costo_greedy
    )
    umbral_elegido = elegido["umbral"]

    print("\n=== Test: decodificando con el punto elegido (400 muestras) ===", flush=True)
    nombres_test = [d.name for d in sorted((RAIZ / "datasets" / "test").glob("sample_*"))]
    prep_test = preparar_split(nombres_test, "test", con_gt=False)
    tr, res = evaluar_decodificado(prep_test["datos"], decodificador_elegido, pesos, umbral_elegido)
    tr.to_csv(SALIDA / "costo_clasico_fila2.csv", index=False)

    print("\n=== Fila 2 (KymoButler trackness + costo clasico), test ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    resumen = {
        "pesos": pesos,
        "escala_normalizacion": escala.tolist(),
        "poda": {
            "max_gap_frames": MAX_GAP_FRAMES, "max_salto_px": MAX_SALTO_PX,
            "n_verdaderos_totales_train": prep_train["n_verdaderos_totales"],
            "n_verdaderos_sobreviven_train": prep_train["n_verdaderos_sobreviven"],
            "frac_perdida": frac_perdida_poda,
        },
        "ajuste_train": {**diag_ajuste, "umbral_base_normalizado": umbral_base},
        "barrido_umbral_val": filas_barrido,
        "punto_elegido": elegido,
        "fila_2_test": res,
        "fallos_train": prep_train["fallos"],
        "fallos_test": prep_test["fallos"],
    }
    (SALIDA / "costo_clasico_resumen.json").write_text(json.dumps(resumen, indent=2, default=str))
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
