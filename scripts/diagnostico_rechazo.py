"""Diagnostico: en la asociacion a nivel segmento, .donde esta realmente el error?

Motivacion (`plan/notebook11-closeout.md`): las tres filas candidatas de NB11 (costo
clasico, atencion v1, atencion v2) pierden contra DecNet en `frac_id_switch`, pero el
notebook nunca separo las dos decisiones que un enlazador toma:

  (a) DESAMBIGUAR -- un segmento con varios candidatos, .cual es el correcto?
  (b) RECHAZAR    -- un segmento cuya particula termina, .no enlazarlo con nada?

Este script mide las dos por separado, sin pasar por el decodificador ni por ningun
umbral global, sobre el split de VAL:

1. Exactitud de argmax en grupos ambiguos (>=2 candidatos, con sucesor verdadero
   presente), contra la linea base de eleccion aleatoria 1/k. Incluye reglas de una sola
   feature (gap, velocidad, proximidad) como control: si una regla de una linea empata
   con un transformer entrenado, el problema no es la capacidad del modelo.
2. Tasa de ENLACE FALSO sobre los segmentos que NO tienen sucesor verdadero, en el punto
   de operacion real de cada metodo.
3. Velocidad como unica senal, alejando la ventana de ajuste de la juncion (`back-off`),
   para descartar que el feature este contaminado por el cruce mismo.

Salida: `results/asociacion/diagnostico_rechazo.json`.

Uso: uv run python scripts/diagnostico_rechazo.py [n_muestras_val]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # antes de `import torch` (SS12.8)

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from entrenar_atencion_v1 import _forward as fwd1
from entrenar_atencion_v1 import _tensorizar as tens1
from entrenar_atencion_v2 import _forward as fwd2
from entrenar_atencion_v2 import _tensorizar_v2, preparar_split_v2

from axonal_tracking import asociacion as aso
from axonal_tracking import atencion as at
from axonal_tracking import evaluacion as ev
from axonal_tracking.asociacion import _x_medio_por_t

SALIDA = RAIZ / "results" / "asociacion"
N_MUESTRAS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
UMBRAL_V1 = 5.0  # punto de operacion de la fila 3 (atencion_v1_resumen.json -> punto_elegido)


def pendiente_backoff(segmento, extremo: str, n_filas: int, saltar: int) -> float:
    """Pendiente local como `asociacion._pendiente_ventana`, pero descartando `saltar`
    filas del extremo que toca la juncion. Si el feature de velocidad estuviera
    contaminado por el cruce, alejarse deberia mejorarlo."""
    x_por_t = _x_medio_por_t(segmento)
    ts = np.array(sorted(x_por_t))
    if saltar:
        ts = ts[: len(ts) - saltar] if extremo == "fin" else ts[saltar:]
    if len(ts) < 2:
        return float("nan")
    ts = ts[-n_filas:] if extremo == "fin" else ts[:n_filas]
    return ev.velocidad_px_frame(ts, [x_por_t[t] for t in ts])


def main() -> None:
    nombres = [d.name for d in sorted((RAIZ / "datasets" / "val").glob("sample_*"))][:N_MUESTRAS]
    t0 = time.time()
    datos, fallos, _ = preparar_split_v2(nombres, "val", con_gt=True)
    _tensorizar_v2(datos, "cpu")
    tens1(datos)
    print(f"preparadas {len(datos)} muestras de val en {time.time() - t0:.0f}s "
          f"({len(fallos)} fallos)", flush=True)

    m1 = at.ModeloAtencionV1()
    m1.load_state_dict(torch.load(SALIDA / "atencion_v1_pesos.pt", map_location="cpu"))
    m1.eval()
    m2 = at.ModeloAtencionV2()
    m2.load_state_dict(torch.load(SALIDA / "atencion_v2_pesos.pt", map_location="cpu"))
    m2.eval()
    resumen_costo = json.loads((SALIDA / "costo_clasico_resumen.json").read_text())
    pesos = resumen_costo["pesos"]
    umbral_costo = resumen_costo["punto_elegido"]["umbral"]

    metodos = ["v1", "v2", "costo_clasico", "solo_gap", "solo_velocidad", "solo_proximidad"]
    aciertos = {k: 0 for k in metodos}
    n_grupos = 0
    tam_grupos: list[int] = []
    backoff = {s: [0, 0] for s in (0, 2, 4, 6)}
    n_con_sucesor = n_sin_sucesor = 0
    falsos = {"v1": 0, "costo_clasico": 0}

    with torch.no_grad():
        for d in datos:
            if not d["pares"]:
                continue
            logits1 = fwd1(m1, d).numpy()
            logits2 = fwd2(m2, d, "cpu").numpy()
            seg, car, pares = d["segmentos"], d["caracteristicas"], d["pares"]
            pos = {p: k for k, p in enumerate(pares)}
            e = d["escena"]
            asign = aso.asignar_gt(seg, e["positions"], e["pixel_scale_um"], e["kymo"].shape[1])
            verdaderos = {i: j for (i, j) in aso.enlaces_verdaderos(seg, asign)}

            por_i: dict[int, list[int]] = {}
            for i, j in pares:
                por_i.setdefault(i, []).append(j)

            for i, candidatos in por_i.items():
                tiene_sucesor = i in verdaderos and verdaderos[i] in candidatos
                if not tiene_sucesor:
                    n_sin_sucesor += 1
                    # .enlaza igual, en el punto de operacion real de cada metodo?
                    costos_v1 = [float(np.log1p(np.exp(-logits1[pos[(i, j)]]))) for j in candidatos]
                    if min(costos_v1) <= UMBRAL_V1:
                        falsos["v1"] += 1
                    costos_c = [aso.costo_asociacion(seg[i], car[i], seg[j], car[j], pesos)
                                for j in candidatos]
                    if min(costos_c) <= umbral_costo:
                        falsos["costo_clasico"] += 1
                    continue

                n_con_sucesor += 1
                if len(candidatos) < 2:
                    continue
                n_grupos += 1
                tam_grupos.append(len(candidatos))
                correcto = verdaderos[i]
                gf = [aso.geom_feats(seg[i], car[i], seg[j], car[j]) for j in candidatos]
                puntajes = {
                    "v1": [logits1[pos[(i, j)]] for j in candidatos],
                    "v2": [logits2[pos[(i, j)]] for j in candidatos],
                    "costo_clasico": [-aso.costo_asociacion(seg[i], car[i], seg[j], car[j], pesos)
                                      for j in candidatos],
                    "solo_proximidad": [-g[0] for g in gf],
                    "solo_velocidad": [-g[2] for g in gf],
                    "solo_gap": [-g[3] for g in gf],
                }
                for nombre, vals in puntajes.items():
                    if candidatos[int(np.argmax(vals))] == correcto:
                        aciertos[nombre] += 1

                for saltar, contador in backoff.items():
                    p_i = pendiente_backoff(seg[i], "fin", 10, saltar)
                    vals = [
                        -abs(p_i - pendiente_backoff(seg[j], "ini", 10, saltar))
                        if np.isfinite(p_i) else 0.0
                        for j in candidatos
                    ]
                    contador[1] += 1
                    if candidatos[int(np.argmax(vals))] == correcto:
                        contador[0] += 1

    n_total = n_con_sucesor + n_sin_sucesor
    resultado = {
        "n_muestras_val": len(datos),
        "desambiguacion": {
            "n_grupos_ambiguos": n_grupos,
            "candidatos_medios": round(float(np.mean(tam_grupos)), 3) if tam_grupos else None,
            "base_aleatoria": round(float(np.mean([1 / t for t in tam_grupos])), 3) if tam_grupos else None,
            "exactitud_argmax": {k: round(v / max(n_grupos, 1), 3) for k, v in aciertos.items()},
        },
        "rechazo": {
            "n_segmentos_con_candidatos": n_total,
            "n_con_sucesor_verdadero": n_con_sucesor,
            "n_sin_sucesor_verdadero": n_sin_sucesor,
            "frac_sin_sucesor": round(n_sin_sucesor / max(n_total, 1), 3),
            "tasa_enlace_falso": {k: round(v / max(n_sin_sucesor, 1), 3) for k, v in falsos.items()},
        },
        "velocidad_backoff": {str(s): round(ok / max(n, 1), 3) for s, (ok, n) in backoff.items()},
    }
    (SALIDA / "diagnostico_rechazo.json").write_text(json.dumps(resultado, indent=2))

    d_ = resultado["desambiguacion"]
    r_ = resultado["rechazo"]
    print(f"\n=== (a) DESAMBIGUAR: {d_['n_grupos_ambiguos']} grupos ambiguos, "
          f"{d_['candidatos_medios']} candidatos de media, base aleatoria {d_['base_aleatoria']} ===")
    for k, v in sorted(d_["exactitud_argmax"].items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:.3f}")
    print(f"\n=== (b) RECHAZAR: {r_['n_sin_sucesor_verdadero']}/{r_['n_segmentos_con_candidatos']} "
          f"({r_['frac_sin_sucesor']:.1%}) segmentos NO tienen sucesor verdadero ===")
    for k, v in r_["tasa_enlace_falso"].items():
        print(f"  tasa de enlace falso, {k:16s} {v:.3f}")
    print("\n=== velocidad sola, alejando la ventana de la juncion ===")
    for s, v in resultado["velocidad_backoff"].items():
        print(f"  back-off {s} filas: {v:.3f}")
    print(f"\nGuardado en {SALIDA / 'diagnostico_rechazo.json'}")


if __name__ == "__main__":
    main()
