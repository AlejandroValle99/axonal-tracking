"""Compara las curvas `frac_id_switch` vs `fragmentos_por_gt` de las filas 2/3/3b/3c
sobre VAL, interpolando a fragmentacion igualada.

Por que hace falta: cada fila reporta UN punto de operacion, y comparar puntos sueltos
confunde "mejor modelo" con "punto distinto de la misma curva" -- el error que NB11 SS5.2
ya documenta para v1-vs-v2. Las cuatro filas barren su parametro de operacion sobre el
MISMO split de val (400 muestras), asi que sus curvas son directamente comparables: a
igual `fragmentos_por_gt`, .cual da menor `frac_id_switch`?

v3 barre `margen` (comparacion contra el dustbin) en vez de un umbral global, pero eso es
justamente lo que se quiere comparar: la pregunta no es "cual umbral" sino "que estructura
de salida traza mejor curva".

Uso: uv run python scripts/comparar_curvas.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
ASOC = RAIZ / "results" / "asociacion"

FUENTES = [
    ("2: costo clasico", "costo_clasico_resumen.json", "barrido_umbral_val"),
    ("3: atencion v1", "atencion_v1_resumen.json", "barrido_umbral_val"),
    ("3b: atencion v2", "atencion_v2_resumen.json", "barrido_umbral_val"),
    ("3c: atencion v3 (dustbin)", "atencion_v3_resumen.json", "barrido_margen_val"),
    ("3d: v3 + estaticos->dustbin", "atencion_v3_resumen_est.json", "barrido_margen_val"),
]
REJILLA = [1.282, 1.4, 1.5, 1.6, 1.8, 2.0, 2.4]  # 1.282 = fragmentacion de DecNet


def curva(filas: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Frontera inferior: a cada nivel de fragmentacion, el menor `frac_id_switch`
    alcanzado (sobre decodificadores y puntos de operacion), ordenada por fragmentacion."""
    pts = sorted(
        (f["fragmentos_por_gt"], f["frac_id_switch"]) for f in filas
        if f.get("fragmentos_por_gt") is not None and f.get("frac_id_switch") is not None
    )
    x, y, mejor = [], [], float("inf")
    for frag, sw in pts:
        mejor = min(mejor, sw)
        x.append(frag)
        y.append(mejor)
    return np.array(x), np.array(y)


def main() -> None:
    curvas, ausentes = {}, []
    for nombre, archivo, clave in FUENTES:
        p = ASOC / archivo
        if not p.exists():
            ausentes.append(nombre)
            continue
        datos = json.loads(p.read_text())
        if clave not in datos:
            ausentes.append(f"{nombre} (falta '{clave}')")
            continue
        curvas[nombre] = curva(datos[clave])

    print("frac_id_switch interpolado a fragmentacion igualada (val, 400 muestras)")
    print("menor es mejor; '-' = la curva no alcanza ese nivel de fragmentacion\n")
    cab = "fragmentos/GT ->".ljust(30) + "".join(f"{g:>8}" for g in REJILLA)
    print(cab)
    print("-" * len(cab))
    for nombre, (x, y) in curvas.items():
        celdas = []
        for g in REJILLA:
            if len(x) == 0 or g < x.min() or g > x.max():
                celdas.append(f"{'-':>8}")
            else:
                celdas.append(f"{float(np.interp(g, x, y)):>8.3f}")
        print(nombre.ljust(30) + "".join(celdas))
    print("-" * len(cab))
    print("DecNet (test)".ljust(30) + "".join(
        f"{0.103:>8.3f}" if abs(g - 1.282) < 1e-9 else f"{'':>8}" for g in REJILLA)
        + "   <- 0.103 a frag 1.282")
    print("oraculo (test, techo)".ljust(30) + f"{0.031:>8.3f}" + "   <- a frag 1.000\n")
    for nombre, (x, y) in curvas.items():
        if len(x):
            print(f"  {nombre}: fragmentacion cubierta {x.min():.3f} - {x.max():.3f}, "
                  f"mejor frac_id_switch {y.min():.3f}")
    if ausentes:
        print(f"\n(ausentes: {', '.join(ausentes)})")


if __name__ == "__main__":
    main()
