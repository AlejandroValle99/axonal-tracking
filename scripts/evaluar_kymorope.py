"""Decodifica KymoRoPE sobre un split y lo evalua a nivel trayectoria con el MISMO harness
que KymoButler (`ev.evaluar_trayectorias_polilineas`, solo moviles, >= 8 px).

Un forward por muestra (float32, modo eval) y despues se decodifica cada fila de la grilla
desde los mismos mapas. La grilla es una escalera diagnostica, para que cada cifra se pueda
atribuir a una pieza:

    D0  trackness GT     + identidad GT     techo del decode + metrica (valida el decode)
    D1  trackness GT     + sin embedding    lo que da la geometria sola
    D2  trackness GT     + embedding modelo el embedding aislado
    M1  trackness modelo + sin embedding    trackness del modelo + geometria
    M2  trackness modelo + embedding modelo el sistema real, barrido de ancho de banda y
                                            peso de orientacion

El punto de operacion se elige SOLO en val: menor `frac_id_switch + frac_gt_fragmentado`
entre las filas M2 con `track_f1` a <= 0.01 de la mejor (`punto_de_operacion`). Test queda
para una unica corrida al final.

Salida: results/kymorope/decode/{split}/ -- resumen.json, trayectorias_{fila}.csv,
polilineas_{fila}.pkl (polilineas por muestra, para superponer en NB13). Con otro
`--checkpoint` que el de 800 muestras, la carpeta lleva el nombre del modelo
(`decode/val_finetune_5k/`), asi no pisa los resultados del anterior.

Uso: uv run python scripts/evaluar_kymorope.py [--split val] [--checkpoint ruta.pt] [--limite N]
     uv run python scripts/evaluar_kymorope.py --checkpoint results/kymorope/checkpoints/finetune_5k/checkpoint_final.pt
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))

from axonal_tracking import datos_pixel as dp
from axonal_tracking import decodificacion as dc
from axonal_tracking import evaluacion as ev
from axonal_tracking import kymorope as kr
from axonal_tracking.etiquetas_deteccion import MIN_DESPLAZAMIENTO_PX_MOVIL

# el backup con fecha, no `checkpoint_final.pt`: ese lo pisa la proxima corrida de `entrenar()`
CHECKPOINT = RAIZ / "results" / "kymorope" / "checkpoints" / "kymorope_40ep_2026-09-24.pt"
TOLERANCIA_F1 = 0.01  # regla del punto de operacion (ver docstring)


def grilla() -> dict[str, tuple[str, dc.ParametrosDecode]]:
    """nombre de fila -> (fuente de los mapas, parametros del decode).

    Base de todas las filas (punto de operacion elegido en val el 2026-09-29): largo minimo
    30 filas y suavizado Savitzky-Golay de 7 filas. Las ablaciones de abajo apagan de a una
    las piezas del decode que se agregaron despues de la primera corrida.

    Historia (lo que ya se midio y se saco de la grilla): h=0.5 sobresegmentaba (3.0-3.6
    fragmentos por GT); el enlace solo geometrico disparaba el id-switch a 0.17-0.19 (el
    enlace va siempre con identidad); el `margen_enlace` de 0.5 y los modos de centro
    `centro_desde_mascara` / pixeles del cluster daban identico; el centro por pico de
    intensidad fue negativo (ver `decodificacion.ParametrosDecode.centro_por_pico`)."""
    base = dc.ParametrosDecode(
        ancho_banda=1.5, min_filas=30, suavizado="sg", ventana_suavizado=7
    )
    enlace = replace(base, max_hueco_enlace=40)
    filas = {
        "D0_gt_oraculo": ("gt_oraculo", base),
        "D0_gt_oraculo_enlace40": ("gt_oraculo", enlace),
        "D1_gt_sin_embedding": ("gt", replace(base, usar_embedding=False)),
        "D2_gt_embedding": ("gt_embedding", base),
        "M1_sin_embedding": ("modelo", replace(base, usar_embedding=False)),
    }
    for h in (1.0, 1.5, 2.0):
        filas[f"M2_h{h:g}_w0"] = ("modelo", replace(base, ancho_banda=h))
        # enlace de fragmentos entre componentes: las pausas que el head de trackness
        # marca estaticas parten la traza en componentes distintas
        filas[f"M2_h{h:g}_w0_enlace40"] = ("modelo", replace(enlace, ancho_banda=h))
    filas["M2_h1.5_w1"] = ("modelo", replace(base, peso_orientacion=1.0))

    # Ablaciones del punto de operacion (M2_h1.5_w0_enlace40), de a una pieza:
    # - `_sin_suavizado`: sin Savitzky-Golay. Offline sobre val, SG-7 bajo el error de
    #   posicion del modelo afinado de 0.041 a 0.038 um sin tocar F1/id-switch/fragmentos
    #   (a KymoButler le hace lo mismo: 0.032 -> 0.029; la brecha no se cierra);
    # - `_min10` / `_min20`: largo minimo de antes (= `min_frames` de KymoButler) e
    #   intermedio. Con 30, 1.283 -> 1.175 trayectorias por particula y la posicion baja,
    #   porque el harness promedia POR trayectoria y las astillas pesaban como una traza;
    # - `_enlace_laxo`: sin el enlace estricto en zonas densas;
    # - `_antes`: todo apagado; tiene que reproducir la corrida del modelo afinado del
    #   2026-09-29 (0.981 / 0.100 / 1.267 / 0.048 um).
    laxo = {"margen_enlace": 0.0, "max_dist_emb_denso": enlace.max_dist_emb_enlace}
    filas["M2_h1.5_w0_enlace40_sin_suavizado"] = ("modelo", replace(enlace, suavizado=None))
    for minimo in (10, 20):
        filas[f"M2_h1.5_w0_enlace40_min{minimo}"] = ("modelo", replace(enlace, min_filas=minimo))
    filas["M2_h1.5_w0_enlace40_enlace_laxo"] = ("modelo", replace(enlace, **laxo))
    filas["M2_h1.5_w0_enlace40_antes"] = (
        "modelo",
        replace(enlace, min_filas=10, suavizado=None, centro_desde_mascara=False, **laxo),
    )
    return filas


def fuentes_de_mapas(salida: kr.SalidaKymoRoPE, muestra: dp.MuestraPixel) -> dict[str, dc.MapasDecode]:
    """Los cuatro juegos de mapas de la escalera, desde un unico forward. Los del GT salen
    de los mismos targets con los que se entreno (mascara dilatada), asi que D0-D2 ven la
    misma geometria de traza que el modelo aprendio a producir."""
    modelo = dc.mapas_desde_salida(salida)
    gt_movil = (muestra.trackness.numpy() == dp.CLASE_MOVIL).astype(np.float32)
    return {
        "modelo": modelo,
        "gt": dc.MapasDecode(p_movil=gt_movil),
        "gt_oraculo": dc.MapasDecode(p_movil=gt_movil, ids_oraculo=muestra.instancias.numpy()),
        "gt_embedding": dc.MapasDecode(
            p_movil=gt_movil, embedding=modelo.embedding, orientacion=modelo.orientacion
        ),
    }


def punto_de_operacion(resumenes: dict[str, dict]) -> str | None:
    """Entre las filas M2 con `track_f1` a <= TOLERANCIA_F1 de la mejor M2, la de menor
    error de identidad por traza: `frac_id_switch` (polilineas que mezclan dos
    particulas) + `frac_gt_fragmentado` (particulas partidas en varias polilineas). Las
    dos cuentan: ordenar solo por id-switch elegia filas sin enlace, con 25% mas de
    fragmentos (20 muestras de val). Empate por menor error de posicion. Se elige en val;
    aplicarlo a test sin re-elegir es lo que mantiene limpio el gate."""
    m2 = {k: r for k, r in resumenes.items() if k.startswith("M2_") and np.isfinite(r["track_f1"])}
    if not m2:
        return None
    mejor_f1 = max(r["track_f1"] for r in m2.values())
    aptas = {k: r for k, r in m2.items() if r["track_f1"] >= mejor_f1 - TOLERANCIA_F1}
    return min(aptas, key=lambda k: (
        aptas[k]["frac_id_switch"] + aptas[k]["frac_gt_fragmentado"], aptas[k]["err_pos_um_medio"]
    ))


def etiqueta_modelo(checkpoint: Path) -> str:
    """"" para el modelo de 800 muestras (el default, cuyos resultados ya estan en
    `decode/val`), y si no el nombre de la corrida (`.../finetune_5k/checkpoint_final.pt`
    -> "finetune_5k") o del archivo. Va en el nombre de la carpeta de salida: evaluar un
    modelo nuevo no pisa los resultados de otro."""
    if checkpoint.resolve() == CHECKPOINT.resolve():
        return ""
    return checkpoint.parent.name if checkpoint.name == "checkpoint_final.pt" else checkpoint.stem


def main(split: str, checkpoint: Path, limite: int | None, prefijos: list[str] | None = None) -> None:
    modelo = etiqueta_modelo(checkpoint)
    sufijo = f"_{modelo}" if modelo else ""
    sufijo += f"_lim{limite}" if limite else ""
    if prefijos:  # subconjunto de filas: carpeta aparte, nunca pisa la corrida completa
        sufijo += "_" + "-".join(prefijos)
    salida_dir = RAIZ / "results" / "kymorope" / "decode" / f"{split}{sufijo}"
    salida_dir.mkdir(parents=True, exist_ok=True)
    dev = kr.dispositivo_preferido()

    ck = torch.load(checkpoint, map_location=dev, weights_only=False)
    modelo = kr.KymoRoPE().to(dev).eval()
    modelo.load_state_dict(ck["model_state_dict"])
    cache = RAIZ / "results" / "kymorope" / "cache" / split
    # la normalizacion de entrada es parte del modelo: sale del checkpoint, no es un argumento
    modo = dp.modo_de_checkpoint(ck)
    ds = dp.DatasetKymografos(
        RAIZ / "datasets" / split, limite=limite, cache_dir=cache if cache.exists() else None,
        modo_normalizacion=modo,
    )
    filas = grilla()
    if prefijos:
        filas = {k: v for k, v in filas.items() if any(k.startswith(p) for p in prefijos)}
    print(f"{len(ds)} muestras de {split} · {len(filas)} filas de decode · {dev} · {checkpoint.name}"
          f" · normalizacion {modo}", flush=True)

    escenas: dict[str, dict] = {nombre: {} for nombre in filas}
    segundos = defaultdict(float)
    t0 = time.time()
    for i in range(len(ds)):
        muestra = ds[i]
        escena = ev.cargar_escena_sintetica(ds.dirs[i])  # kimografo CRUDO para el centroide
        with torch.no_grad():
            salida = modelo([muestra])[0]
        fuentes = fuentes_de_mapas(salida, muestra)
        for nombre, (fuente, params) in filas.items():
            t = time.time()
            polis = dc.decodificar(fuentes[fuente], escena["kymo"], params)
            segundos[nombre] += time.time() - t
            escenas[nombre][escena["nombre"]] = {**escena, "polilineas": polis}
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(ds)}  {el / 60:.1f} min  "
                  f"(~{el / (i + 1) * (len(ds) - i - 1) / 60:.1f} min restantes)", flush=True)

    resumenes = {}
    for nombre, (fuente, params) in filas.items():
        tr, res = ev.evaluar_trayectorias_polilineas(
            escenas[nombre], min_desplazamiento_px=MIN_DESPLAZAMIENTO_PX_MOVIL
        )
        resumenes[nombre] = {
            **res, **ev.resumen_fragmentacion(tr),
            "seg_decode_por_muestra": round(segundos[nombre] / len(ds), 3),
            "fuente": fuente, "params": asdict(params),
        }
        tr.to_csv(salida_dir / f"trayectorias_{nombre}.csv", index=False)
        with open(salida_dir / f"polilineas_{nombre}.pkl", "wb") as f:
            pickle.dump({n: e["polilineas"] for n, e in escenas[nombre].items()}, f)

    elegido = punto_de_operacion(resumenes)
    (salida_dir / "resumen.json").write_text(json.dumps({
        "split": split, "n_muestras": len(ds), "checkpoint": str(checkpoint),
        "modo_normalizacion": modo,
        "min_desplazamiento_px": MIN_DESPLAZAMIENTO_PX_MOVIL, "thr_px_track": ev.THR_PX_TRACK,
        "punto_de_operacion": elegido,
        "regla": f"min frac_id_switch + frac_gt_fragmentado con track_f1 >= mejor M2 - {TOLERANCIA_F1}",
        "filas": resumenes,
    }, indent=2))

    columnas = ["track_f1", "track_precision", "track_recall", "frac_id_switch",
                "fragmentos_por_gt", "err_pos_um_medio", "err_vel_um_s_mediana", "seg_decode_por_muestra"]
    tabla = pd.DataFrame(resumenes).T[columnas]
    print(f"\n{tabla.to_string()}\n\npunto de operacion ({split}): {elegido}")
    print(f"total {(time.time() - t0) / 60:.1f} min · guardado en {salida_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="val", help="val (default) para elegir; test una sola vez al final")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--limite", type=int, default=None,
                    help="solo las primeras N muestras (prueba; carpeta con sufijo _limN)")
    ap.add_argument("--filas", default=None,
                    help="prefijos de filas separados por coma, p.ej. D0,M1 (carpeta con sufijo)")
    args = ap.parse_args()
    main(args.split, args.checkpoint, args.limite, args.filas.split(",") if args.filas else None)
