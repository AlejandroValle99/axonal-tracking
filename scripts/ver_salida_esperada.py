"""Dibuja la SALIDA ESPERADA del modelo por-pixel sobre un kymografo sintetico:
lo que un modelo perfecto tendria que emitir en cada head, derivado del ground
truth exacto de `positions.csv`.

Cuatro paneles:

  (a) ENTRADA    -- el kymografo a resolucion nativa, sin resize.
  (b) TRACKNESS  -- 3 clases: fondo / estatica / movil. La distincion no es
      cosmetica: sobre 20 muestras de train, el 32.7% de los frames tiene una
      particula movil a menos de 2 px de una estatica (ancho de traza ~1.6 px,
      mediana), asi que estatica-vs-movil hay que resolverlo POR PIXEL, no
      filtrando despues. Ademas el 78% de las particulas son estaticas y todas
      las metricas de este repo cuentan solo moviles.
  (c) EMBEDDING  -- una instancia por particula MOVIL, como tendria que quedar
      tras clusterizar los embeddings. La identidad se conserva a traves de los
      cruces: es lo unico que este panel esta probando.
  (d) SALIDA     -- las polilineas subpixel (t, x) por instancia, que es lo que
      entra a `evaluacion.evaluar_trayectorias_polilineas` y a la conversion a
      velocidad de `parametros.py`.

Todo se deriva desde `positions.csv` con las MISMAS funciones que exportan las
etiquetas (`etiquetas_deteccion.mascara_traza` / `ids_que_se_mueven`), incluido
el espejo L-R de la columna: el dibujo es el target de entrenamiento, no una
reconstruccion aparte. Si hubiera un bug en las etiquetas, aca se veria.

Salida: results/kymorope/salida_esperada_{split}_{nombre}.png

Uso:

  # la muestra por defecto (test/sample_00099: 5 moviles, 4 de ellos se cruzan)
  uv run python scripts/ver_salida_esperada.py

  # una muestra cualquiera, por nombre o por indice
  uv run python scripts/ver_salida_esperada.py test sample_00098
  uv run python scripts/ver_salida_esperada.py train 42
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))

from axonal_tracking.etiquetas_deteccion import (
    ids_que_se_mueven,
    mascara_traza,
)
from axonal_tracking.parametros import PIXEL_SIZE_UM

DATASETS = RAIZ / "datasets"
SALIDA = RAIZ / "results" / "kymorope"
POR_DEFECTO = ("test", "sample_00099")
MIN_DESPLAZAMIENTO_PX = 4.0  # mismo umbral de "movil" que usa el resto del repo

FONDO = "#0b0b10"
COLOR_ESTATICA = "#4a5568"
COLOR_MOVIL = "#f6c945"
# ciclo de colores por instancia movil (se repite si hay mas moviles que colores)
CICLO = ["#e4572e", "#4cb5f5", "#76c893", "#f6c945", "#c77dff", "#ff8fab", "#ffd166"]


def cargar(dir_muestra: Path) -> tuple[np.ndarray, pd.DataFrame, list[int], list[int]]:
    """Kymografo, positions, ids moviles y ids estaticos de una muestra."""
    kymo = tifffile.imread(dir_muestra / "kymograph.tif").astype(np.float32)
    positions = pd.read_csv(dir_muestra / "positions.csv")
    moviles = sorted(ids_que_se_mueven(positions, PIXEL_SIZE_UM, MIN_DESPLAZAMIENTO_PX))
    estaticas = [p for p in sorted(positions["particle_id"].unique()) if p not in moviles]
    return kymo, positions, moviles, estaticas


def mapas_esperados(
    positions: pd.DataFrame, shape: tuple[int, int], moviles: list[int], estaticas: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """`(trackness, instancias)`.

    `trackness` es (T, L) con 0=fondo / 1=estatica / 2=movil -- el argmax del head
    de 3 clases. Las moviles se pintan DESPUES de las estaticas a proposito: donde
    una movil pasa por encima de una estatica el pixel es de la movil, que es el
    caso que decide la metrica (y el 32.7% de los frames tiene uno).

    `instancias` es (T, L) con 0=fondo y n=1..len(moviles) -- el resultado de
    clusterizar el head de embedding, supervisado solo sobre pixeles moviles.
    """
    trackness = np.zeros(shape, np.uint8)
    for pid in estaticas:
        trackness[mascara_traza(positions, pid, PIXEL_SIZE_UM, shape)] = 1
    for pid in moviles:
        trackness[mascara_traza(positions, pid, PIXEL_SIZE_UM, shape)] = 2

    instancias = np.zeros(shape, np.int16)
    for n, pid in enumerate(moviles, start=1):
        instancias[mascara_traza(positions, pid, PIXEL_SIZE_UM, shape)] = n
    return trackness, instancias


def polilineas(positions: pd.DataFrame, moviles: list[int], ancho_px: int) -> pd.DataFrame:
    """Salida decodificada: una fila por (instancia, frame), en el mismo sistema de
    columnas que las mascaras (espejo L-R, ver `mascara_traza`)."""
    filas = []
    for n, pid in enumerate(moviles, start=1):
        sub = positions[positions["particle_id"] == pid].sort_values("frame")
        for frame, pos_um in zip(sub["frame"], sub["pos_um"]):
            filas.append(
                {
                    "instancia": n,
                    "t_frame": int(frame),
                    "x_px": (ancho_px - 1) - pos_um / PIXEL_SIZE_UM,
                    "pos_um": pos_um,
                }
            )
    return pd.DataFrame(filas)


def figura(kymo: np.ndarray, trackness: np.ndarray, instancias: np.ndarray,
           polis: pd.DataFrame, titulo: str) -> plt.Figure:
    """Los cuatro paneles. El contraste es el mismo p50-p99.8 que usa el resto del
    repo (`preprocesamiento.frame_a_rgb_uint8`)."""
    alto, ancho = kymo.shape
    lo, hi = np.percentile(kymo, [50, 99.8])
    n_inst = int(instancias.max())
    colores = [CICLO[i % len(CICLO)] for i in range(n_inst)]

    fig, axs = plt.subplots(4, 1, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(titulo, fontsize=11)

    axs[0].imshow(kymo, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    axs[0].set_title(f"(a) ENTRADA -- kymografo T={alto}, L={ancho}, resolucion nativa, sin resize")

    axs[1].imshow(trackness, cmap=ListedColormap([FONDO, COLOR_ESTATICA, COLOR_MOVIL]),
                  vmin=0, vmax=2, aspect="auto")
    axs[1].set_title("(b) head TRACKNESS -- 3 clases: fondo / estatica (gris) / movil (amarillo)")

    axs[2].imshow(instancias, cmap=ListedColormap([FONDO, *colores]), vmin=0, vmax=n_inst, aspect="auto")
    axs[2].set_title(f"(c) head EMBEDDING tras clustering -- {n_inst} instancias moviles, "
                     "identidad conservada a traves de los cruces")

    axs[3].imshow(kymo, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    for n, color in enumerate(colores, start=1):
        sub = polis[polis["instancia"] == n]
        axs[3].plot(sub["x_px"], sub["t_frame"], lw=1.1, color=color)
    axs[3].set_title("(d) SALIDA -- polilineas subpixel (t, x) por instancia "
                     "-> velocidad via parametros.py")

    for ax in axs:
        ax.set_ylabel("t (frame)")
        ax.set_xlim(0, ancho)
        ax.set_ylim(alto, 0)
    axs[3].set_xlabel("x (px a lo largo del axon)")
    return fig


def _resolver_nombre(ident: str) -> str:
    """'42' -> 'sample_00042'; 'sample_00042' -> tal cual."""
    return f"sample_{int(ident):05d}" if ident.isdigit() else ident


def main() -> None:
    args = sys.argv[1:]
    split, nombre = POR_DEFECTO if len(args) < 2 else (args[0], _resolver_nombre(args[1]))
    dir_muestra = DATASETS / split / nombre
    if not dir_muestra.exists():
        sys.exit(f"No existe {dir_muestra}. Genera el dataset sintetico primero "
                 f"(scripts/generar_dataset_sintetico.py, ver docs/regenerar-dataset-sintetico.md).")

    kymo, positions, moviles, estaticas = cargar(dir_muestra)
    if not moviles:
        sys.exit(f"{split}/{nombre} no tiene particulas moviles -- elegi otra muestra.")

    trackness, instancias = mapas_esperados(positions, kymo.shape, moviles, estaticas)
    polis = polilineas(positions, moviles, kymo.shape[1])

    SALIDA.mkdir(parents=True, exist_ok=True)
    out = SALIDA / f"salida_esperada_{split}_{nombre}.png"
    fig = figura(kymo, trackness, instancias, polis,
                 f"Salida esperada del modelo por-pixel -- {split}/{nombre}")
    fig.savefig(out, dpi=115)
    plt.close(fig)

    frac_movil = (trackness == 2).mean()
    frac_estatica = (trackness == 1).mean()
    print(f"{split}/{nombre}: T={kymo.shape[0]} L={kymo.shape[1]} | "
          f"{len(moviles)} moviles, {len(estaticas)} estaticas")
    print(f"  pixeles: movil {frac_movil:.3f} | estatica {frac_estatica:.3f} | "
          f"fondo {1 - frac_movil - frac_estatica:.3f}")
    print(f"  heads esperados: trackness {kymo.shape + (3,)} | embedding {kymo.shape + (8,)} | "
          f"orientacion {kymo.shape + (2,)}")
    print("\n  salida decodificada (instancia 1, primeras filas):")
    print(polis[polis["instancia"] == 1].head(4).round(3).to_string(index=False))
    print(f"\n-> {out}")
    if sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
