"""Extrae frames representativos del dataset para etiquetar en Roboflow / CVAT.

Filosofia del muestreo:

  1. Solo videos que tengan ROI (~142 de 168) — para que se pueda validar
     despues que las anotaciones caen dentro del axon.
  2. Mezcla balanceada de las 3 sesiones (N1, N2, N3) — para que el dataset
     etiquetado no sobre-represente una sola condicion de adquisicion.
  3. Frames espaciados en el tiempo dentro de cada video — para que no se
     etiqueten N frames casi identicos.
  4. Frames PRE-procesados con el MISMO pipeline que se usa en inferencia
     (sin fondo, sin ruido, enmascarado con ROI). Asi el modelo se entrena
     sobre la misma distribucion que despues vera, y el etiquetado es mas
     facil porque solo se ve la region del axon, no el ruido extracelular.

     Si querés etiquetar sin enmascarar (por ejemplo para construir un modelo
     que generalice mas alla del axon), usa --sin-roi.

Salida:

  data/etiquetado/
    frames/
      Movie_NNN_f000.png
      Movie_NNN_f042.png
      ...
    metadata.csv          (info de cada frame extraido)
    instrucciones.md      (proximos pasos para subir a Roboflow)

Uso:

    uv run scripts/extraer_frames_etiquetado.py \\
        --n-videos 8 \\
        --frames-por-video 4 \\
        --salida data/etiquetado
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent / "src"))
from axonal_tracking.ets_reader import encontrar_ets, leer_header_ets, leer_frame
from axonal_tracking import preprocesamiento as pp


VIDEOS_ROOT = Path("/Users/alejandrovalle/Desktop/Videos-Kymos-experimental data")


def listar_videos_con_roi() -> list[tuple[Path, Path, str]]:
    """Devuelve [(ruta_vsi, ruta_roi_primaria, sesion)] para todos los videos con ROI.

    Si un video tiene varios ROIs (axones multiples), toma el primero por orden.
    La sesion se infiere del path (N1/N2/N3/Ex).
    """
    resultados = []
    for vsi in sorted(VIDEOS_ROOT.rglob("Movie_*.vsi")):
        # Buscar cualquier .roi cuyo nombre empiece con el numero del video
        numero = vsi.stem.replace("Movie_", "")
        rois = sorted(vsi.parent.parent.parent.parent.rglob(f"{numero}*.roi"))
        if not rois:
            # buscar mas amplio
            rois = sorted(VIDEOS_ROOT.rglob(f"{numero}*.roi"))
        if not rois:
            continue

        # Inferir sesion del path
        s = str(vsi)
        if "/Ex/" in s:
            sesion = "Ex"
        elif "/N1-" in s:
            sesion = "N1"
        elif "/N2-" in s:
            sesion = "N2"
        elif "/N3-" in s:
            sesion = "N3"
        else:
            sesion = "?"

        resultados.append((vsi, rois[0], sesion))
    return resultados


def muestrear_videos(
    videos: list[tuple[Path, Path, str]],
    n: int,
    semilla: int,
) -> list[tuple[Path, Path, str]]:
    """Muestrea n videos balanceando por sesion (N1, N2, N3)."""
    rng = random.Random(semilla)
    por_sesion: dict[str, list] = {}
    for v in videos:
        por_sesion.setdefault(v[2], []).append(v)

    # Cuantos de cada sesion: distribuir uniformemente.
    sesiones = [s for s in ("N1", "N2", "N3") if s in por_sesion]
    if not sesiones:
        sesiones = list(por_sesion.keys())
    por_cada = max(1, n // len(sesiones))

    elegidos = []
    for s in sesiones:
        candidatos = por_sesion[s][:]
        rng.shuffle(candidatos)
        elegidos.extend(candidatos[:por_cada])

    # Completar hasta n con cualquier sesion
    if len(elegidos) < n:
        resto = [v for v in videos if v not in elegidos]
        rng.shuffle(resto)
        elegidos.extend(resto[: n - len(elegidos)])

    return elegidos[:n]


def frames_a_muestrear(n_frames_video: int, k: int) -> list[int]:
    """Devuelve k indices de frames espaciados uniformemente en [0, n-1]."""
    if k >= n_frames_video:
        return list(range(n_frames_video))
    return list(np.linspace(0, n_frames_video - 1, k).astype(int))


def procesar_y_guardar(
    vsi: Path,
    ruta_roi: Path,
    sesion: str,
    indices_frames: list[int],
    dir_salida: Path,
    aplicar_roi: bool = True,
) -> list[dict]:
    """Procesa los frames pedidos y los guarda como PNG. Devuelve metadata."""
    ets = encontrar_ets(vsi)
    info = leer_header_ets(ets)

    # Pre-calcular fondo temporal sobre TODO el video para mejor limpieza.
    # Cargamos solo los frames que vamos a usar + algunos extras para el fondo.
    n = info["n_frames"]
    indices_fondo = sorted(set(indices_frames + list(np.linspace(0, n - 1, min(15, n)).astype(int))))
    subvideo = np.stack([leer_frame(info, i) for i in indices_fondo])
    fondo_t = pp.calcular_fondo_temporal(subvideo, metodo="mediana")

    metadata = []
    for idx in indices_frames:
        frame = leer_frame(info, idx)
        # Mismo pipeline que en inferencia: con o sin ROI segun aplicar_roi.
        pasos = pp.pipeline_frame(
            frame,
            fondo_temporal=fondo_t,
            sigma_fondo=20,
            sigma_ruido=0.8,
            ruta_roi=ruta_roi if aplicar_roi else None,
        )
        rgb = pasos["rgb_uint8"]

        nombre = f"{vsi.stem}_f{idx:03d}.png"
        ruta_png = dir_salida / nombre
        Image.fromarray(rgb).save(ruta_png, optimize=True)

        metadata.append({
            "archivo": nombre,
            "video": vsi.name,
            "frame_idx": idx,
            "sesion": sesion,
            "roi_aplicada": "si" if aplicar_roi else "no",
            "roi_archivo": ruta_roi.name,
            "ancho": rgb.shape[1],
            "alto": rgb.shape[0],
        })
    return metadata


def escribir_instrucciones(dir_salida: Path, metadata: list[dict]) -> None:
    """Genera un README con los siguientes pasos para etiquetar."""
    contenido = f"""# Frames extraidos para etiquetado

Total: **{len(metadata)} frames** de **{len({m['video'] for m in metadata})} videos**
distintos, balanceados por sesion.

## Siguientes pasos

### Opcion A — Roboflow (recomendado, mas rapido)

1. Crear cuenta gratuita en https://roboflow.com
2. New Project → Object Detection → nombre `vesiculas-axonales`
3. Upload images → arrastrar la carpeta `frames/` entera
4. Iniciar etiquetado:
   - Una sola clase: `vesicle`
   - Dibujar bounding box ajustada alrededor de **cada vesicula visible** (incluidas las estelas en movimiento)
   - Ignorar: soma, agregados grandes, debris fuera del axon
   - Tip: usar el ROI del axon como guia visual mental
5. Una vez etiquetadas todas (~1-2 horas para 30 frames):
   - Generate → Train/Val Split 80/20
   - Export → COCO JSON Format → descargar
6. Bajar el ZIP y descomprimir en `data/etiquetado/coco/`

### Opcion B — CVAT (self-hosted, mas potente)

1. https://www.cvat.ai/ o instalar local con `docker compose`
2. Crear task → Object Detection → upload imagenes
3. Label `vesicle`, exportar como COCO 1.0

### Opcion C — LabelMe (local, simple)

```bash
pip install labelme
labelme frames/
```

Etiquetar a mano, exportar a COCO con `labelme2coco`.

## Que etiquetar como vesicula

- ✅ Puntos brillantes pequenios (~2-5 px) dentro del axon
- ✅ Estelas alargadas (vesiculas en movimiento, motion blur)
- ❌ Soma (cuerpo de la neurona, manchas grandes brillantes)
- ❌ Agregados grandes (>10 px) fuera del axon
- ❌ Ruido aislado (1 px brillante sin estructura)

## Despues de etiquetar

El dataset COCO sirve para fine-tunear:
- **Grounding DINO** (notebook a crear): el detector open-vocabulary aprende
  qué es una vesicula para vos.
- **YOLO11** (alternativa): mas rapido, mejor para datasets chicos.

Cantidad recomendada para llegar a calidad decente:
- 50 vesiculas: prueba de concepto
- 200 vesiculas: punto bueno costo/beneficio
- 1000+: produccion
"""
    (dir_salida / "instrucciones.md").write_text(contenido)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-videos", type=int, default=8,
                    help="Cuantos videos muestrear (default 8)")
    ap.add_argument("--frames-por-video", type=int, default=4,
                    help="Cuantos frames por video (default 4)")
    ap.add_argument("--salida", type=Path, default=Path("data/etiquetado"),
                    help="Carpeta de salida (default data/etiquetado)")
    ap.add_argument("--semilla", type=int, default=42, help="Random seed")
    ap.add_argument("--sin-roi", action="store_true",
                    help="No aplicar mascara del ROI (default: aplicar para coincidir con inferencia)")
    args = ap.parse_args()

    print(f"Buscando videos con ROI en {VIDEOS_ROOT}...")
    todos = listar_videos_con_roi()
    print(f"  {len(todos)} videos con ROI encontrados.")

    elegidos = muestrear_videos(todos, n=args.n_videos, semilla=args.semilla)
    print(f"\nVideos elegidos ({len(elegidos)}):")
    for vsi, _, sesion in elegidos:
        print(f"  [{sesion}] {vsi.name}")

    dir_frames = args.salida / "frames"
    dir_frames.mkdir(parents=True, exist_ok=True)

    todos_meta = []
    for vsi, roi, sesion in elegidos:
        ets = encontrar_ets(vsi)
        info = leer_header_ets(ets)
        idxs = frames_a_muestrear(info["n_frames"], args.frames_por_video)
        print(f"\nProcesando {vsi.name}: frames {idxs}")
        meta = procesar_y_guardar(vsi, roi, sesion, idxs, dir_frames,
                                  aplicar_roi=not args.sin_roi)
        todos_meta.extend(meta)

    # Escribir metadata CSV
    with open(args.salida / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=todos_meta[0].keys())
        writer.writeheader()
        writer.writerows(todos_meta)

    escribir_instrucciones(args.salida, todos_meta)

    print(f"\n[OK] {len(todos_meta)} frames guardados en {dir_frames}")
    print(f"     Metadata: {args.salida / 'metadata.csv'}")
    print(f"     Instrucciones para etiquetar: {args.salida / 'instrucciones.md'}")


if __name__ == "__main__":
    main()
