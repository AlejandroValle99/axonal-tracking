"""
Ver un video de microscopía Olympus (VSI/ETS) en napari.

Uso:
    python scripts/ver_video.py                          # abre el video de ejemplo
    python scripts/ver_video.py ruta/al/Movie_491.vsi   # abre el video indicado
"""

import sys
from pathlib import Path
import numpy as np
import napari

from axonal_tracking.ets_reader import encontrar_ets, leer_header_ets, leer_video


# ── Main ─────────────────────────────────────────────────────────────────────

DATOS = Path.home() / 'Desktop' / 'Videos-Kymos-experimental data'
EJEMPLO = DATOS / 'Ex' / 'Movie_674.vsi'

if len(sys.argv) > 1:
    vsi = Path(sys.argv[1])
else:
    vsi = EJEMPLO

print(f'Cargando: {vsi.name}')
ets  = encontrar_ets(vsi)
info = leer_header_ets(ets)
print(f'  {info["ancho"]}×{info["alto"]} px  |  {info["n_frames"]} frames  |  uint16')
print('  Leyendo frames...', end=' ', flush=True)
video = leer_video(info)
print('listo.')

viewer = napari.Viewer(title=vsi.stem)
viewer.add_image(
    video,
    name=vsi.stem,
    colormap='green',       # canal verde de fluorescencia
    contrast_limits=[
        int(np.percentile(video, 1)),
        int(np.percentile(video, 99.5)),
    ],
)
print()
print('Controles napari:')
print('  ← → o slider inferior : cambiar frame')
print('  scroll : zoom')
print('  click+drag : mover imagen')
print('  Ctrl+Shift+C : ajustar contraste')

napari.run()
