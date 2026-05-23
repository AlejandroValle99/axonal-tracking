"""Lector de archivos VSI/ETS de Olympus IX83 sin dependencias de Java.

Formato documentado en docs/guia-formato-vsi-ets.md.
"""

import struct
from pathlib import Path

import numpy as np


DATA_OFFSET = 0x120  # offset fijo donde empiezan los píxeles


def encontrar_ets(ruta_vsi: Path) -> Path:
    """Dado un .vsi, devuelve la ruta al .ets correspondiente."""
    ets = ruta_vsi.parent / f'_{ruta_vsi.stem}_' / 'stack1' / 'frame_t_0.ets'
    if not ets.exists():
        raise FileNotFoundError(f'No se encontró el ETS en: {ets}')
    return ets


def leer_header_ets(ruta_ets: Path) -> dict:
    """Lee la cabecera del archivo ETS y devuelve dimensiones + número de frames."""
    with open(ruta_ets, 'rb') as f:
        if f.read(4) != b'SIS\x00':
            raise ValueError('No es un archivo SIS/ETS')
        f.seek(0x40)
        if f.read(4) != b'ETS\x00':
            raise ValueError('Bloque ETS no encontrado')
        f.read(4)   # version
        f.read(4)   # pix_type (siempre uint16 en este dataset)
        f.read(4)   # n_canales
        f.read(8)   # campos desconocidos
        f.read(4)   # tile_size
        width  = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        fsize  = f.seek(0, 2)

    frame_bytes = width * height * 2
    n_frames    = (fsize - DATA_OFFSET) // frame_bytes
    return {'ruta': ruta_ets, 'ancho': width, 'alto': height,
            'n_frames': n_frames, 'frame_bytes': frame_bytes}


def leer_frame(info: dict, frame_idx: int) -> np.ndarray:
    """Lee un solo frame uint16 como array 2D (alto, ancho), sin cargar el video entero."""
    if frame_idx < 0 or frame_idx >= info['n_frames']:
        raise IndexError(f'frame {frame_idx} fuera de rango (0..{info["n_frames"] - 1})')
    offset = DATA_OFFSET + frame_idx * info['frame_bytes']
    with open(info['ruta'], 'rb') as f:
        f.seek(offset)
        raw = f.read(info['frame_bytes'])
    return np.frombuffer(raw, dtype=np.uint16).reshape(info['alto'], info['ancho'])


def leer_video(info: dict) -> np.ndarray:
    """Carga el video completo como array (T, H, W) uint16. Atención: ~470 MB por video."""
    n, H, W = info['n_frames'], info['alto'], info['ancho']
    video = np.empty((n, H, W), dtype=np.uint16)
    with open(info['ruta'], 'rb') as f:
        f.seek(DATA_OFFSET)
        for i in range(n):
            raw = f.read(info['frame_bytes'])
            video[i] = np.frombuffer(raw, dtype=np.uint16).reshape(H, W)
    return video
