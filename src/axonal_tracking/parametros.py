"""Parametros experimentales del setup IBCN-FMED-UBA.

Fuente: `Experimental Data.docx` del lab, documentado en
docs/parametros-experimentales.md.

Estas constantes son la fuente unica de verdad para conversion px->um y
frame->segundos en el pipeline. Cualquier calculo de velocidad debe usarlas.
"""

from pathlib import Path


# ── Espacial (comun a todas las sesiones) ────────────────────────────────────

PIXEL_SIZE_UM = 0.107  # micrometros por pixel — Olympus IX83-DSU + ORCA-Flash4.0


# ── Temporal (varia por sesion) ──────────────────────────────────────────────

# Segundos entre frames consecutivos. None = sesion no documentada.
SEGUNDOS_POR_FRAME: dict[str, float | None] = {
    "N1": 0.304,
    "N2": 0.207,   # exposicion 160 ms es valor estimado por el lab
    "N3": 0.193,
    "Ex": None,    # Movie_674 — no documentado
    "i3": None,    # sesion por iniciarse
}


# Rangos de numero de video por sesion, observados en el dataset.
# Ver docs/dataset-findings.md para el detalle de la composicion.
RANGOS_SESION: dict[str, tuple[int, int]] = {
    "N1": (264, 333),
    "N2": (348, 406),
    "N3": (467, 537),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def sesion_por_path(ruta_vsi: Path) -> str | None:
    """Identifica la sesion (N1/N2/N3/Ex) leyendo el path del video.

    Es el metodo preferido: el path lleva la marca de sesion explicitamente
    (`/N1-11 and 12-09-25/`, `/N2-231025/`, `/N3-160126/`, `/Ex/`).
    """
    s = str(ruta_vsi)
    if "/Ex/" in s:
        return "Ex"
    if "/N1-" in s or "/N1/" in s:
        return "N1"
    if "/N2-" in s or "/N2/" in s:
        return "N2"
    if "/N3-" in s or "/N3/" in s:
        return "N3"
    if "/i3-" in s:
        return "i3"
    return None


def sesion_por_numero(numero_vsi: int) -> str | None:
    """Identifica la sesion a partir del numero del video (ej: 674 -> Ex).

    Fallback util cuando no se tiene el path completo. Si el numero no cae
    en ningun rango conocido, devuelve None.
    """
    for sesion, (lo, hi) in RANGOS_SESION.items():
        if lo <= numero_vsi <= hi:
            return sesion
    if numero_vsi == 674:
        return "Ex"
    return None


def segundos_por_frame(ruta_o_numero) -> float | None:
    """Devuelve el framerate (s/frame) del video. Acepta Path o int.

    Devuelve None si la sesion no esta documentada (Ex, i3) — en ese caso
    el pipeline debe abortar el calculo de velocidad o pedir el parametro
    al usuario.
    """
    if isinstance(ruta_o_numero, (int,)):
        sesion = sesion_por_numero(ruta_o_numero)
    else:
        sesion = sesion_por_path(Path(ruta_o_numero))
    if sesion is None:
        return None
    return SEGUNDOS_POR_FRAME.get(sesion)
