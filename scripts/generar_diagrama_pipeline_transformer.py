"""Generates a SEQUENCE diagram (UML-like) of the proposed transformer
pipeline for end-to-end tracking on kymographs (query-based instance
segmentation, no KymoButler components). Saves the result to diagrams/.

Color = family, same fixed order as pipeline_kymobutler.png (blue, aqua,
yellow, green, violet), so the two diagrams read as one system: Encoder and
Decoder share aqua (both are the transformer model), Extractor gets yellow
(the classical/bookkeeping role KymoButler's Tracker plays, but far thinner).
"""

from pathlib import Path

from _diagrama_secuencia_utils import SeqDiagram

OUT_DIR = Path(__file__).resolve().parent.parent / "diagrams"
OUT_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"

PARTICIPANTS = [
    ("kymo", "Kymograph\n(T x L)", BLUE),
    ("pre", "Preproc\nnormalization", BLUE),
    ("enc", "Encoder\nself-attention", AQUA),
    ("dec", "Decoder\nN object queries", AQUA),
    ("ext", "Track\nExtractor", YELLOW),
    ("kin", "Kinetics\npostproc", GREEN),
    ("out", "Output\nCSV/JSON + PNG", VIOLET),
]

d = SeqDiagram(PARTICIPANTS)

d.call("kymo", "pre", "raw kymograph (T x L)")
d.self_call("pre", "per-row normalization", "divide by temporal mean -> removes bleach decay (same rationale as KymoButler, reimplemented)")
d.call("pre", "enc", "normalized kymograph")

d.self_call("enc", "patchify / tokenize", "full kymograph, no cropping")
d.self_call(
    "enc", "multi-layer self-attention",
    "every token attends to ALL others -> no receptive-field limit (vs. ~50-100px in KymoButler)",
)
d.call("enc", "dec", "encoded memory (global features)")

loop_q = d.open_frame("loop", "for each of the N object queries (in parallel)", ("dec", "dec"))
d.self_call("dec", "cross-attention query_i -> memory", "no local crop, sees the whole kymograph")
d.self_call("dec", "mask head", "binary prediction track_i, or 'no object' class")
d.close_frame(loop_q)

d.call("dec", "ext", "N instance masks", "1 query = 1 complete track (crossings already resolved by attention)")

loop_m = d.open_frame("loop", "for each valid mask (not 'no object')", ("ext", "ext"))
d.self_call("ext", "walk the time axis", "active column per row -> polyline (t, x); no skeletonize/KDTree/greedy")
d.close_frame(loop_m)

d.call("ext", "kin", "N polylines (tracks)")
d.self_call("kin", "compute kinetics", "direction=sign(dx); velocity=dx/dt; distance; duration")
d.self_call("kin", "physical scaling", "PIXEL_SIZE_UM, SEGUNDOS_POR_FRAME (parametros.py)")
d.call("kin", "out", "tracks.csv/json + overlay.png")

d.finish(
    title="Sequence diagram — transformer pipeline (no KymoButler components)",
    out_path=OUT_DIR / "pipeline_transformer_kymografo.png",
    caption="Compare against pipeline_kymobutler.png — same output format, no classical tracking cascade or separate decision network",
    legend=[
        ("Data / preprocessing", BLUE),
        ("Transformer model (encoder / decoder)", AQUA),
        ("Bookkeeping (mask -> polyline)", YELLOW),
        ("Kinetics / postprocessing", GREEN),
        ("Output", VIOLET),
    ],
)
