"""Generates a SEQUENCE diagram (UML-like) of the KymoButler pipeline
(baseline), with the method/parameter detail documented in
docs/kymobutler-analisis.md. Saves the result to diagrams/.

Color = family, in the dataviz skill's fixed categorical order (blue, aqua,
yellow, green, violet): DecNet reuses U-Net's aqua because both are neural
networks (same entity family), not a new hue — see the legend.
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
    ("kymo", "Kymograph\n(PNG/TIFF)", BLUE),
    ("pre", "Preprocessing\npreprocessing.py", BLUE),
    ("unet", "U-Net\nsegmentation.py", AQUA),
    ("trk", "Tracker\ntracking.py", YELLOW),
    ("dec", "DecNet\nvision_net.py", AQUA),
    ("post", "Postproc\npostprocessing.py", GREEN),
    ("out", "Output\nCSV/JSON + PNG", VIOLET),
]

d = SeqDiagram(PARTICIPANTS)

d.call("kymo", "pre", "load_and_preprocess(path)", "PNG / TIFF / RGBA")
d.self_call("pre", "RemoveAlphaChannel", "composite over WHITE background")
d.self_call("pre", "grayscale + float32 [0,1]")
d.self_call("pre", "rescale_intensity()", "global intensity stretch")
d.self_call("pre", "is_negated()", "n1>=n2 -> invert (light bg -> dark)")
d.self_call("pre", "normalize_lines()", "divide each row by its mean -> removes bleach decay")
d.self_call("pre", "resize_to_multiple_of_16()", "required by the 4 levels of MaxPool2d")

d.call("pre", "unet", "normalized kymograph (1,H,W)")
d.self_call(
    "unet", "BiNet / UniNet forward",
    "4-level encoder (n,2n,4n,8n) + 16n bottleneck + decoder, BasicBlock=Conv+BN+LeakyReLU(0.1)",
)
d.call("unet", "trk", "foreground probability map (HxW)", "bidirectional: 1 map | unidirectional: ant+ret")

d.self_call("trk", "process_segmentation_bi()", "threshold=0.2, hit-or-miss morphology -> 1px skeletonization")
d.self_call("trk", "detect_seeds()", "endpoints with no neighbor in the previous row")
d.self_call("trk", "KDTree(skeleton_px)", "SEARCH_RADIUS = 1.5 px")

loop_seeds = d.open_frame("loop", "for each unvisited seed", ("trk", "dec"))
d.self_call("trk", "radius query -> candidates", "pick highest row_index if >1 on the first step")
alt_amb = d.open_frame("alt", "1 candidate -> append  |  0 or >=2 candidates and len>=3 -> vision", ("trk", "dec"))
d.call("trk", "dec", "get_candidates(tile 48x48)", "channels: kymo | track_mask (dropout .05) | fullbin_mask (dropout .5)")
d.call("dec", "trk", "prob. map 2x48x48 -> next point", "vision_threshold = 0.5, decision_prob is stored")
d.close_frame(alt_amb)
d.self_call("trk", "backward protection", "1 backward step allowed; 2nd step -> _go_back() and ends the track")
d.close_frame(loop_seeds)

loop_strad = d.open_frame("loop", "straddler recovery, up to STRADDLER_MAX_ITERATIONS", ("trk", "trk"))
d.self_call("trk", "subtract tracked px + re-filter", "min_size=5, min_frames=3 -> new seeds -> _make_track()")
d.close_frame(loop_strad)

d.self_call(
    "trk", "cleanup cascade",
    "clamp -> average_duplicates -> remove_subset_tracks -> resolve_overlaps (DecNet mean prob.) -> split_at_gaps -> min_frames filter",
)
d.call("trk", "post", "list[Track] (points + decision_probs)")
d.self_call("post", "compute kinetics", "direction=sign(dcol); velocity=mean|dcol/drow|; distance=sum|dcol|; duration")
d.self_call("post", "physical scaling", "pixel_space um/px, pixel_time s/frame")
d.call("post", "out", "tracks.csv/json + overlay.png")

d.finish(
    title="Sequence diagram — KymoButler pipeline (baseline)",
    out_path=OUT_DIR / "pipeline_kymobutler.png",
    caption="Source: docs/kymobutler-analisis.md — unidirectional mode skips DecNet (8-connected components = track)",
    legend=[
        ("Data / preprocessing", BLUE),
        ("Neural network (U-Net / DecNet)", AQUA),
        ("Classical algorithm (tracking.py)", YELLOW),
        ("Kinetics / postprocessing", GREEN),
        ("Output", VIOLET),
    ],
)
