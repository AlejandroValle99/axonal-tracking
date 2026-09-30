# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Experimental testbed (not a stable pipeline/library) for a CEIA master's thesis on automating
**axonal transport particle tracking** in hippocampal neuron microscopy. The lab has ~142-168 VSI
videos with manual ground truth (currently tracked by hand in ImageJ/TrackMate). The research
question: does transformer-based vision (SAM2/SAM3 + detectors) beat a classical CNN baseline
(KymoButler), and is it better to work on the **raw video** or on the **kymograph** (a 2D
time-vs-position projection that discards z/defocus/off-axis information)?

Notebooks, modules, and results are expected to change, break, or be discarded without notice —
don't over-engineer for stability.

## Environment & commands

Dependency management is via [uv](https://docs.astral.sh/uv/) (Python >= 3.13, see `.python-version`):

```bash
uv sync                  # install dependencies
uv sync --extra dev      # also install jupyter/ipykernel to run notebooks
uv run jupyter lab       # run notebooks
uv run python scripts/ver_video.py [ruta/al/Movie_NNN.vsi]   # view a video in napari
uv run python scripts/extraer_frames_etiquetado.py            # sample frames for Roboflow/CVAT labeling
```

`scripts/` has additional one-off utilities beyond the two above (synthetic dataset generation,
detection-label export/visualization, pipeline diagrams) — check there before writing a new script.

There is no test suite or build step configured in this repo — don't invent one unless asked.
Linting: `uv run ruff check .` (config in `pyproject.toml`'s `[tool.ruff]`).

Two sibling repos are pulled in as **editable path dependencies** (`[tool.uv.sources]` in
`pyproject.toml`) and must exist as checkouts next to this repo for `uv sync` to work:
- `../KymoButler` — Python/PyTorch port of the KymoButler tracker (used as the classical baseline).
- `../ingenia-kymograph` — the `synthkymo` package, synthetic kymograph generator.

Raw experimental data (135 GB of VSI videos/kymographs, not in this repo) lives at
`~/Desktop/Videos-Kymos-experimental data/` (see `data/README.md`).

## Architecture

### Notebooks = experiments, one per approach (`notebooks/`)

Numbered notebooks form the experimental pipeline, in rough dependency order:

| Notebook | Approach | Status |
|---|---|---|
| `01_preprocesamiento.ipynb` | VSI/ETS reading + preprocessing pipeline | Support |
| `02_sam2_prototipo.ipynb` | SAM2 segmentation directly on video | Prototype |
| `03_roi_sintetico_sam3.ipynb` | SAM3 axon-ROI detection from synthetic video, vs GT | Active |
| `04_sam3_prototipo.ipynb` | SAM3 segmentation directly on video | Prototype |
| `05_kimografo.ipynb` | Synthetic kymograph generation + exact ground truth | Active |
| `06_tracking_kymobutler.ipynb` | KymoButler tracking on (synthetic) kymographs vs GT | Baseline |
| `07_validacion_prompts_sam3.ipynb` | Cheap box- vs point-prompt validation for SAM3 before committing to the detect-then-segment build (Paso 4.0 of `plan/detection-segmentation-guide.md`) | Done (fed 08–09) |
| `08_deteccion_yolo.ipynb` | YOLO detector locates each kymograph track as a box (Stage 1 of detect-then-segment) | Baseline (negative result, frozen) |
| `09_segmentacion_transformer.ipynb` | SAM3 turns YOLO's boxes into a per-track mask (Stage 2 of detect-then-segment) | Baseline (negative result, frozen) |
| `10_mask2former_kymografo.ipynb` | Mask2Former segmentation on (synthetic) kymographs, vs KymoButler | Baseline (negative result, frozen) |
| `11_asociacion_atencion.ipynb` | Classical cost + global attention over KymoButler's own segments, replacing DecNet's greedy-local association | Baseline (negative result, frozen) |
| `12_kymorope.ipynb` | KymoRoPE: per-pixel transformer (physical-unit RoPE, native resolution) — build, verify, train, pixel-level inspection | Active |
| `13_kymorope_decode.ipynb` | KymoRoPE decode (per-pixel maps → trajectories) vs GT and KymoButler on val; reads `scripts/evaluar_kymorope.py` / `evaluar_kymobutler_400.py --split val` outputs | Active |

Notebooks 07–09 implement the "detect-then-segment" architecture described in
`plan/detection-segmentation-guide.md` (YOLO detector → SAM3 segmenter), which maps to
WBS items 5–6 of the formal thesis plan (`plan/Plan-Proyecto.pdf`). **This pipeline is closed
as a documented negative result** — frozen SAM3 assigns identity at chance on crossings
regardless of prompting (measured in NB09 §6–7; verdict and remaining-work list in
`plan/notebooks-08-09-closeout.md`). The successor, `plan/mask2former-guide.md` (notebook 10),
is **also closed as a negative result**: on the same 400-sample synthetic split, KymoButler
beats it on all three axes (F1 0.968 vs 0.899, ID-switch 9.9% vs 34.6%, position error 0.053 µm
vs 0.119 — `docs/revision-rumbo-vit.md` §2bis). That measurement pointed at KymoButler's
explicit association stage as the structural difference, which motivated notebook 11 — giving
the same segment representation an explicit association stage (classical cost, then global
attention, with and without DecNet's own visual tiles). **Notebook 11 is also a negative
result**: none of the three candidate rows passes Gate B, and DecNet's greedy-local design
remains unbeaten on ID-switch at every difficulty stratum (`plan/notebook11-closeout.md`,
`docs/revision-rumbo-vit.md` update of 2026-09-17). The remaining escalation the plan allows is
the per-pixel embedding head in `plan/kymotransformer-proposal.md` §3 — a change to the
representation, not the association reasoning.

This list shifts as notebooks get renumbered/split (see git history for current numbering — it has
moved before). When adding a new approach, follow the `NN_descripcion.ipynb` convention and update
`README.md`'s approach table.

### `src/axonal_tracking/` — shared modules imported by notebooks

- `configuracion.py` — shared config flowing *between* notebooks. NB01 writes
  `data/configuracion_actual.yaml` (selected video, frame range, processing mode); later notebooks
  read it and apply preprocessing via `aplicar_modo_frame` / `aplicar_modo_video`. Three modes:
  `rgb_crudo` (no background/noise removal, for seeing raw model behavior), `completo` (original
  full pipeline), `configurable` (stages individually toggleable).
- `ets_reader.py` — dependency-free reader for Olympus IX83 VSI/ETS files (format documented in
  `docs/guia-formato-vsi-ets.md`). Avoids needing Java/Bio-Formats.
- `preprocesamiento.py` — pure array-in/array-out functions: uint16→RGB uint8 (percentile stretch,
  defaults anchored at p50-p99.8 to land on background, not the vignette corners), spatial/temporal
  background subtraction, noise reduction, ROI masking. Chainable via `pipeline_frame()`.
- `kimografo.py` — builds a kymograph (T x L image: rows=time, columns=position along axon) from a
  video + ImageJ `.roi` polyline, by projecting pixel values along the ROI. Moving vesicles → diagonal
  lines (slope = velocity); static vesicles → vertical lines. Validated against the lab's manual
  `Kymograph_NNN.tif` files.
- `kimografo_sintetico.py` — generates **synthetic** kymographs with exact, per-frame ground truth
  (no video is rendered, works directly in kymograph space). Particle types: `estacionaria` (constant
  x, vertical line), `anterograda`/`retrograda` (stochastic runs at ~constant velocity interleaved
  with pauses — anterograde = position increases with time). This exists specifically to validate
  trackers (KymoButler, transformers) before fighting with noisy real data — see
  `transcripts/2026-06-12 09-03-26_resumen.md` (points 6-7) for the originating decision.
- `parametros.py` — single source of truth for px→µm and frame→seconds conversion
  (`PIXEL_SIZE_UM`, `SEGUNDOS_POR_FRAME` per session N1/N2/N3/Ex, `RANGOS_SESION`). Any velocity
  calculation must go through these, not hardcoded constants — sampling interval varies per
  acquisition session and isn't documented for all sessions (`None` = undocumented).
- `visualizacion.py` — matplotlib helpers for frames/masks/blobs/boxes; all take an optional `ax` to
  compose into larger layouts. Default contrast stretch matches `preprocesamiento.frame_a_rgb_uint8`
  (p50-p99.8).
- `etiquetas_deteccion.py` — exports detection/segmentation labels from synthetic synthkymo datasets
  to train the YOLO detector and/or SAM3 segmenter (notebooks 07-09's data layer).
- `evaluacion.py` — the shared trajectory harness (`evaluar_trayectorias_polilineas`) every
  pipeline is measured with, plus `extraer_subpixel` (the repo's only centroid) and the shared
  scene loader / mobility filter / fragmentation and stratified summaries.
- KymoRoPE (notebooks 12–13): `kymorope.py` (model + device-agnostic helpers), `datos_pixel.py`
  (per-pixel targets + disk cache), `entrenamiento.py` (adapter over `notebooks/trainer.py`),
  `decodificacion.py` (per-pixel maps → trajectory polylines for the harness). The KymoRoPE path
  must not import `kymobutler` (it has to run on Colab without the sibling repo).
  "Mobile" means total position range ≥ 8 px everywhere (targets import the harness's
  `ed.MIN_DESPLAZAMIENTO_PX_MOVIL`; until 2026-09-27 the targets used 4 px by mistake, and the
  checkpoint `kymorope_40ep_2026-09-24.pt` was trained that way). The lab's own static/mobile rule
  is a 2° inclination angle; it is applied *after* decoding, per trajectory
  (`evaluacion.clasificar_por_angulo`), not used as a training label.
  Data: `datasets/train` holds 33,000 samples (14 generator profiles; the original 800 are
  `manifest.orig.csv` and are backed up in `datasets/train_800_viejo/`). Use
  `subconjunto_train=N` (representative, nested, includes the 800) — not `limite_train`, whose
  first N are almost all one profile. Train via NB12 `entrenar("run_name", ...)`: from scratch,
  fine-tune (`pesos_iniciales=`) or resume (`reanudar=True`, same arguments); each run has its own
  `results/kymorope/checkpoints/<run>/`.

### `config.yaml` — synthetic kymograph generation parameters

Top-level config (separate from `data/configuracion_actual.yaml`) consumed by the `synthkymo`
package: video/kymograph dimensions, particle speed/size/brightness ranges, particle type counts
(static/anterograde/retrograde/etc.), noise model (background, read noise, shot noise, hot pixels,
photobleaching), and rendering artifacts (static particles, tails, smoke) used to progressively add
realism ("nivel 2", "nivel 3", ...) on top of the ideal noiseless case.

### `docs/` — technical reference notes, in Spanish

Read before touching the related code:
- `guia-formato-vsi-ets.md` — VSI/ETS binary format, backs `ets_reader.py`.
- `kymobutler-analisis.md` — deep dive on KymoButler's architecture (U-Net + DecNet) and tracking
  algorithm, written specifically to ground the baseline comparison.
- `dataset-findings.md` — dataset audit (counts of videos/ROIs/kymographs/ground-truth per session,
  missing-ROI/orphan-ROI lists). Source of truth for "how much usable data do we actually have."
- `parametros-experimentales.md` — backs `parametros.py`.
- `alternativa-java-bioformats.md` — notes on the Bio-Formats/Java alternative that `ets_reader.py`
  avoids needing.
- `handover.md` — handover notes for picking the project back up.
- `regenerar-dataset-sintetico.md` — how to regenerate the synthetic kymograph dataset.
- `synthkymo-axon-path-followup.md` — follow-up notes on `synthkymo` axon-path generation.
- `KymoTransformer_Research_Proposal.md` — research proposal backing
  `plan/kymotransformer-proposal.md` (pre-existing exception to the naming convention below).

New docs in this directory should follow the same convention: Spanish, kebab-case filenames.

`docs/`, `plan/`, `transcripts/`, and `data/` are gitignored — a fresh clone won't have them. They
hold private thesis-in-progress material (design docs, meeting notes, raw data) rather than
code, so treat file references into these directories as accurate for the current checkout, not
guaranteed to exist elsewhere.

### `transcripts/`

Dated meeting-summary markdown files recording decisions with the thesis advisor/lab. Cite these
when a design choice in the code traces back to a specific meeting (e.g. the synthetic-kymograph
strategy in `kimografo_sintetico.py`).
