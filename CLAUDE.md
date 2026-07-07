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

There is no test suite, linter, or build step configured in this repo — don't invent one unless asked.

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

New docs in this directory should follow the same convention: Spanish, kebab-case filenames.

### `transcripts/`

Dated meeting-summary markdown files recording decisions with the thesis advisor/lab. Cite these
when a design choice in the code traces back to a specific meeting (e.g. the synthetic-kymograph
strategy in `kimografo_sintetico.py`).
