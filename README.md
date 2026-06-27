# axonal-tracking

> ⚠️ **Experimental repository.** This repo is **not a finished pipeline or a stable library**:
> it is a testbed where different approaches to automating axonal transport tracking are
> explored and compared. The notebooks, modules, and results represent work in progress and
> may change, break, or be discarded without notice.

Exploratory analysis of multiple approaches for **particle tracking in axonal transport**
of hippocampal neurons, as part of a master's thesis (CEIA).

The lab has ~142–168 microscopy videos with manual ground truth (currently analyzed by hand
in ImageJ/TrackMate). The goal of this research is to **automate that analysis** and evaluate
which family of techniques works best, comparing them against each other and against a
classical baseline.

## Approaches under analysis

This repository collects experiments across several evaluation branches:

| Approach | Idea | Status |
|----------|------|--------|
| **Transformers over raw video** | Segmentation with SAM2 / SAM3 guided by detectors (blob_log, Grounding DINO) directly on the VSI stack | Prototypes (NB02, NB03) |
| **Transformers over kymographs** | Same model family, but operating on the kymograph image instead of the video | Exploration |
| **KymoButler baseline** | U-Net + classical tracker (Jakobs, Franze & Bhatt 2019, *eLife*) over kymographs | Baseline (NB05) |
| **Synthetic kymographs** | Generation of kymographs with exact ground truth to validate trackers at progressive difficulty | Active (NB04) |
| **Classical preprocessing** | VSI/ETS reading, background subtraction, kymograph extraction | Support (NB01) |

The central question running through the experiments: does transformer-based vision add value
over a classical CNN, and is it better to work on the **raw video** (which preserves information
about z, defocus, and off-axis motion) or on the **kymograph** (a projection that discards part
of that information)?

## Structure

```
notebooks/   Experiments, one per approach/stage
  01_preprocesamiento.ipynb     Reading and preparing the videos
  02_sam2_prototipo.ipynb       SAM2 segmentation over video
  03_sam3_prototipo.ipynb       SAM3 segmentation over video
  04_kimografo.ipynb            Synthetic kymograph generation + GT
  05_tracking_kymobutler.ipynb  KymoButler tracking and comparison vs GT
src/axonal_tracking/   Support modules reused by the notebooks
  ets_reader.py, preprocesamiento.py, kimografo.py,
  kimografo_sintetico.py, configuracion.py, visualizacion.py, ...
docs/        Technical notes and findings (in Spanish)
scripts/     Standalone utilities (view video, extract frames)
data/        Shared configuration and generated synthetic data
```

> 📄 Detailed documentation for each approach lives in [`docs/`](docs/)
> (KymoButler analysis, VSI/ETS format, experimental parameters, dataset findings).

## Environment

The project uses [uv](https://docs.astral.sh/uv/) to manage dependencies (Python ≥ 3.13):

```bash
uv sync                 # install dependencies
uv sync --extra dev     # include jupyter / ipykernel to run the notebooks
```

> KymoButler is referenced as an editable dependency from a sibling checkout
> (`../KymoButler`); see `pyproject.toml`.

---

*Research work in progress — CEIA master's thesis. The results here are preliminary and part
of the exploration of approaches, not a closed product.*
