# XASBind

An ImageBind-style multimodal model that binds three views of the same X-ray
absorption event into a shared embedding space:

- **crystal structure** (periodic graph)
- **XANES** (near-edge spectrum)
- **EXAFS** (extended fine-structure spectrum)

Data comes from the Materials Project XAS dataset.

## Environment setup

Create an isolated Python environment, then install everything with pip. Conda
provides just the interpreter; all packages come from pip (avoid mixing
`conda install` and `pip install` in the same env).

```bash
conda create -n xasbind python=3.11
conda activate xasbind
pip install -e ".[dev]"     # drop "[dev]" to skip pytest + matplotlib
```

`pyproject.toml` lists every dependency, so that single pip command installs the
full stack plus this package in editable mode (your code changes take effect
without reinstalling). A `venv` works just as well in place of conda:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The `src/` layout means scripts and tests import the package
(`from xasbind.data import make_dataloaders`) rather than via path hacks.

**GPU vs CPU PyTorch.** `pip install` pulls `torch` from PyPI, which on Linux
defaults to a CUDA build. To pin a specific CUDA version (or force CPU), install
torch from the official index after the steps above, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only
```

## Pipeline

The dataset is not committed (see `.gitignore`); the scripts regenerate it.

```bash
# 1. Fetch all XAS spectra -> data/xas/matproj/{xas.h5, metadata.csv}
export MP_API_KEY=your_key_here
python scripts/fetch_matproj.py

# 2. Pair XANES/EXAFS + grouped split -> paired_metadata.csv, *_indices.npy
python scripts/prepare_data.py --valid-size 0.2 --test-size 0.1

# 3. Load for training
python -c "from xasbind.data import make_dataloaders; \
           dl = make_dataloaders('data/xas/matproj'); print(dl['train'])"
```

The split groups on `material_id` so no crystal structure leaks across
train/valid/test, while approximately stratifying by absorbing element.
Fractions are approximate (whole materials move together); the script prints the
achieved sizes and a leakage check.

## Layout

```
scripts/        thin CLI entrypoints (fetch, prepare)
src/xasbind/
  data/         dataset.py (HDF5 dataloader), splits.py (grouped split)
  models/       encoders + model           (TODO)
  losses/       contrastive loss           (TODO)
  training/     trainer + metrics          (TODO)
  utils/        seeding, helpers
tests/          split correctness (leakage / disjointness)
```

## Tests

```bash
pytest
```

## Notes

- Set the Materials Project key via `MP_API_KEY`; never commit it.
