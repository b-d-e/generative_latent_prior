# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv sync
```

All dependencies including torch, vllm, nnsight, and transformers are declared in `pyproject.toml`. The `transformers==4.47.0` pin is enforced via `[tool.uv] override-dependencies` to maintain compatibility with vllm and nnsight (vllm installs its own transformers version which would otherwise conflict).

## Common Commands

**Train a GLP (quickstart with Llama1B, ~few minutes):**
```bash
# Download 1M activation dataset first
huggingface-cli download generative-latent-prior/llama1b-layer07-fineweb-1M \
    --repo-type dataset --local-dir data/llama1b-layer07-fineweb-1M --local-dir-use-symlinks False
# Launch training
uv run python3 glp_train.py config=configs/train_llama1b_static.yaml
# Override config values via CLI (OmegaConf merge)
uv run python3 glp_train.py config=configs/train_llama1b_static.yaml batch_size=2048
```

**Run scalar probing evaluation (113 binary classification datasets):**
```bash
uv run python3 glp/script_probe.py
# Override config, e.g. reduce memory usage:
uv run python3 glp/script_probe.py batch_size=64
```

**Load a pretrained GLP in Python:**
```python
from glp.denoiser import load_glp
model = load_glp("generative-latent-prior/glp-llama8b-d6", device="cuda:0", checkpoint="final")
```

**Demo notebook:** `glp_demo.ipynb`

## Architecture

### Key Concept: Timestep Convention
The paper uses `t` for timestep. The codebase follows diffusers convention and uses `u = 1 - t` instead.

### Core Model: `glp/denoiser.py`
- **`GLP`** — top-level wrapper combining `Normalizer`, `Denoiser`, and a `FlowMatchEulerDiscreteScheduler`. The `forward()` method runs a full flow-matching training step (adds noise, predicts velocity, returns MSE loss).
- **`Denoiser`** — wraps `TransformerMLPDenoiser`; handles reshaping between `(batch, seq, dim)` and `(batch*seq, dim)` since the denoiser operates per-token.
- **`TransformerMLPDenoiser`** — MLP with SwiGLU gating and multiplicative timestep conditioning. Optionally accepts a `layer_idx` for multi-layer models.
- **`Normalizer`** — normalizes/denormalizes activations using precomputed mean/variance statistics from `rep_statistics.pt`.
- **`load_glp(weights_folder, device, checkpoint)`** — downloads from HuggingFace if not local, loads config + weights.

### Flow Matching: `glp/flow_matching.py`
- **`fm_prepare()`** — adds noise at a given timestep `u` for training (interpolates between clean activation and noise).
- **`sample()`** — generates activations from pure noise using the denoiser iteratively.
- **`sample_on_manifold()`** — SDEdit-style: noise an existing activation partway and denoise it back, projecting it onto the learned manifold.

### Training: `glp_train.py`
- Config via OmegaConf: base `TrainConfig` dataclass merged with YAML config file, then CLI overrides (`config=path.yaml key=value`).
- Dataset stored as memory-mapped numpy arrays (`MemmapReader`/`MemmapWriter` in `glp/utils_acts.py`). Each activation is saved as a flat `(dim,)` array.
- Dataset format: a directory containing `data_0000.npy`, `data_0001.npy`, ..., `data_indices.npy`, and `dtype.txt`.
- `ActivationCollator` normalizes activations at collation time using the `Normalizer`.

### Activation Utilities: `glp/utils_acts.py`
- **`save_acts()`** — extracts intermediate activations from a HuggingFace model using `baukit.TraceDict`.
- **`MemmapWriter`/`MemmapReader`** — chunked memory-mapped storage for large activation datasets.

### Applications
- **Scalar probing** (`glp/script_probe.py`): Extracts internal GLP activations ("meta-neurons") at intermediate denoiser layers as features, then runs logistic regression for binary classification.
- **On-manifold steering** (`glp/script_steer.py`): `postprocess_on_manifold_wrapper()` takes edited activations and projects them back onto the activation manifold via SDEdit. `addition_intervention()` integrates with `baukit.TraceDict` to hook into a HuggingFace model during generation.
- **Persona Vectors integration** (`integrations/persona_vectors/`): Requires a separate `persona` conda environment; see `integrations/persona_vectors/README.md`.

### Config Files
- `configs/train_llama1b_static.yaml` — Llama-3.2-1B GLP (d_input=2048, layer 7)
- `configs/train_llama8b_static.yaml` — Llama-3.1-8B GLP (d_input=4096, layer 15)

### Multi-layer Models
When training on multiple layers simultaneously, dataset paths should contain `layer_<idx>` in the directory name (used to extract `layer_idx` for the denoiser embedding). Set `multi_layer_n_layers` in `denoiser_config` to enable layer embeddings.

## Pretrained Weights

All weights on HuggingFace at `generative-latent-prior/`. Main model: `glp-llama8b-d6`. Checkpoints labeled `epoch_N` correspond to N million activations; `final` = `epoch_1024` (1B activations).
