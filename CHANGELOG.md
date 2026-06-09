# Changelog

All notable changes to llm_sas are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com), and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Next-token demo: sidebar model selector to toggle between `gpt2` and
  `EleutherAI/pythia-160m`. Switching models re-tokenizes the current prompt.
- Tokenization demo (`pages/tokenizer.py`): shows how a prompt maps to integer
  token ids, with an inline colored-pill view and a detail table listing each
  id alongside its raw BPE piece and decoded text. Shares the same model
  selector so users can compare tokenizers on the same input.
- Attention demo (`pages/attention.py`): interactive token-to-token heatmap
  with a single layer slider sized from the selected model's config. Each
  cell shows attention averaged across the layer's heads — heads are hidden
  from the audience since this is a high-level talk. Uses a new
  `get_attentions` helper on the shared models module that pulls
  per-layer self-attention via `output_attentions=True`.
- `plotly` added as a runtime dependency for the attention visualizations.
- Streamlit Community Cloud deployment: instructions in `README.md`, plus
  `scripts/gen_cloud_requirements.sh` and a pre-commit hook that export
  `src/llm_sas/demos/requirements.txt` from `uv.lock`.

### Changed

- Model selection is now a single shared setting across all demos.
- Inference demo: the "Tokens to generate per click" slider is now the first
  control under "Next token controls".
- Dropped demo numbering from page titles, nav entries, and docstrings.
- Model caches (`load_model`, `load_tokenizer`, `get_attentions`) limited to
  `max_entries=1` so only one model is resident at a time, staying within
  Community Cloud's 1 GB per-app memory limit.
- On Linux, `torch` now resolves to the CPU-only build from the PyTorch CPU
  index (via `[tool.uv.sources]`), avoiding the multi-GB CUDA download on
  Community Cloud. macOS is unaffected.
- `demos.py` puts `src/` on `sys.path` (so the package imports without being
  installed, since Streamlit Cloud can't build it) and builds page paths from
  `llm_sas.PROJECT_DIR`.
- `load_model` now loads with `attn_implementation="eager"` so per-head
  attention probabilities are available to the attention demo. Negligible
  perf impact on the small models in the catalog.

### Fixed

- `load_model` now loads in `bfloat16` instead of `float32` to prevent out of
  memory errors on Streamlit Cloud while retaining the numeric range of fp32
  (required to prevent NaN errors in the attention demo).
