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

### Changed

- Dropped demo numbering from page titles, nav entries, and docstrings.
- `load_model` now loads with `attn_implementation="eager"` so per-head
  attention probabilities are available to the attention demo. Negligible
  perf impact on the small models in the catalog.
