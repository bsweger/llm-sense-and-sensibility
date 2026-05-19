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
- Attention demo (`pages/attention.py`): interactive per-head heatmap with
  layer/head sliders sized from the selected model's config, plus a Winograd
  minimal-pair view that auto-picks the head whose attention pattern flips
  most cleanly between the two sentences. Uses GPT-2's `output_attentions`
  via a new `get_attentions` helper on the shared models module.
- `plotly` added as a runtime dependency for the attention visualizations.

### Changed

- Dropped demo numbering from page titles, nav entries, and docstrings.
- `load_model` now loads with `attn_implementation="eager"` so per-head
  attention probabilities are available to the attention demo. Negligible
  perf impact on the small models in the catalog.
