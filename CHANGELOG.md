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

### Changed

- Dropped demo numbering from page titles, nav entries, and docstrings.
