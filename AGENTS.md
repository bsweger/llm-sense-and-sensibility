# Agent Instructions

This is Python application with two main components:

1. A marimo notebook used for slides (slides.py)
2. A streamlit application used for demos that accompany the slides (demos.py)

## Dependencies and Environment

- Use `uv` for all dependency management operations, not pip or poetry
- Prefix commands with `uv run` to execute in the project's virtual environment
- Add dependencies via `uv add <package>` (updates both `pyproject.toml` and `uv.lock`)

## Code Style

- Don't number the Streamlit demos in comments or in any naming conventions
- Python doctrings should follow NumPy docstring format
- Python functions parameters and outputs should have type hints
- Ruff enforces double quotes and 120-character line length in Python files
- Pre-commit hooks run automatically on commit and must pass
- Run `uv run pre-commit run --all-files` to check before committing
- Markdown files should have an 80-character line length limit
- For data manipulation, prefer polars over pandas

## Commands

- Python setup and package commands use `uv`
    - `uv sync` to create a virtual environment and sync it to uv.lockfile
    - `uv run pytest` to run tests
    - `uv run llmdemo` to run the Streamlit application locally

## Testing

- Tests use pytest with `pytest-random-order` plugin
- Run tests via `uv run pytest`
- New features require test coverage

## Pull Requests

- Update `CHANGELOG.md` `[Unreleased]` section with your changes
- Link PRs to their corresponding issue
- Ensure all pre-commit checks pass before creating PR