"""llm_sas initialization."""

from pathlib import Path

from llm_sas.logger import configure_logging

PROJECT_DIR: Path = Path(__file__).parent

# configure structlog
configure_logging()
