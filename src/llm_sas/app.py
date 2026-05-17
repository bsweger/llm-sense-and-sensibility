"""Entry point for Streamlit handoff."""

import sys
from pathlib import Path

import structlog
from streamlit.web import cli as stcli

from llm_sas import PROJECT_DIR


logger = structlog.get_logger()


def main():
    app_path = PROJECT_DIR / "demo.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
