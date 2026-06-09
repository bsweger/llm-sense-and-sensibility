import logging
import sys
from pathlib import Path

import streamlit as st

# Quiet the warnings Streamlit's source watcher logs while walking transformers'
# lazy modules (failed optional torchvision imports) on every rerun.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Put src/ on the path so the llm_sas package imports without being installed:
# Streamlit Community Cloud can't build it (no git metadata for the version).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_sas import PROJECT_DIR  # noqa: E402
from llm_sas.demos.models import render_model_selector  # noqa: E402

_PAGES_DIR = PROJECT_DIR / "demos" / "pages"

st.set_page_config(
    page_title="Deep Dive: LLMs",
    layout="centered",
)

pages = [
    st.Page(_PAGES_DIR / "home.py", title="Home", icon="🏠", default=True),
    st.Page(_PAGES_DIR / "tokenizer.py", title="Tokenization", icon="🔡"),
    st.Page(_PAGES_DIR / "next_token.py", title="Inference", icon="🎯"),
    st.Page(_PAGES_DIR / "attention.py", title="Attention", icon="🔗"),
]

pg = st.navigation(pages)

# Render the shared model picker once, before running the page, so the same
# widget appears on every page and the selection carries across demos.
render_model_selector(
    help=(
        "All demos use this model. Each model has its own tokenizer, was trained on "
        "different data, and has its own layer/head counts, so switching it changes "
        "tokenization, the predicted distribution, and the attention pattern."
    ),
)

pg.run()
