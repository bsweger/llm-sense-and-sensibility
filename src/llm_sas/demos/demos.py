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
pg.run()
