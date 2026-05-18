"""Home page — landing content."""

import streamlit as st

st.title("NPL Deep Dive: LLMs Under the Hood")
st.markdown(
    """
    Demos for the 2026-05-20 deep dive.

    Use the sidebar to navigate between demos.

    ### Next-token prediction
    Step through a sentence one token at a time. The model produces a probability
    distribution over the vocabulary at each step; you choose which candidate to
    accept.

    ### Tokenization
    See how a prompt gets split into integer tokens. The same text gets sliced
    differently by different tokenizers — switch models in the sidebar to compare.

    ---
    """
)
