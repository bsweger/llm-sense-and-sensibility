"""Home page — landing content."""
import streamlit as st

st.title("How LLMs Work Under the Hood")
st.markdown(
    """
    Interactive demos for a 40-minute presentation.

    Use the sidebar to navigate between demos.

    ### Part 1 — Next-token prediction
    Step through a sentence one token at a time. The model produces a probability
    distribution over the vocabulary at each step; you choose which candidate to
    accept. Candidates are hidden by default so the audience can guess first.

    ---

    *More parts (attention, sampling temperature, top-p) can be added as
    additional pages.*
    """
)
