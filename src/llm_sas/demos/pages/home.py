"""Home page — landing content."""

import streamlit as st

st.title("NPL Deep Dive: LLMs Under the Hood")
st.markdown(
    """
    Demos for the 2026-05-20 deep dive.

    Use the sidebar to pick a demo.

    ### Tokenization
    Shows how a prompt gets split into individual tokens. The same text gets sliced
    differently by different tokenizers — switch models in the sidebar to compare.

    ### Inference
    Step through a sentence one token at a time. The selected model produces
    a probability distribution over the vocabulary at each step. You can choose
    to complete the prompt using the token with the highest probability, or
    you can choose to sample.

    ### Attention (WIP)
    We didn't talk much about _attention_ in the deep dive, but this demo
    is an attempt to show how attention works during inference. For a given
    prompt, you choose a token, and the demo shows how that token "pays
    attention" to the other tokens in the prompt.

    _Attention_ is an important input for generating the probability scores
    used for inference.

    ---
    """
)
