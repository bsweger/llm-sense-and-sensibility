"""
Shared model + tokenizer loading and selection for the demos.

All demo pages pick from the same catalog of Hugging Face models. This module
centralizes the catalog, the cached loaders, and the sidebar selector so adding
a new model is a one-line change to ``MODELS``.
"""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# Display label -> Hugging Face model id. First entry is the default.
MODELS = {
    "GPT-2 (124M)": "gpt2",
    "Pythia-160m": "EleutherAI/pythia-160m",
}


@st.cache_resource(show_spinner=True)
def load_tokenizer(model_id: str):
    """Load just the tokenizer for the given model id. Cached per model."""
    return AutoTokenizer.from_pretrained(model_id)


@st.cache_resource(show_spinner=True)
def load_model(model_id: str):
    """Load a causal LM and its tokenizer by Hugging Face id. Cached per model id."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def render_model_selector(*, key: str, help: str | None = None) -> str:
    """Render a sidebar model picker and return the chosen Hugging Face model id.

    Parameters
    ----------
    key : str
        Streamlit widget key — must be unique per page so each demo tracks its
        own selection independently.
    help : str, optional
        Tooltip text shown next to the selector. Pages can use this to explain
        what changing the model means in the context of their own demo.
    """
    with st.sidebar:
        st.header("Model")
        label = st.selectbox("Hugging Face model", list(MODELS.keys()), key=key, help=help)
    return MODELS[label]
