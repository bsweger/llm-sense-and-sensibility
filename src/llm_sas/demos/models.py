"""
Shared model + tokenizer loading and selection for the demos.

All demo pages pick from the same catalog of Hugging Face models. This module
centralizes the catalog, the cached loaders, and the sidebar selector so adding
a new model is a one-line change to ``MODELS``.
"""

from dataclasses import dataclass

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


@dataclass(frozen=True)
class ModelSpec:
    """Per-model metadata used across the demos.

    Add fields as new attributes become relevant (e.g. a ``homepage`` URL, a
    parameter count, a license, a family tag). Frozen so instances are hashable
    and can be used as ``@st.cache_resource`` keys.

    Attributes
    ----------
    model_id : str
        Hugging Face id used to load the causal LM weights.
    tokenizer_id : str, optional
        Hugging Face id used to load the tokenizer. Defaults to ``model_id``.
        Set explicitly when a fine-tune reuses its base model's tokenizer
        without re-uploading the tokenizer files (e.g. ``contextlab/gpt2-austen``
        ships only the model weights, so the tokenizer must come from
        ``openai-community/gpt2``).
    """

    model_id: str
    tokenizer_id: str | None = None

    @property
    def effective_tokenizer_id(self) -> str:
        return self.tokenizer_id or self.model_id


# Display label -> ModelSpec. First entry is the default.
MODELS: dict[str, ModelSpec] = {
    "GPT-2 (124M)": ModelSpec(model_id="openai-community/gpt2"),
    "Pythia-160m": ModelSpec(model_id="EleutherAI/pythia-160m"),
    "GPT-2 Austen": ModelSpec(
        model_id="contextlab/gpt2-austen",
        tokenizer_id="openai-community/gpt2",
    ),
}


@st.cache_resource(show_spinner=True)
def load_tokenizer(spec: ModelSpec):
    """Load just the tokenizer for the given spec. Cached per spec."""
    return AutoTokenizer.from_pretrained(spec.effective_tokenizer_id)


@st.cache_resource(show_spinner=True)
def load_model(spec: ModelSpec):
    """Load a causal LM and its tokenizer for the given spec. Cached per spec."""
    tokenizer = AutoTokenizer.from_pretrained(spec.effective_tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(spec.model_id)
    model.eval()
    return tokenizer, model


def render_model_selector(*, key: str, help: str | None = None) -> ModelSpec:
    """Render a sidebar model picker and return the chosen ModelSpec.

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
