"""
Shared model + tokenizer loading and selection for the demos.

All demo pages pick from the same catalog of Hugging Face models. This module
centralizes the catalog, the cached loaders, and the sidebar selector so adding
a new model is a one-line change to ``MODELS``.
"""

from dataclasses import dataclass

import streamlit as st
import torch
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
    "Qwen2.5-0.5B": ModelSpec(model_id="Qwen/Qwen2.5-0.5B"),
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
    """Load a causal LM and its tokenizer for the given spec. Cached per spec.

    The model is loaded with ``attn_implementation="eager"`` so that
    ``output_attentions=True`` returns the per-head attention probability tensors
    used by the attention-visualization demo. The fused SDPA / flash kernels
    that transformers picks by default skip producing those tensors. Eager is
    a touch slower but the perf hit on these small models is negligible.

    ``dtype=torch.float32`` is forced because some checkpoints (e.g. Pythia-160m)
    ship fp16 weights whose pre-softmax attention scores overflow under the eager
    kernel, producing NaN logits. fp32 is safe across every model in MODELS.
    """
    tokenizer = load_tokenizer(spec)
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, attn_implementation="eager", dtype=torch.float32)
    model.eval()
    return tokenizer, model


@st.cache_data(show_spinner=False)
def get_attentions(text: str, spec: ModelSpec) -> tuple[list[str], torch.Tensor]:
    """Run a forward pass and return per-layer self-attention weights.

    Parameters
    ----------
    text : str
        Input prompt. Tokenized with the spec's tokenizer.
    spec : ModelSpec
        Model to use. Loaded (and cached) via :func:`load_model`.

    Returns
    -------
    tokens : list of str
        Decoded per-token strings, in order. ``len(tokens) == seq_len``.
    attn : torch.Tensor
        Stacked attention weights, shape ``(num_layers, num_heads, seq, seq)``.
        ``attn[l, h, q, k]`` is the weight from query token ``q`` to key token
        ``k`` in layer ``l``, head ``h``. Rows sum to 1.

    Notes
    -----
    Cached on ``(text, spec)`` so editing the prompt re-runs the model but
    moving layer/head sliders is free.
    """
    tokenizer, model = load_model(spec)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    # outputs.attentions is a tuple of length num_layers, each (1, heads, seq, seq).
    attn = torch.stack([a.squeeze(0) for a in outputs.attentions])  # (L, H, S, S)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    return tokens, attn


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


def render_model_link(spec: ModelSpec) -> None:
    """Render a sidebar section linking to the model's Hugging Face page.

    Call this at the bottom of each page (after any other ``st.sidebar`` writes)
    so the link appears at the bottom of the sidebar.
    """
    with st.sidebar:
        st.divider()
        st.markdown(
            f"**Model details**\n\n"
            f'<span style="font-size: 0.85rem;">'
            f'<a href="https://huggingface.co/{spec.model_id}">View {spec.model_id} on Hugging Face ↗</a>'
            f"</span>",
            unsafe_allow_html=True,
        )
