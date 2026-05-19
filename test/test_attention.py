"""Tests for the attention demo.

Only ``get_attentions`` has non-trivial behavior worth covering here — the
rest of the page is Streamlit glue and a Plotly heatmap that's better
verified by eye in the running app.
"""

import torch

from llm_sas.demos.models import MODELS, get_attentions


def test_get_attentions_returns_expected_shape_for_gpt2():
    """Smoke-test the full pipeline against the real GPT-2 model.

    First call downloads ~500MB of weights; subsequent calls hit the local
    Hugging Face cache and run in well under a second.
    """
    spec = MODELS["GPT-2 (124M)"]
    tokens, attn = get_attentions("Hello world.", spec)
    assert len(tokens) == attn.shape[-1]
    # GPT-2 small: 12 layers, 12 heads.
    assert attn.shape[0] == 12
    assert attn.shape[1] == 12
    # Causal mask: each row sums to ~1.
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
