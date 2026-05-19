"""Tests for the attention demo's pure helpers.

These cover the parts of ``pages/attention.py`` that don't touch Streamlit:
token lookup, head selection, and attention-row extraction. The model itself
is lightly exercised via ``get_attentions`` against GPT-2 small (already a
required dependency of the demo).
"""

import torch

from llm_sas.demos.models import MODELS, get_attentions
from llm_sas.demos.pages.attention import (
    attention_row,
    best_flipping_head,
    find_token_index,
)


def test_find_token_index_matches_case_insensitively():
    tokens = ["The", " trophy", " didn", "'t", " fit"]
    assert find_token_index(tokens, "trophy") == 1
    assert find_token_index(tokens, "TROPHY") == 1
    assert find_token_index(tokens, " trophy") == 1


def test_find_token_index_returns_none_when_missing():
    tokens = ["a", " b", " c"]
    assert find_token_index(tokens, "z") is None


def test_find_token_index_picks_last_occurrence():
    tokens = ["the", " cat", " sat", " on", " the", " mat"]
    # "the" appears twice — function returns the later index so later prompts
    # querying the pronoun "it" pick up the most recent occurrence.
    assert find_token_index(tokens, "the") == 4


def test_attention_row_returns_python_list_of_correct_length():
    # (L=2, H=2, S=3, S=3) fake attention tensor
    attn = torch.rand(2, 2, 3, 3)
    row = attention_row(attn, layer=1, head=0, query=2)
    assert isinstance(row, list)
    assert len(row) == 3
    assert row == attn[1, 0, 2].tolist()


def test_best_flipping_head_picks_obvious_flip():
    # Construct a 2-layer, 2-head attention where head (1, 0) flips cleanly.
    # Shape: (L, H, S, S) with S=4. We only set the query-row that matters.
    s = 4
    attn_a = torch.zeros(2, 2, s, s)
    attn_b = torch.zeros(2, 2, s, s)

    # query=3 attends to ref_a=1 and ref_b=2.
    # Head (0, 0): no flip — both sentences prefer ref_a.
    attn_a[0, 0, 3, 1] = 0.4
    attn_a[0, 0, 3, 2] = 0.1
    attn_b[0, 0, 3, 1] = 0.3
    attn_b[0, 0, 3, 2] = 0.1

    # Head (1, 0): clean flip — A prefers ref_a, B prefers ref_b.
    attn_a[1, 0, 3, 1] = 0.5
    attn_a[1, 0, 3, 2] = 0.1
    attn_b[1, 0, 3, 1] = 0.1
    attn_b[1, 0, 3, 2] = 0.5

    layer, head, score = best_flipping_head(attn_a, attn_b, q_a=3, q_b=3, ref_a=1, ref_b=2)
    assert (layer, head) == (1, 0)
    assert score > 0.3


def test_best_flipping_head_returns_zero_score_when_no_flip():
    s = 4
    attn_a = torch.zeros(1, 1, s, s)
    attn_b = torch.zeros(1, 1, s, s)
    # Same preference in both sentences — no flip.
    attn_a[0, 0, 3, 1] = 0.4
    attn_a[0, 0, 3, 2] = 0.1
    attn_b[0, 0, 3, 1] = 0.4
    attn_b[0, 0, 3, 2] = 0.1
    _, _, score = best_flipping_head(attn_a, attn_b, q_a=3, q_b=3, ref_a=1, ref_b=2)
    assert score == 0.0


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
