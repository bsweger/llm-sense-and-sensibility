"""
Attention mechanism demo.

Two sections, both driven by per-head self-attention weights pulled from the
selected model's forward pass:

1. **Heatmap** — pick a layer and head and inspect the full token-to-token
   attention matrix. Early layers tend to show positional/syntactic patterns
   (strong diagonals, attention to neighbors); later layers tend to show more
   semantically driven patterns.

2. **Winograd minimal pair** — two sentences differing in one word that flips
   the most likely referent of a pronoun. We auto-pick the (layer, head) that
   shows the strongest sign-flip on the candidate referents across the pair,
   then visualize that head side-by-side. The audience sees attention shift
   to track meaning when grammar stays constant.

Both sections work with every model in :mod:`llm_sas.demos.models`; layer/head
counts are read from ``model.config`` so the controls auto-fit each model.
"""

import plotly.graph_objects as go
import streamlit as st
import torch

from llm_sas.demos.models import ModelSpec, get_attentions, load_model, render_model_selector
from llm_sas.theme import HIGHLIGHT_BG, HUD_ACCENT

PRESETS: dict[str, str] = {
    "Pronoun reference": "The trophy didn't fit in the suitcase because it was too big.",
    "Subject-verb agreement": "The keys to the cabinet are on the table.",
    "Long-range dependency": "The book that the student who sat in the back read was interesting.",
    "Simple narrative": "She opened the door and walked into the room.",
}

WINOGRAD_PAIRS: dict[str, dict[str, str]] = {
    "Trophy / suitcase": {
        "a": "The trophy didn't fit in the suitcase because it was too big.",
        "b": "The trophy didn't fit in the suitcase because it was too small.",
        "ref_a": "trophy",
        "ref_b": "suitcase",
        "query_a": "big",
        "query_b": "small",
    },
    "Truck / city": {
        "a": "The truck couldn't drive through the city because it was too big.",
        "b": "The truck couldn't drive through the city because it was too small.",
        "ref_a": "truck",
        "ref_b": "city",
        "query_a": "big",
        "query_b": "small",
    },
    "Hammer / nail": {
        "a": "The hammer broke the nail because it was strong.",
        "b": "The hammer broke the nail because it was thin.",
        "ref_a": "hammer",
        "ref_b": "nail",
        "query_a": "strong",
        "query_b": "thin",
    },
}


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit calls — testable in isolation)
# ---------------------------------------------------------------------------


def find_token_index(tokens: list[str], word: str) -> int | None:
    """Last token whose decoded value equals ``word`` (case-insensitive, whitespace-stripped).

    Many tokenizers prefix whitespace to non-initial tokens (e.g. GPT-2's
    leading ``Ġ`` / a literal space). We strip and lowercase before comparing
    so the demo accepts plain words from the user.

    Returns
    -------
    int or None
        The token index, or ``None`` if the word is not in ``tokens``.
    """
    target = word.strip().lower()
    matches = [i for i, t in enumerate(tokens) if t.strip().lower() == target]
    return matches[-1] if matches else None


def best_flipping_head(
    attn_a: torch.Tensor,
    attn_b: torch.Tensor,
    q_a: int,
    q_b: int,
    ref_a: int,
    ref_b: int,
    min_gap: float = 0.02,
) -> tuple[int, int, float]:
    """Pick the (layer, head) where the dominant referent flips most cleanly.

    Score: for each head, compute the gap ``attn[q, ref_a] - attn[q, ref_b]`` in
    each sentence; the score is the smaller absolute gap (so both directions
    must be clear), and is zeroed unless the sign actually flips.

    Returns
    -------
    layer : int
    head : int
    score : float
        The min-magnitude flip score. ``0.0`` means no clean flip was found in
        any head; the caller can render the chosen head anyway and warn.
    """
    diff_a = attn_a[:, :, q_a, ref_a] - attn_a[:, :, q_a, ref_b]
    diff_b = attn_b[:, :, q_b, ref_a] - attn_b[:, :, q_b, ref_b]
    flipped = torch.sign(diff_a) != torch.sign(diff_b)
    min_mag = torch.minimum(diff_a.abs(), diff_b.abs())
    score = torch.where(
        flipped & (diff_a.abs() > min_gap) & (diff_b.abs() > min_gap),
        min_mag,
        torch.zeros_like(min_mag),
    )
    flat_idx = int(torch.argmax(score.flatten()).item())
    n_heads = score.shape[1]
    layer, head = divmod(flat_idx, n_heads)
    return layer, head, float(score[layer, head].item())


def attention_row(attn: torch.Tensor, layer: int, head: int, query: int) -> list[float]:
    """Single attention row as a plain Python list (easier to test, easier to plot)."""
    return attn[layer, head, query].tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _short_token_labels(tokens: list[str]) -> list[str]:
    """Token labels for axes: visible whitespace, position suffix to disambiguate repeats."""
    labels = []
    for i, t in enumerate(tokens):
        shown = t.replace(" ", "·").replace("\n", "↵")
        labels.append(f"{shown}<sub>{i}</sub>")
    return labels


def render_heatmap(tokens: list[str], attn_matrix: torch.Tensor) -> go.Figure:
    """Plotly heatmap for a single (layer, head). Rows = queries, columns = keys."""
    labels = _short_token_labels(tokens)
    fig = go.Figure(
        data=go.Heatmap(
            z=attn_matrix.numpy(),
            x=labels,
            y=labels,
            colorscale="Oranges",
            zmin=0.0,
            zmax=float(attn_matrix.max().item()),
            hovertemplate="query: %{y}<br>key: %{x}<br>weight: %{z:.3f}<extra></extra>",
            colorbar=dict(title="weight"),
        )
    )
    fig.update_layout(
        xaxis=dict(title="key (attended-to)", tickangle=-45, side="bottom"),
        yaxis=dict(title="query (attending-from)", autorange="reversed"),
        margin=dict(l=40, r=20, t=20, b=80),
        height=max(360, 22 * len(tokens) + 120),
    )
    return fig


def render_attention_bars(tokens: list[str], weights: list[float], highlight: dict[str, str]) -> go.Figure:
    """Horizontal bar chart of attention from one query token to every key token.

    ``highlight`` maps a token's stripped lowercase form to a fill color, used
    to color the two candidate referents distinctly from the rest.
    """
    base_color = "#cbd5e1"
    colors = [highlight.get(t.strip().lower(), base_color) for t in tokens]
    labels = _short_token_labels(tokens)
    fig = go.Figure(
        data=go.Bar(
            x=weights,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title="attention weight", range=[0, max(weights) * 1.1 if weights else 1]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=max(280, 22 * len(tokens) + 80),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
st.title("Attention")
st.markdown(
    """
    Each transformer layer mixes information across tokens using **self-attention**:
    every token computes a weighted sum over every other token in the sequence.
    The weights — the *attention pattern* — are how the model routes context.

    A model has many *heads* per layer. Different heads learn different routing
    strategies: some track positional patterns, some link related words.
    """
)

spec: ModelSpec = render_model_selector(
    key="attention_model",
    help=(
        "Different models have different layer/head counts and different learned routing "
        "strategies. Switching models recomputes the attention pattern and re-fits the controls."
    ),
)
with st.spinner(f"Loading {spec.model_id}…"):
    _, model = load_model(spec)
n_layers = int(model.config.num_hidden_layers)
n_heads = int(model.config.num_attention_heads)

st.caption(f"Loaded **{spec.model_id}** — {n_layers} layers × {n_heads} heads per layer.")

# ---------------------------------------------------------------------------
# Section 1: interactive heatmap
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### How tokens attend to each other")
st.markdown(
    """
    Pick a sentence (or write your own), then move the **layer** and **head** sliders
    to see how a single attention head routes information across the sequence.

    - Each row is a *query* token; each column is a *key* token.
    - Brighter cells = stronger attention from the row to the column.
    - The strict upper-triangle is empty: this is a causal LM, so a token can only
      attend to itself and the tokens before it.
    """
)

def _apply_preset() -> None:
    """Push the selected preset's text into the text-input's session state.

    Needed because ``st.text_input(value=..., key=...)`` ignores ``value`` after
    the widget has been registered — once a key is in session state, that wins.
    Updating session state directly inside the selectbox's ``on_change`` is the
    documented way to drive one widget from another.
    """
    label = st.session_state.attention_preset
    if label in PRESETS:
        st.session_state.attention_text = PRESETS[label]


# Seed the text-input's session state on first render so the widget has
# something to show before the user has touched anything.
if "attention_text" not in st.session_state:
    st.session_state.attention_text = PRESETS["Pronoun reference"]

st.selectbox(
    "Preset sentence",
    list(PRESETS.keys()) + ["Custom"],
    key="attention_preset",
    on_change=_apply_preset,
)
text = st.text_input("Sentence", key="attention_text")

col_layer, col_head = st.columns(2)
with col_layer:
    layer = st.slider("Layer", 0, n_layers - 1, value=min(4, n_layers - 1), key="attention_layer")
with col_head:
    head = st.slider("Head", 0, n_heads - 1, value=min(3, n_heads - 1), key="attention_head")

if text.strip():
    tokens, attn = get_attentions(text, spec)
    fig = render_heatmap(tokens, attn[layer, head])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Layer {layer}, head {head}. Try layer 0 (often a strong diagonal — local positional attention) "
        "vs. a middle/late layer (often longer-range, more semantic-looking)."
    )
else:
    st.info("Type a sentence above to see its attention pattern.")

# ---------------------------------------------------------------------------
# Section 2: Winograd minimal pair
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Attention follows meaning, not just grammar")
st.markdown(
    """
    These two sentences are identical except for **one word at the end** — but that
    word flips which earlier noun the pronoun refers to. We auto-pick the head
    whose attention pattern flips most cleanly across the pair, then plot it.

    What to watch: in each sentence we query attention from the **disambiguating
    word** back to every token. The two highlighted bars are the candidate
    referents. When the pair flips, the dominant bar should flip with it.
    """
)

pair_label = st.selectbox("Sentence pair", list(WINOGRAD_PAIRS.keys()), key="attention_pair")
pair = WINOGRAD_PAIRS[pair_label]

tokens_a, attn_a = get_attentions(pair["a"], spec)
tokens_b, attn_b = get_attentions(pair["b"], spec)

q_a = find_token_index(tokens_a, pair["query_a"])
q_b = find_token_index(tokens_b, pair["query_b"])
ref_a_idx_in_a = find_token_index(tokens_a, pair["ref_a"])
ref_b_idx_in_a = find_token_index(tokens_a, pair["ref_b"])
ref_a_idx_in_b = find_token_index(tokens_b, pair["ref_a"])
ref_b_idx_in_b = find_token_index(tokens_b, pair["ref_b"])

required = [q_a, q_b, ref_a_idx_in_a, ref_b_idx_in_a, ref_a_idx_in_b, ref_b_idx_in_b]
if any(idx is None for idx in required):
    st.warning(
        "Couldn't locate the query or referent tokens after tokenization with this model. "
        "Try a different sentence pair or model."
    )
else:
    # Auto-pick the cleanest flipping head for this model + pair.
    # Use the same referent index in both sentences (they share the prefix), so look up in tokens_a.
    layer_w, head_w, score = best_flipping_head(attn_a, attn_b, q_a, q_b, ref_a_idx_in_a, ref_b_idx_in_a)

    if score == 0.0:
        st.info(
            f"No head in **{spec.model_id}** cleanly flips the dominant referent on this pair. "
            f"Showing the layer-{layer_w}, head-{head_w} attempt anyway — read the bars carefully, "
            "the signal will be subtle (or absent) in small models."
        )
    else:
        st.markdown(
            f"Auto-selected **layer {layer_w}, head {head_w}** "
            f"<span style='color:{HUD_ACCENT}'>(flip score: {score:.3f})</span>",
            unsafe_allow_html=True,
        )

    override = st.checkbox("Override head selection", key="attention_override")
    if override:
        col_l, col_h = st.columns(2)
        with col_l:
            layer_w = st.slider("Layer (override)", 0, n_layers - 1, value=layer_w, key="attention_override_layer")
        with col_h:
            head_w = st.slider("Head (override)", 0, n_heads - 1, value=head_w, key="attention_override_head")

    highlight = {pair["ref_a"].lower(): HUD_ACCENT, pair["ref_b"].lower(): HIGHLIGHT_BG}

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Sentence A:** *{pair['a']}*")
        st.caption(f"Querying from `{pair['query_a']}` at position {q_a}.")
        weights_a = attention_row(attn_a, layer_w, head_w, q_a)
        st.plotly_chart(render_attention_bars(tokens_a, weights_a, highlight), use_container_width=True)
    with col_b:
        st.markdown(f"**Sentence B:** *{pair['b']}*")
        st.caption(f"Querying from `{pair['query_b']}` at position {q_b}.")
        weights_b = attention_row(attn_b, layer_w, head_w, q_b)
        st.plotly_chart(render_attention_bars(tokens_b, weights_b, highlight), use_container_width=True)

    st.caption(
        f"Highlighted bars: **{pair['ref_a']}** (orange) and **{pair['ref_b']}** (peach). "
        "Same grammar, same vocabulary except for one word — and the model's attention shifts to match."
    )
