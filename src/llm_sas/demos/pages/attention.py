"""
Attention mechanism demo.

Pick a sentence, a focus word, and a layer. The page renders the sentence as
a horizontal row of tokens with arcs from the focus word back to every earlier
token, with arc opacity proportional to the attention weight at the chosen
layer (averaged across heads). The arc form was chosen over a heatmap because
the heatmap's row/column distinction and empty upper-triangle were intimidating
to non-technical viewers.

The page works with every model in :mod:`llm_sas.demos.models`; layer counts
are read from ``model.config`` so the slider auto-fits each model. The head
dimension is hidden from the audience — heads are an internal detail in this
high-level talk.
"""

import plotly.graph_objects as go
import streamlit as st

from llm_sas.demos.models import ModelSpec, get_attentions, load_model, render_model_selector

PRESETS: dict[str, str] = {
    "Pronoun reference": "The trophy didn't fit in the suitcase because it was too big.",
    "Subject-verb agreement": "The keys to the cabinet are on the table.",
    "Long-range dependency": "The book that the student who sat in the back read was interesting.",
    "Simple narrative": "She opened the door and walked into the room.",
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def render_attention_arcs(
    tokens: list[str],
    weights: list[float],
    focus_idx: int,
    hide_sink: bool = True,
) -> go.Figure:
    """Render the sentence as a vertical column of tokens with arcs from the focus token.

    Tokens stack top-to-bottom (token 0 at the top), left-aligned. The focus
    token is rendered in bold. Arcs sweep to the left of the column from the
    focus token to every other token, with opacity proportional to the
    attention weight. Future-token weights are zero in a causal LM, so those
    arcs don't draw.

    Parameters
    ----------
    hide_sink : bool
        If True (default), suppress the arc to the first token. Many attention
        heads dump residual attention onto the first token as a "sink," and
        averaging across heads makes that sink dominate the visual. Hiding it
        produces a cleaner picture of what the rest of the model is doing.
    """
    n = len(tokens)
    # Token i sits at y = -i so token 0 is at the top of the figure.
    ys = [-i for i in range(n)]

    display = []
    for i, t in enumerate(tokens):
        shown = t.strip() or t.replace(" ", "·").replace("\n", "↵") or "·"
        display.append(f"<b>{shown}</b>" if i == focus_idx else shown)

    fig = go.Figure()
    arc_anchor_x = -0.05
    fig.add_trace(
        go.Scatter(
            x=[0] * n,
            y=ys,
            mode="text",
            text=display,
            textposition="middle right",  # text extends right from x=0 → left-aligned column
            textfont=dict(size=16),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    def _is_drawable(i: int) -> bool:
        if i == focus_idx:
            return False
        if hide_sink and i == 0 and focus_idx != 0:
            return False
        return True

    drawable_weights = [w for i, w in enumerate(weights) if _is_drawable(i) and w > 0]
    max_w = max(drawable_weights) if drawable_weights else 1.0
    max_peak_x_offset = 0.0
    for i, w in enumerate(weights):
        if not _is_drawable(i) or w <= 0:
            continue
        peak_x_offset = 0.4 + 0.15 * min(abs(focus_idx - i), 12)
        max_peak_x_offset = max(max_peak_x_offset, peak_x_offset)
        control_x = arc_anchor_x - peak_x_offset * 2
        y_s = -focus_idx
        y_t = -i
        y_mid = (y_s + y_t) / 2
        # Encode attention weight twice — opacity (fades low-attention arcs out) and
        # line thickness (makes high-attention arcs visually pop). Both ramp from
        # minimums when w is tiny to full strength when w == max_w.
        w_norm = w / max_w if max_w > 0 else 0
        fig.add_shape(
            type="path",
            path=f"M {arc_anchor_x},{y_s} Q {control_x},{y_mid} {arc_anchor_x},{y_t}",
            line=dict(color="#9a3412", width=1 + 6 * w_norm),
            opacity=min(1.0, w_norm),
        )

    fig.update_layout(
        xaxis=dict(visible=False, range=[arc_anchor_x - max_peak_x_offset - 0.3, 5]),
        yaxis=dict(visible=False, range=[-(n - 1) - 0.5, 0.5]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(280, 32 * n + 40),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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

    Different layers learn different routing strategies: early layers tend to
    focus on nearby tokens, deeper layers link more distant, semantically
    related ones.
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

st.markdown("---")
st.markdown("### How tokens attend to each other")
st.markdown(
    """
    Pick a sentence (or write your own), then pick a **focus word**. The arcs
    show which earlier words the focus word "paid attention to" inside the
    model — darker arcs mean stronger attention. Move the **layer** slider to
    see how the focus word's attention shifts at different depths of the
    model.
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

# Sliders are 1-indexed for display (friendlier for a non-technical audience);
# the tensor is 0-indexed, so we subtract 1 before using the value internally.
layer_display = st.slider("Layer", 1, n_layers, value=min(5, n_layers), key="attention_layer")
layer = layer_display - 1

if text.strip():
    tokens, attn = get_attentions(text, spec)
    # Use indices as the dropdown's underlying options so duplicate-token sentences
    # ("the trophy ... the suitcase") still pick exactly one row; the labels stay bare.
    # Default focus is the penultimate token — the last is usually a period.
    default_focus = max(0, len(tokens) - 2)
    focus_idx = st.selectbox(
        "Focus word",
        options=list(range(len(tokens))),
        index=default_focus,
        format_func=lambda i: tokens[i].strip() or repr(tokens[i]),
    )
    show_sink = st.checkbox(
        "Show attention to the first token",
        value=False,
        help=(
            "Transformer heads often dump residual attention onto the first token of the "
            "sequence as a 'sink' — a known artifact. Hiding it (the default) usually makes "
            "the rest of the attention pattern much easier to see."
        ),
    )
    layer_attn = attn[layer].mean(dim=0)
    weights = layer_attn[focus_idx].tolist()
    fig = render_attention_arcs(tokens, weights, focus_idx, hide_sink=not show_sink)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Layer {layer_display} of {n_layers}. Try layer 1 (the model is mostly looking at recent tokens) "
        "vs. a deeper layer (longer-range connections)."
    )
else:
    st.info("Type a sentence above to see its attention pattern.")

st.markdown("---")
with st.expander("Under the hood"):
    st.markdown(
        f"""
        **Heads.** Each transformer layer doesn't compute one attention
        pattern — it computes several in parallel, called *heads*. Each head
        learns its own routing strategy: some attend to immediate neighbors,
        some look for grammatical agreement, some link semantically related
        words. The model loaded above ({spec.model_id}) has **{n_heads}
        heads** per layer across **{n_layers} layers**.

        Heads are an internal detail in this talk, so the arcs above the
        sentence show the **mean attention across all {n_heads} heads** at
        the selected layer. That gives a single, audience-friendly summary
        of "what does this layer do" without having to pick a specific head.
        The early-vs-late layer story (positional → semantic) still comes
        through clearly in the average.

        **The first-token attention sink.** Trained transformers tend to
        dump a large share of their attention onto the very first token of
        the sequence — regardless of what that token is. This is a
        well-documented phenomenon called an *attention sink* (Xiao et al.,
        2023). The intuition: softmax forces every attention head's weights
        to sum to 1, so when a head doesn't strongly need to attend to
        anything in particular, the leftover mass gets dumped on whatever
        token is always available — typically the first one. Averaging
        across heads then makes that sink dominate the visual.

        To keep the demo readable, we **hide the arc to the first token by
        default**. The "Show attention to the first token" checkbox above
        the chart toggles it back on if you want to see the raw pattern. The
        underlying attention values aren't changed — only the rendering.
        """
    )
