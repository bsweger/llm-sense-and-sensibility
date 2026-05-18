"""
Demo 1 — Next-token prediction.

Interactive demo: step through a sentence one token at a time. The sidebar
toggles between greedy decoding (always pick argmax) and sampling, with
controls for temperature and top-k filtering.
"""
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

TOP_K = 5  # number of candidate bars to display
BAR = "█"
BAR_W = 30
DEFAULT_PROMPT = "I like to think"


@st.cache_resource(show_spinner="Loading GPT-2 (one-time, ~500MB)…")
def load_model():
    """Load GPT-2 model and tokenizer. Cached across reruns and sessions."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def apply_sampling_filters(logits, temperature, top_k):
    """Apply temperature scaling and top-k truncation to logits.

    Parameters
    ----------
    logits : torch.Tensor
        1-D tensor of raw logits over the vocabulary.
    temperature : float
        Divisor for logits. < 1.0 sharpens the distribution; > 1.0 flattens it.
    top_k : int
        If > 0, keep only the top-k highest logits; the rest are set to -inf.

    Returns
    -------
    torch.Tensor
        Modified logits, same shape as input. Filtered-out positions get -inf,
        which becomes 0 probability after softmax.
    """
    logits = logits / max(temperature, 1e-9)

    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    return logits


def get_candidates(token_ids, tokenizer, model, temperature=1.0, top_k=0):
    """Return top-K (label, token_id, percent) tuples after applying sampling filters."""
    if not token_ids:
        return []
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        # model(input_ids).logits is a tensor with a raw score for *every*
        # token in the model's vocabulary for *every* input token.
        # line below slices the logits as follows:
        # 0: the first batch (in this example, there's only a single batch)
        # -1: get predictions for the last token in the sequence only
        # : include all vocab entries
        logits = model(input_ids).logits[0, -1, :]
    # Defaults (T=1.0, top_k=0) leave logits unchanged → natural distribution.
    logits = apply_sampling_filters(logits, temperature, top_k)
    # softmax normalizes logits into an actual probability distribution.
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, TOP_K)
    return [
        (
            tokenizer.decode([tid]).strip() or repr(tokenizer.decode([tid])),
            int(tid),
            float(prob) * 100,
        )
        for prob, tid in zip(top_probs.tolist(), top_ids.tolist())
        if prob > 0  # filter out zero-prob tokens left over from top-k truncation
    ]


def init_state():
    """Initialize session state on first run."""
    if "token_ids" not in st.session_state:
        st.session_state.token_ids = None  # populated after model loads
    if "prompt_value" not in st.session_state:
        st.session_state.prompt_value = DEFAULT_PROMPT


def reset_to_prompt(tokenizer, prompt_text):
    """Reset the running sentence to the given prompt."""
    st.session_state.token_ids = tokenizer.encode(prompt_text) if prompt_text.strip() else []
    st.session_state.prompt_value = prompt_text


def pick_token(token_id):
    """Append a token to the running sentence."""
    st.session_state.token_ids = st.session_state.token_ids + [token_id]


def sample_token_id(token_ids, model, temperature, top_k):
    """Sample one next-token id from the (filtered) distribution. Pure — easy to unit-test."""
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    logits = apply_sampling_filters(logits, temperature, top_k)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def sample_next_token():
    """Streamlit on_click shim: read state, call sample_token_id, append sampled token."""
    _, model = load_model()
    token_ids = st.session_state.token_ids
    if not token_ids:
        return
    sampled = sample_token_id(
        token_ids,
        model,
        st.session_state.temperature,
        st.session_state.top_k,
    )
    st.session_state.token_ids = token_ids + [sampled]


def render_sentence(tokenizer):
    """Render the running sentence with the most recent token highlighted."""
    token_ids = st.session_state.token_ids or []
    if not token_ids:
        st.markdown("**Sentence so far:** *(empty)*")
        return
    text = tokenizer.decode(token_ids)
    toks = text.split()
    hl = " ".join(
        f"<mark style='background:#dbeafe;padding:2px 6px;border-radius:4px'>{t}</mark>"
        if i == len(toks) - 1
        else t
        for i, t in enumerate(toks)
    )
    st.markdown(
        f"**Sentence so far:**<br>"
        f"<span style='font-family:monospace;font-size:18px;line-height:2.2'>{hl}</span>",
        unsafe_allow_html=True,
    )


def render_candidates(candidates):
    """Render the top-K candidates as bars."""
    if not candidates:
        st.info("Enter a prompt above to see candidates.")
        return

    max_pct = candidates[0][2]
    rows = []
    for i, (label, _tid, pct) in enumerate(candidates):
        bar = BAR * round(pct / max_pct * BAR_W)
        weight = "bold" if i == 0 else "normal"
        color = "#15803d" if i == 0 else "#1f2937"
        rows.append(
            f"<span style='font-family:monospace;white-space:pre;font-weight:{weight};color:{color};font-size:15px'>"
            f"{i + 1}. {label:<14} {bar:<{BAR_W}} {pct:>5.1f}%</span>"
        )
    st.markdown(
        "**Top candidates for next token:**<br>" + "<br>".join(rows),
        unsafe_allow_html=True,
    )


def render_sampling_controls():
    """Render sidebar widgets and return (sampling_on, temperature, top_k)."""
    with st.sidebar:
        st.header("Next token controls")
        mode = st.radio(
            "Mode",
            ["Greedy", "Sampling"],
            key="sampling_mode",
            horizontal=True,
            help=(
                "**Greedy** always picks the model's top choice — same result every time, "
                "can feel repetitive. **Sampling** picks randomly, with more-likely words "
                "winning more often — adds variety and surprise."
            ),
        )
        sampling_on = mode == "Sampling"

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="temperature",
            disabled=not sampling_on,
            # Temperature values > 1 flatten the probability distribution for the next
            # likely token (i.e, the more unlikely tokens get a larger slice of the
            # distribution than they normall would))
            help=(
                'A "creativity dial. Lower values make the model play it safe and stick to '
                "obvious choices. Higher values make it take chances and pick unusual words. "
                "1.0 is the model's natural setting."
            ),
        )
        top_k = st.slider(
            "Top-k (0 = off)",
            min_value=0,
            max_value=200,
            value=50,
            step=1,
            key="top_k",
            disabled=not sampling_on,
            help=(
                "Only let the top K most-likely words compete; ignore the rest. Smaller K "
                "keeps choices focused on safe bets. 0 turns the filter off (every word in "
                "the vocabulary is eligible)."
            ),
        )

    return sampling_on, temperature, top_k


def render_action_buttons(candidates, tokenizer, sampling_on):
    """Primary action (sample or pick-most-likely) and reset button."""
    col_action, col_reset = st.columns([1, 1])
    with col_action:
        if sampling_on:
            st.button(
                "Sample next token",
                on_click=sample_next_token,
                disabled=not candidates,
                type="primary",
                use_container_width=True,
            )
        elif candidates:
            st.button(
                f"Pick most likely: {candidates[0][0]}",
                on_click=pick_token,
                args=(candidates[0][1],),
                type="primary",
                use_container_width=True,
            )
        else:
            st.button("Pick most likely", disabled=True, use_container_width=True)
    with col_reset:
        st.button(
            "Reset to prompt",
            on_click=reset_to_prompt,
            args=(tokenizer, st.session_state.prompt_value),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
st.title("Demo 1 — Text as Prediction")
st.markdown(
    """
    LLMs are next-token predictors. At each step the model produces a probability
    distribution over the vocabulary. Step through the sentence
    below to see this token by token.
    """
)

init_state()
tokenizer, model = load_model()

# Seed token_ids from the default prompt on the very first run.
if st.session_state.token_ids is None:
    st.session_state.token_ids = tokenizer.encode(DEFAULT_PROMPT)

sampling_on, temperature, top_k = render_sampling_controls()

# Prompt input — pressing Enter or clicking outside commits a new starting prompt.
prompt_text = st.text_input(
    "Starting prompt",
    value=st.session_state.prompt_value,
    key="prompt_input_widget",
)
if prompt_text != st.session_state.prompt_value:
    reset_to_prompt(tokenizer, prompt_text)
    st.rerun()

st.markdown("---")

render_sentence(tokenizer)
# In greedy mode, show the natural (unfiltered) distribution; in sampling mode,
# show what the sampler will actually draw from so the audience can watch the
# bars change as temperature/top-k change.
candidates = get_candidates(
    st.session_state.token_ids,
    tokenizer,
    model,
    temperature=temperature if sampling_on else 1.0,
    top_k=top_k if sampling_on else 0,
)

st.markdown("")
render_candidates(candidates)

st.markdown("")
render_action_buttons(candidates, tokenizer, sampling_on)
