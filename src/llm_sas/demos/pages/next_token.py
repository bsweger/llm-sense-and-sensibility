"""
Next-token prediction demo.

Interactive demo: step through a sentence one token at a time. The sidebar
toggles between greedy decoding (always pick argmax) and sampling, with
controls for temperature and top-k filtering.
"""

import streamlit as st
import torch

from llm_sas.demos.models import load_model, render_model_link, render_model_selector
from llm_sas.theme import BODY_TEXT, HIGHLIGHT_BG, HUD_ACCENT

TOP_K = 10  # number of candidate bars to display
BAR = "█"
BAR_W = 30
DEFAULT_PROMPT = "I like to"


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
        (changes the denominator used in the softmax calculation)

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
    _, model = load_model(st.session_state.active_model)
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


def greedy_token_id(token_ids, model):
    """Return the argmax of the natural (unfiltered) next-token distribution. Pure."""
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    return int(torch.argmax(logits).item())


def pick_next_token():
    """Streamlit on_click shim: read state, call greedy_token_id, append picked token.

    Recomputing the argmax inside the handler — rather than passing a pre-bound
    token id via ``args=`` — keeps rapid double-clicks correct. If the id were
    bound at render time, a second click that landed before the rerun completed
    would re-fire with the *previous* frame's id and append the same token
    twice. Reading current state here means each click advances by exactly one
    step against the latest sentence.
    """
    _, model = load_model(st.session_state.active_model)
    token_ids = st.session_state.token_ids
    if not token_ids:
        return
    st.session_state.token_ids = token_ids + [greedy_token_id(token_ids, model)]


def generate_token_ids(token_ids, model, sampling_on, temperature, top_k, n_new_tokens):
    """Run ``n_new_tokens`` decoding steps and return the extended token list.

    One step is exactly what the per-click buttons do today — a forward pass
    plus a decoding decision (argmax for greedy, multinomial draw from the
    filtered distribution for sampling). Pure — easy to unit-test.
    """
    ids = list(token_ids)
    for _ in range(n_new_tokens):
        if sampling_on:
            ids.append(sample_token_id(ids, model, temperature, top_k))
        else:
            ids.append(greedy_token_id(ids, model))
    return ids


def generate_next_tokens():
    """Streamlit on_click shim: run ``max_tokens`` decoding steps in a row."""
    _, model = load_model(st.session_state.active_model)
    token_ids = st.session_state.token_ids
    if not token_ids:
        return
    st.session_state.token_ids = generate_token_ids(
        token_ids,
        model,
        sampling_on=(st.session_state.sampling_mode == "Sampling"),
        temperature=st.session_state.temperature,
        top_k=st.session_state.top_k,
        n_new_tokens=st.session_state.max_tokens,
    )


def render_sentence(tokenizer):
    """Render the running sentence with the most recent token highlighted."""
    token_ids = st.session_state.token_ids or []
    if not token_ids:
        st.markdown("**Sentence so far:** *(empty)*")
        return
    text = tokenizer.decode(token_ids)
    toks = text.split()
    hl = " ".join(
        f"<mark style='background:{HIGHLIGHT_BG};padding:2px 6px;border-radius:4px'>{t}</mark>"
        if i == len(toks) - 1
        else t
        for i, t in enumerate(toks)
    )
    st.markdown(
        f"**Sentence so far:**<br><span style='font-family:monospace;font-size:18px;line-height:2.2'>{hl}</span>",
        unsafe_allow_html=True,
    )


def _candidate_rows_html(candidates, scale_max_pct, header, label_w, bar_w):
    """Build the HTML for one column of candidate bars. Pure — no Streamlit calls."""
    rows = []
    for i, (label, _tid, pct) in enumerate(candidates):
        bar = BAR * round(pct / scale_max_pct * bar_w) if scale_max_pct > 0 else ""
        weight = "bold" if i == 0 else "normal"
        color = HUD_ACCENT if i == 0 else BODY_TEXT
        rows.append(
            f"<span style='font-family:monospace;white-space:pre;font-weight:{weight};color:{color};font-size:15px'>"
            f"{i + 1:>2}. {label:<{label_w}} {bar:<{bar_w}} {pct:>5.1f}%</span>"
        )
    return f"**{header}:**<br>" + "<br>".join(rows)


def render_candidates(natural, filtered, sampling_on, temperature, top_k):
    """Render the candidate bars. Single column in greedy mode, side-by-side when sampling."""
    if not natural:
        st.info("Enter a prompt above to see candidates.")
        return

    if not sampling_on:
        st.markdown(
            _candidate_rows_html(natural, natural[0][2], "Top candidates for next token", 14, BAR_W),
            unsafe_allow_html=True,
        )
        return

    # Each column normalized to its own top, so the Natural column stays visually
    # stable as the sliders move (it's a fixed reference). Compare absolute
    # magnitudes by reading the printed percentages; compare shapes by eye.
    label_w, bar_w = 12, 15  # narrower bars so two columns fit in the centered layout

    col_natural, col_filtered = st.columns(2)
    with col_natural:
        st.markdown(
            _candidate_rows_html(natural, natural[0][2], "Natural (T=1.0)", label_w, bar_w),
            unsafe_allow_html=True,
        )
    with col_filtered:
        filter_header = f"After filters (T={temperature}, top-k={top_k})"
        if filtered:
            st.markdown(
                _candidate_rows_html(filtered, filtered[0][2], filter_header, label_w, bar_w),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{filter_header}:** *(no candidates)*")


def render_generation_controls():
    """Render sidebar widgets and return (sampling_on, temperature, top_k, max_tokens)."""
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
        max_tokens = st.slider(
            "Tokens to generate per click",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            key="max_tokens",
            help=(
                "How many tokens to produce in one click. At 1 (default) you advance one "
                "token at a time. At N>1 the same decoding strategy runs in a loop N times — "
                "every step makes its own forward pass and its own decoding decision."
            ),
        )

    return sampling_on, temperature, top_k, max_tokens


def render_action_buttons(candidates, tokenizer, sampling_on, max_tokens):
    """Primary action (sample, pick-most-likely, or generate N) and reset button."""
    col_action, col_reset = st.columns([1, 1])
    with col_action:
        if max_tokens > 1:
            st.button(
                f"Generate next {max_tokens} tokens",
                on_click=generate_next_tokens,
                disabled=not candidates,
                type="primary",
                use_container_width=True,
            )
        elif sampling_on:
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
                on_click=pick_next_token,
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
st.title("Inference: Text as Prediction")
st.markdown(
    """
    LLMs are next-token predictors. At each step the model produces a probability
    distribution over the vocabulary. Step through the sentence
    below to see this token by token.
    """
)

init_state()
spec = render_model_selector(
    key="next_token_model",
    help=(
        "Different models were trained on different data and produce different "
        "distributions for the same prompt. Switching models resets the running "
        "sentence because tokenizers differ between model families."
    ),
)
with st.spinner(f"Loading {spec.model_id}…"):
    tokenizer, model = load_model(spec)

# Seed token_ids on first run, and re-tokenize whenever the user switches models
# (token ids are not portable across tokenizers).
if st.session_state.token_ids is None or st.session_state.get("active_model") != spec:
    st.session_state.active_model = spec
    st.session_state.token_ids = tokenizer.encode(st.session_state.prompt_value or DEFAULT_PROMPT)

sampling_on, temperature, top_k, max_tokens = render_generation_controls()

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
# Always show the model's natural (T=1, unfiltered) distribution as the reference.
# In sampling mode, also show the post-filter distribution side-by-side so the
# audience can see what the sampler will actually draw from.
natural_candidates = get_candidates(st.session_state.token_ids, tokenizer, model)
if sampling_on:
    filtered_candidates = get_candidates(
        st.session_state.token_ids,
        tokenizer,
        model,
        temperature=temperature,
        top_k=top_k,
    )
else:
    filtered_candidates = natural_candidates

st.markdown("")
render_candidates(natural_candidates, filtered_candidates, sampling_on, temperature, top_k)

# The primary "Sample"/"Pick most likely" button operates on the filtered distribution
# in sampling mode and the natural one in greedy mode.
action_candidates = filtered_candidates if sampling_on else natural_candidates
st.markdown("")
render_action_buttons(action_candidates, tokenizer, sampling_on, max_tokens)

render_model_link(spec)
