"""
Part 1 — Next-token prediction.

Interactive demo: step through a sentence one token at a time. Top-K candidates
are hidden by default; click "Reveal candidates" to show them, then pick one to
extend the sentence.
"""
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

TOP_K = 5
BAR = "█"
BAR_W = 30
DEFAULT_PROMPT = "The cat sat on the"


@st.cache_resource(show_spinner="Loading GPT-2 (one-time, ~500MB)…")
def load_model():
    """Load GPT-2 model and tokenizer. Cached across reruns and sessions."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def get_candidates(token_ids, tokenizer, model):
    """Return top-K (label, token_id, percent) tuples for the next token."""
    if not token_ids:
        return []
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, TOP_K)
    return [
        (
            tokenizer.decode([tid]).strip() or repr(tokenizer.decode([tid])),
            int(tid),
            float(prob) * 100,
        )
        for prob, tid in zip(top_probs.tolist(), top_ids.tolist())
    ]


def init_state():
    """Initialize session state on first run."""
    if "token_ids" not in st.session_state:
        st.session_state.token_ids = None  # populated after model loads
    if "revealed" not in st.session_state:
        st.session_state.revealed = False
    if "prompt_value" not in st.session_state:
        st.session_state.prompt_value = DEFAULT_PROMPT


def reset_to_prompt(tokenizer, prompt_text):
    """Reset the running sentence to the given prompt."""
    st.session_state.token_ids = tokenizer.encode(prompt_text) if prompt_text.strip() else []
    st.session_state.revealed = False
    st.session_state.prompt_value = prompt_text


def pick_token(token_id):
    """Append a token to the running sentence and re-hide candidates."""
    st.session_state.token_ids = st.session_state.token_ids + [token_id]
    st.session_state.revealed = False


def reveal():
    """Reveal the hidden candidates."""
    st.session_state.revealed = True


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
    """Render the top-K candidates — either hidden placeholders or revealed bars."""
    if not candidates:
        st.info("Enter a prompt above to see candidates.")
        return

    if not st.session_state.revealed:
        rows = "<br>".join(
            f"<span style='font-family:monospace;color:#9ca3af'>"
            f"{i + 1}. ?{'░' * BAR_W:>{BAR_W + 4}}   ?.?%</span>"
            for i in range(len(candidates))
        )
        st.markdown(
            f"**Top candidates for next token:**<br>{rows}",
            unsafe_allow_html=True,
        )
        st.caption("Candidates hidden — take audience guesses, then click 'Reveal candidates'.")
        return

    max_pct = candidates[0][2]
    rows = []
    for i, (label, _tid, pct) in enumerate(candidates):
        bar = BAR * round(pct / max_pct * BAR_W)
        weight = "bold" if i == 0 else "normal"
        color = "#15803d" if i == 0 else "#1f2937"
        rows.append(
            f"<span style='font-family:monospace;font-weight:{weight};color:{color};font-size:15px'>"
            f"{i + 1}. {label:<14} {bar:<{BAR_W}} {pct:>5.1f}%</span>"
        )
    st.markdown(
        "**Top candidates for next token:**<br>" + "<br>".join(rows),
        unsafe_allow_html=True,
    )


def render_action_buttons(candidates, tokenizer):
    """Reveal button, pick buttons, reset button."""
    # Reveal button — disabled once revealed.
    col_reveal, col_pick_top, col_reset = st.columns([1, 1, 1])
    with col_reveal:
        st.button(
            "Reveal candidates",
            on_click=reveal,
            disabled=st.session_state.revealed or not candidates,
            type="primary" if not st.session_state.revealed else "secondary",
            use_container_width=True,
        )
    with col_pick_top:
        if candidates and st.session_state.revealed:
            st.button(
                f"Pick most likely: {candidates[0][0]}",
                on_click=pick_token,
                args=(candidates[0][1],),
                type="primary",
                use_container_width=True,
            )
        else:
            st.button(
                "Pick most likely",
                disabled=True,
                use_container_width=True,
            )
    with col_reset:
        st.button(
            "Reset to prompt",
            on_click=reset_to_prompt,
            args=(tokenizer, st.session_state.prompt_value),
            use_container_width=True,
        )

    # Individual pick buttons appear only when revealed.
    if candidates and st.session_state.revealed:
        st.markdown("**Or pick any candidate:**")
        cols = st.columns(len(candidates))
        for i, (label, tid, _pct) in enumerate(candidates):
            with cols[i]:
                st.button(
                    label,
                    key=f"pick_{i}_{tid}",
                    on_click=pick_token,
                    args=(tid,),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
st.title("Part 1 — Text as Prediction")
st.markdown(
    """
    LLMs are next-token predictors. At each step the model produces a probability
    distribution over the vocabulary and samples from it. Step through the sentence
    below to see this token by token.
    """
)

init_state()
tokenizer, model = load_model()

# Seed token_ids from the default prompt on the very first run.
if st.session_state.token_ids is None:
    st.session_state.token_ids = tokenizer.encode(DEFAULT_PROMPT)

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
candidates = get_candidates(st.session_state.token_ids, tokenizer, model)

st.markdown("")
render_candidates(candidates)

st.markdown("")
render_action_buttons(candidates, tokenizer)
