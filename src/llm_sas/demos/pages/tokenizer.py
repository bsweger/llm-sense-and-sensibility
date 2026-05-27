"""
Tokenization demo.

Show how a model converts a prompt into integer token ids, and what each id
decodes back to as a string. Different tokenizers slice the same text very
differently — switch the model in the sidebar to compare.
"""

from textwrap import dedent

import streamlit as st

from llm_sas.demos.models import load_tokenizer, render_model_link, render_model_selector
from llm_sas.theme import BODY_TEXT, TOKEN_COLORS

DEFAULT_PROMPT = dedent("""\
    "I don't want to be human."

    Dr. Mensah said, "That's not an attitude a lot of humans are going to understand. We tend to think that because a bot or a construct looks human, its ultimate goal would be to become human."

    "That's the dumbest thing I've ever heard."
    """)


def tokenize(tokenizer, prompt: str) -> list[tuple[int, int, str, str]]:
    """Tokenize a prompt and return per-token detail.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        Any Hugging Face tokenizer.
    prompt : str
        The input text to tokenize.

    Returns
    -------
    list of (position, token_id, raw_piece, decoded_piece) tuples
        - ``raw_piece`` is what ``convert_ids_to_tokens`` returns. For BPE
          tokenizers this exposes the leading-space marker (e.g. GPT-2's "Ġ").
        - ``decoded_piece`` is what ``decode([id])`` returns: the actual text,
          including any leading whitespace that is part of the token.
    """
    if not prompt:
        return []
    token_ids = tokenizer.encode(prompt)
    raw_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    return [(i, int(tid), raw, tokenizer.decode([tid])) for i, (tid, raw) in enumerate(zip(token_ids, raw_pieces))]


def _visible(text: str) -> str:
    """Make whitespace and newlines visible inside an HTML pill. Pure."""
    return text.replace(" ", "·").replace("\n", "↵").replace("\t", "→")


def format_inline_html(rows: list[tuple[int, int, str, str]], colors: list[str] = TOKEN_COLORS) -> str:
    """Render each token as a colored pill with its id underneath. Pure — no Streamlit calls."""
    pills = []
    for i, tid, _raw, decoded in rows:
        bg = colors[i % len(colors)]
        pills.append(
            f"<span style='display:inline-block;background:{bg};"
            f"padding:4px 8px;margin:3px 2px;border-radius:4px;"
            f"font-family:monospace;line-height:1.4;vertical-align:top;'>"
            f"<span style='font-size:15px;color:{BODY_TEXT};white-space:pre'>{_visible(decoded)}</span>"
            f"<span style='display:block;font-size:10px;color:{BODY_TEXT};text-align:center'>{tid}</span>"
            f"</span>"
        )
    return "<div style='line-height:1.8'>" + "".join(pills) + "</div>"


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
st.title("Tokenization")
st.markdown(
    """
    Before a language model sees your text, it splits the string into integer
    **tokens** from a fixed vocabulary. Tokens aren't always words: common words
    are usually a single token, but rare words, punctuation, and leading
    whitespace are often pieces of their own.

    Edit the prompt below to see how it's encoded into tokens.
    """
)

spec = render_model_selector(
    key="tokenizer_model",
    help=(
        "Each model has its own tokenizer. The same prompt can result in a different "
        "number of tokens and different token boundaries depending on which model you pick."
    ),
)
with st.spinner(f"Loading {spec.model_id} tokenizer…"):
    tokenizer = load_tokenizer(spec)

prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, key="tokenizer_prompt", height=250)

if st.button("Tokenize prompt", type="primary"):
    st.session_state["tokenizer_rows"] = tokenize(tokenizer, prompt)
    st.session_state["tokenizer_prompt_used"] = prompt

rows = st.session_state.get("tokenizer_rows")

if rows is None:
    st.info("Click **Tokenize prompt** to see how this prompt is encoded into tokens.")
elif not rows:
    st.info("Type something in the prompt above and click **Tokenize prompt**.")
else:
    prompt_used = st.session_state.get("tokenizer_prompt_used", "")

    st.markdown("---")
    st.markdown("#### Tokenized prompt")
    st.markdown(
        """
        Large language models break text into tokens, which are
        represented by integers.
        """
    )
    token_ids = [r[1] for r in rows]
    st.code(str(token_ids), language=None, wrap_lines=True)

    st.markdown("---")
    st.markdown("#### Decoded tokens")
    st.markdown(
        """
        Each token integer can be mapped back ("decoded") to its corresponding string
        representation:
        """
    )

    n_tokens = len(rows)
    n_words = len(prompt_used.split())

    st.markdown(
        f"""
    This prompt contains **{n_words}** words and **{n_tokens}** tokens, shown below:

    - `·` represents a space
    - `↵` represents a newline

    """
    )

    st.markdown(format_inline_html(rows), unsafe_allow_html=True)

render_model_link(spec)
