"""
Tokenization demo.

Show how a model converts a prompt into integer token ids, and what each id
decodes back to as a string. Different tokenizers slice the same text very
differently — switch the model in the sidebar to compare.
"""
from textwrap import dedent

import polars as pl
import streamlit as st
from transformers import AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# Display label -> Hugging Face model id. First entry is the default.
MODELS = {
    "GPT-2 (124M)": "gpt2",
    "Pythia-160m": "EleutherAI/pythia-160m",
}
DEFAULT_PROMPT = dedent("""\
    "I don't want to be human."

    Dr. Mensah said, "That's not an attitude a lot of humans are going to understand. We tend to think that because a bot or a construct looks human, its ultimate goal would be to become human."

    "That's the dumbest thing I've ever heard."
    """)

# Token pill palette — inspired by the Murderbot Apple TV show: muted corporate-
# dystopia tones with an amber/red HUD accent. Color carries no meaning; it just
# chunks adjacent tokens visually. Kept light so dark text stays legible.
TOKEN_COLORS = [
    "#e7e5e4",  # stone — corporate interior beige
    "#d4d4d8",  # gunmetal — SecUnit armor
    "#fed7aa",  # amber — HUD targeting overlay
    "#b6c8a5",  # muted moss — Preservation Alliance survey team
    "#fecaca",  # red — alert / warning indicator
]
ID_COLOR = "#9a3412"  # burnt amber, echoing HUD text


@st.cache_resource(show_spinner=True)
def load_tokenizer(model_id: str):
    """Load just the tokenizer for the given model id. Cached per model."""
    return AutoTokenizer.from_pretrained(model_id)


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
            f"<span style='font-size:15px;color:#111827;white-space:pre'>{_visible(decoded)}</span>"
            f"<span style='display:block;font-size:10px;color:{ID_COLOR};text-align:center'>{tid}</span>"
            f"</span>"
        )
    return "<div style='line-height:1.8'>" + "".join(pills) + "</div>"


def render_model_selector() -> str:
    """Render the sidebar model picker and return the chosen Hugging Face model id."""
    with st.sidebar:
        st.header("Model")
        label = st.selectbox(
            "Hugging Face model",
            list(MODELS.keys()),
            key="tokenizer_selected_model_label",
            help=(
                "Each model has its own tokenizer. The same prompt can result in "
                "a different number of tokens and different token boundaries "
                "depending on which model you pick."
            ),
        )
    return MODELS[label]


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

    Edit the prompt below to see how it gets sliced up.
    """
)

model_id = render_model_selector()
with st.spinner(f"Loading {model_id} tokenizer…"):
    tokenizer = load_tokenizer(model_id)

prompt = st.text_area(
    "Prompt",
    value=DEFAULT_PROMPT,
    key="tokenizer_prompt",
    height=200
)

rows = tokenize(tokenizer, prompt)

st.markdown("---")
st.markdown("#### Tokens")

if not rows:
    st.info("Type something in the prompt above to see it tokenized.")
else:
    n_tokens = len(rows)
    n_words = len(prompt.split())

    st.markdown(
    f"""
    This prompt contains **{n_words}** words and **{n_tokens}** tokens, shown below:

    - `·` represents a space
    - `↵` represents a newline

    """
)

    st.markdown(format_inline_html(rows), unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### Token detail")
    df = pl.DataFrame(
        {
            "Token Position": [r[0]+1 for r in rows],
            "Token ID": [r[1] for r in rows],
            "Decoded": [_visible(r[3]) for r in rows],
        }
    )
    st.dataframe(df, width="stretch", hide_index=True, height="content")

