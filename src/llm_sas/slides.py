import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", layout_file="layouts/llm_peek.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sense and Sensibility: A Peek at LLMs
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging

    logging.set_verbosity_error()

    MODEL_PATH = Path("~/models/huggingface/hub/models--openai-community--gpt2").expanduser()
    snapshots_dir = MODEL_PATH / "snapshots"
    snapshot = next(d for d in snapshots_dir.iterdir() if d.is_dir())


    def predict_next_word(text: str) -> str:
        tokenizer = GPT2Tokenizer.from_pretrained(snapshot, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(snapshot, local_files_only=True)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        return tokenizer.decode(next_token_id)


    return (predict_next_word,)


@app.cell
def _(predict_next_word):
    prompt = "I like to think that"
    print(f"{prompt} {predict_next_word(prompt)}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Same words, different weights

    Look at the **Natural** and **After filters** columns side-by-side: the
    words listed are always the same. Only the percentages change.

    That's not a bug — it's how temperature and top-k actually work:

    - **Temperature** is a "spread" dial. It changes *how concentrated* the
      model's confidence is across its choices — but it can't promote a new
      word into the running or kick one out. The same words stay on stage,
      just taking up different amounts of space.

    - **Top-k** is a doorman. It lets only the K most-likely words past the
      rope. As long as K is at least 5 (the display size), all five top picks
      are admitted, in the same order.

    So the sampling controls don't change *which* words the model considers —
    they change *how* it weights the words it already had in mind. People are
    sometimes surprised that "creativity" doesn't introduce new options; it
    just re-balances them.

    A different category of tools *can* change which words appear at the top.
    None of these are in this demo, but they're worth knowing about:

    - **Repetition penalty** — a "you already said that" suppressor. If a word
      has already appeared in the sentence, this tool reduces its probability
      so the model is less likely to pick it again. Useful for avoiding loops
      like "the the the" or paragraphs that keep restating themselves.

    - **Logit bias** — a manual override list. The developer can say "boost
      the chance of word X" or "never pick word Y, ever." Used to forbid
      offensive words, force specific output formats (e.g., always start with
      "The answer is..."), or steer style.

    - **Beam search** — instead of picking one word at a time, the model
      looks a few words ahead and picks the sequence with the best *combined*
      score. Like a chess engine thinking several moves ahead instead of just
      the next move. Tends to produce safer, more coherent text but with
      less variety.
    """)
    return


if __name__ == "__main__":
    app.run()
