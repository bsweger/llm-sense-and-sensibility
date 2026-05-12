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
    print("what's going on?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hi
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


if __name__ == "__main__":
    app.run()
