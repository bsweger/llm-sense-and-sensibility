# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.6",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(
    width="medium",
    app_title="NPL Deep Dive: LLMs",
    layout_file="layouts/slides.slides.json",
)


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    ASSETS = mo.notebook_dir() / "assets"
    DEMO_URL = Path("https://llm-demos.streamlit.app/")
    return ASSETS, DEMO_URL, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM deep dive: what's in there?
    ![Murderbot looking with trepidation](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDQxZDJzbm81NngyejhpbjM4NTByeGQ4bmxzaW5oczJucHAzbWtkZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9E4bVh3EKeM9Ls6kZR/giphy.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Goals

    - Demystify jargon
    - Demo foundational concepts
    - Strengthen mental models around LLMs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What we won't cover

    Important topics that are out of scope:

    - Ethical concerns
    - Ecosystems surrounding the models (harnesses, agents)
    - Non-language examples (music, image generation)
    - Practical use cases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Large Language Models (LLMs)

    > A language model of sufficient size and complexity that interprets and generates text using natural
    human language. LLMs can predict text from input, based on their training from large amounts of
    text data.

    - LLMs are neural networks
    - Usually refers to models based on a transformer architecture
    - [_Attention is All You Need_](https://arxiv.org/abs/1706.03762): 2017 paper that started our current era of language models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    #
    #

    # Yes, but what is a neural network?

    #
    #
    #
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Click below for a one-minute explainer
    [![Thumbnail for neural networks explained in a minute](https://img.youtube.com/vi/rEDzUT3ymw4/hqdefault.jpg)](https://www.youtube.com/watch?v=rEDzUT3ymw4)
    """)
    return


@app.cell(hide_code=True)
def _(ASSETS, mo):
    mo.vstack([
        mo.md("""
        # Creating an LLM's neural network
        - architecture: defines a model's vocabulary, number of network layers, and other parameters
        - training process: creates the embeddings and weights used when predicting words
        """),
        mo.md("&nbsp;"),
        mo.image(src=ASSETS / "model_training_phases.png", width=600),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # It all starts with tokens

    - lego pieces that represent the most atomic units of languge
    - represented by integers called _token ids_
    - every model has its own "tokenizer" that determines how language is broken down into tokens
    - a model's _vocabulary_ is the number of unique tokens a model understands and generates
    """)
    return


@app.cell(hide_code=True)
def _(DEMO_URL, mo):
    mo.md(f"""
    # Demo: tokenization
    [{DEMO_URL}/tokenizer]({DEMO_URL}/tokenizer)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Jargon: embeddings

    - Each token has an embedding that captures how it relates to every other token in the vocabulary
    - An _embedding_ is a list of numbers
    - The training process adjusts the embeddings until similar tokens end up with similar values
    """)
    return


@app.cell
def _(ASSETS, mo):
    mo.vstack([
        mo.md("""
        # Embeddings: _bank_
        """),
        mo.md("&nbsp;"),
        mo.image(src=ASSETS / "embedding_space_before_after_training.png", width=700),
    ])
    return


@app.cell
def _(ASSETS, mo):
    mo.vstack([
        mo.md("""
        # Embeddings: model's view
        """),
        mo.md("&nbsp;"),
        mo.image(src=ASSETS / "embeddings_as_model_data.png", width=700),
    ])
    return


@app.cell
def _(ASSETS, mo):
    mo.vstack([
        mo.md("""
        # Jargon - inference
        - _Inference_ is what happens when you send a prompt to an LLM
        - The last token in the prompt represents the token _and_ everything the model has "figured out" about the rest of the prompt
        """),
        mo.md("&nbsp;"),
        mo.image(src=ASSETS / "llm_inference.png"),
    ])
    return


@app.cell
def _(DEMO_URL, mo):
    mo.md(f"""
    # Demo: inference
    [{DEMO_URL}/inference]({DEMO_URL}/next_token)
    """)
    return


@app.cell
def _(ASSETS, mo):
    mo.vstack([
        mo.md("""
        # "Embeddings through neural network layers" is not auditable
        """),
        mo.md("&nbsp;"),
        mo.image(src=ASSETS / "llm_inference_wat.png"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Discussion and last thoughts

    - Biases
        - Some organizations release "open weights" models (Google's Gemma family)
        - Typically, these "open weights" models don't publish their training data
    - Retaining voice
    """)
    return


if __name__ == "__main__":
    app.run()
