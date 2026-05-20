# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.6",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(
    width="medium",
    app_title="NPL Deep Dive: LLMs",
    layout_file="layouts/slides.slides.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


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

    - **decoder (causal):** predicts one word at a time in left-to-right sequence, used to generate text — most modern generative AI models
    - **encoder:** evaluate entire input at once, designed to read text, not write it — used for classification, search retrieval, clustering, etc.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ![Thumbnail for neural networks explained in a minute](https://img.youtube.com/vi/rEDzUT3ymw4/hqdefault.jpg)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
        # Creating an LLM's neural network
        - architecture: defines a model's vocabulary, number of network layers, and other parameters
        - training process: creates the embeddings and weights used when predicting words
        """),
        mo.md("&nbsp;"),
        mo.image(src="src/llm_sas/assets/model training phases.png", width=600),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # It all starts with tokens

    - lego pieces that represent the most atomic units of languge
    - represented by integers called _token ids_
    - the token ids map to _embdeddings_: lists of numbers creating during training that represent what the token "means"
    - every model has its own "tokenizer" that determines how language is broken down into tokens
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo: tokenization

    http://localhost:8501/tokenizer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What you type

    > "I don't want to be human."
    >
    > Dr. Mensah said, "That's not an attitude a lot of humans are going to understand. We tend to think that because a bot or a construct looks human, its ultimate goal would be to become human."
    >
    > "That's the dumbest thing I've ever heard."

    ### Token ids: what the model "sees"

    [1, 40, 836, 470, 765, 284, 307, 1692, 526, 198, 198, 6187, 13, 43103, 993, 531, 11, 366, 2504, 338, 407, 281, 9408, 257, 1256, 286, 5384, 389, 1016, 284, 1833, 13, 775, 4327, 284, 892, 326, 780, 257, 10214, 393, 257, 5678, 3073, 1692, 11, 663, 8713, 3061, 561, 307, 284, 1716, 1692, 526, 198, 198, 1, 2504, 338, 262, 13526, 395, 1517, 314, 1053, 1683, 2982, 526, 198]

    ### Token ids decoded

    ```
    "
    I
     don
    't
     want
     to
     be
     human
    ."
    ↵
    ↵
    Dr
    .
     Mens
    ah
     said
    ,
     "
    That
    's
     not
     an
     attitude
     a
     lot
     of
     humans
     are
     going
     to
     understand
    .
     We
     tend
     to
     think
     that
     because
     a
     bot
     or
     a
     construct
     looks
     human
    ,
     its
     ultimate
     goal
     would
     be
     to
     become
     human
    ."
    ↵
    ↵
    "
    That
    's
     the
     dumb
    est
     thing
     I
    've
     ever
     heard
    ."
    ↵
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Vocabulary

    - the number of unique tokens that a model is able understand and generate
    - GPT2 = 50,257
    - Pythia-160m = 50,304
    - GPT4 = ~100,000
    - Opus 4.7 = ???
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Jargon: embeddings

    - An _embedding_ is a list of numbers
    - Each token has an embedding that captures how it relates to every other token in the vocabulary
    - The training process adjusts the embeddings until similar tokens end up with similar values
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
        # Embeddings mental model
        Consider the word _bank_
        """),
        mo.md("&nbsp;"),
        mo.image(src="src/llm_sas/assets/embedding_space_before_after_training.png", width=600),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
        # Embeddings - model's actual data structure (simplified)
        """),
        mo.md("&nbsp;"),
        mo.image(src="src/llm_sas/assets/embeddings_as_model_data.png", width=600),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
        # Jargon - inference
        - _Inference_ is what happens when you send a prompt to an LLM
        - The last token in the prompt represents the token _and_ everything the model has "figured out" about the rest of the prompt
        """),
        mo.md("&nbsp;"),
        mo.image(src="src/llm_sas/assets/npl-deep-dive-llm-inference.png"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo: inference

    http://localhost:8501/next_token
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
        # Inference - one more thing
        """),
        mo.md("&nbsp;"),
        mo.image(src="src/llm_sas/assets/npl-deep-dive-llm-inference wat.png"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Last thoughts

    - Biases
        - Some organizations release "open weights" models (Google's Gemma family)
        - Typically, these "open weights" models don't publish their training data
    -
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
