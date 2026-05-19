# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.6",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", layout_file="layouts/slides.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM deep dive: what's in there?
    ![Alt Text](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDQxZDJzbm81NngyejhpbjM4NTByeGQ4bmxzaW5oczJucHAzbWtkZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9E4bVh3EKeM9Ls6kZR/giphy.gif)
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
    ```
    Good call, you're absolutely right! I'm not an expert in this space, and
    I should have made that clearer at the beginning of the presentation.

    What I can do is share what I've been learning and understand how that
    might connect with our collective hopes, dreams, and fears of using
    LLMs at work.
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Jargon: tokens

    > A word, word part, or character that a language model uses to interpret and generate text. Tokens can also include special characters and symbols, or even individual bytes that are not by themselves valid characters.

    Think of tokens as lego pieces that represent the most atomic units of languge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo: tokenization
    """)
    return


if __name__ == "__main__":
    app.run()
