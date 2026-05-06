import marimo

__generated_with = "0.23.3"
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


if __name__ == "__main__":
    app.run()
