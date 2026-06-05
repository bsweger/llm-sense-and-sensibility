"""Demos home page."""

import streamlit as st

st.title("LLMs Under the Hood: Demos")
st.markdown(
    """
    Use the sidebar to pick a demo. Each of the demos below can be run using
    one of several models, to illustrate the differences.

    **Note:** The _Qwen2.5-0.5B_ model is included to represent a newer
    generation of local models. While still small, it's much larger than the
    other models used in the demo, and using it will be slow.

    ### Tokenization
    Shows how a prompt gets split into individual tokens. The same text gets sliced
    differently by different tokenizers — switch models in the sidebar to compare.

    ### Inference
    Step through a sentence one token at a time. The selected model produces
    a probability distribution over the vocabulary at each step. You can choose
    to complete the prompt using the token with the highest probability, or
    you can choose to sample.

    ### Attention (WIP)
    We didn't talk much about _attention_ in the deep dive, but this demo
    is an attempt to show how attention works during inference. For a given
    prompt, you choose a token, and the demo shows how that token "pays
    attention" to the other tokens in the prompt.

    Attention is an important input for generating the probability scores
    used for inference.

    ----------------------

    ### Comparing the models

    Every demo lets you pick from the same set of models in the sidebar. Each
    model has a different vocabulary, training corpus, and size — the table
    below summarizes the parts that show up most directly in the demos.

    | Model | Released | Parameters | Vocabulary | Context window | Trained on |
    |---|---|---|---|---|---|
    | GPT-2 (124M) | Feb 2019 | 124M | 50,257 tokens · English | 1,024 tokens | ~8M web pages [retrieved from qualified outbound links in Reddit posts](https://huggingface.co/openai-community/gpt2#training-data) |
    | Pythia-160m | Feb 2023 | 160M | 50,304 tokens · English | 2,048 tokens | [The Pile](https://pile.eleuther.ai) — ~800 GB of web pages, books, academic papers, code, and chat logs |
    | Qwen2.5-0.5B | Sep 2024 | 494M | 151,936 tokens · 29+ languages | 32,768 tokens | ~18 trillion tokens of multilingual web text, code, and math |
    | TinyStories-33M | May 2023 | 33M | Reuses GPT-Neo's 50,257 tokens · English | 512 tokens | [Short children's stories](https://huggingface.co/datasets/roneneldan/TinyStories) using only words a 3–4 year old would know |
    | GPT-2 Austen | community fine-tune | 124M | Inherits GPT-2's 50,257 tokens · English | 1,024 tokens | Jane Austen's novels (on top of GPT-2 pretraining) |

    - Parameters: How many internal numbers (across embedding tables, attention layers, etc.) the model adjusts during training. More parameters generally means more coherent output.
    - Vocaulary: Number of unique tokens a model understands and generates. Larger vocabularies usually mean that each word takes fewer tokens to represent.
    - Context window: Maximum length of the input a model can read when picking the next token

    ---
    """
)
