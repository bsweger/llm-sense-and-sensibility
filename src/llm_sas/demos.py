import streamlit as st

st.set_page_config(
    page_title="Deep Dive: LLMs",
    layout="centered",
)

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/next_token.py", title="Next-token prediction", icon="🎯"),
    st.Page("pages/tokenizer.py", title="Tokenization", icon="🔡"),
]

pg = st.navigation(pages)
pg.run()
