import streamlit as st

st.set_page_config(
    page_title="Deep Dive: LLMs",
    layout="centered",
)

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/next_token.py", title="Demo 1 — Next-token prediction", icon="🎯"),
]

pg = st.navigation(pages)
pg.run()