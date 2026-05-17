import streamlit as st

st.set_page_config(
    page_title="Deep Dive: LLMs",
    layout="centered",
)

pages = [
    st.Page("pages/0_home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_next_token.py", title="Part 1 — Next-token prediction", icon="🎯"),
]

pg = st.navigation(pages)
pg.run()