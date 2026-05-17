"""
LLM Explainer — interactive demos for a 40-minute presentation.

Entrypoint. Defines navigation and shared page config.
Run locally with:
    streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="LLM Explainer",
    page_icon="🧠",
    layout="centered",
)

pages = [
    st.Page("pages/0_home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_next_token.py", title="Part 1 — Next-token prediction", icon="🎯"),
]

pg = st.navigation(pages)
pg.run()
