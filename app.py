import streamlit as st

from styles import render_page_style_and_header
from sidebar import render_sidebar
from chat import render_chat_history, handle_user_prompt

render_page_style_and_header()

# Initialisation des variables dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed_files_summary" not in st.session_state:
    st.session_state.indexed_files_summary = []

llm_enabled = render_sidebar()

render_chat_history()
handle_user_prompt(llm_enabled)
