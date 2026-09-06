import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, p, span, h1, h2, h3, h4, h5, h6, label, input, textarea, button {
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main-header {
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.15);
}

.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
}

.subtitle {
    color: #718096;
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: 0.2rem;
}

div.stButton > button[kind="primary"] {
    border-radius: 6px;
    font-weight: 500;
    padding: 0.5rem 1rem;
    width: 100%;
    transition: all 0.2s ease-in-out;
}

.streamlit-expanderHeader {
    font-weight: 500;
    font-size: 0.9rem;
}

.stChatMessage {
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
</style>
"""

HEADER_HTML = """
<div class="main-header">
    <h1 class="main-title">NotebookLM Local</h1>
    <p class="subtitle">Analyse et interrogation semantique de vos documents locaux en toute confidentialite.</p>
</div>
"""


def render_page_style_and_header():
    """Configure la page Streamlit et injecte le CSS + l'en-tete."""
    st.set_page_config(page_title="NotebookLM Local", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(HEADER_HTML, unsafe_allow_html=True)
