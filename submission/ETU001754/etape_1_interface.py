"""Etape 1 : Interface utilisateur Streamlit du systeme RAG local.

Ce fichier assemble l'application complete :
- mise en page / style de la page Streamlit
- barre laterale (upload de documents, declenchement de l'indexation, toggle RAG)
- zone de chat (historique + reponses)

L'ingestion (etape_2_ingestion.py) et la recherche/generation RAG
(etape_3_4_rag.py) sont importees depuis leurs modules dedies.
"""
import streamlit as st

from etape_2_ingestion import process_and_index_files
from etape_3_4_rag import render_chat_history, handle_user_prompt


# --- Style et en-tete de la page ------------------------------------------------

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


# --- Barre laterale : upload, indexation, toggle LLM --------------------------------

def _render_upload_and_index_section():
    uploaded_files = st.file_uploader(
        "Chargez vos documents",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True
    )

    if st.button("Indexer les documents", type="primary"):
        if uploaded_files:
            with st.spinner("Indexation en cours (extraction, chunking, vectorisation)..."):
                try:
                    vectorstore, total_chunks = process_and_index_files(uploaded_files)

                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.indexed_files_summary = [
                            f"{f.name} ({f.size} octets)" for f in uploaded_files
                        ]
                        st.success(f"Indexation reussie : {total_chunks} fragments enregistres.")
                    else:
                        st.error("Aucun document n'a pu etre extrait.")
                except Exception as e:
                    st.error(f"Erreur lors de l'indexation : {e}")
        else:
            st.warning("Veuillez charger au moins un document avant d'indexer.")


def _render_vectorstore_status_section():
    if st.session_state.vectorstore is not None:
        st.subheader("Etat de la base vectorielle")
        st.info(f"Documents indexes : {len(st.session_state.indexed_files_summary)}")
        with st.expander("Voir les details des fichiers indexes"):
            for summary in st.session_state.indexed_files_summary:
                st.write(f"- {summary}")


def render_sidebar() -> bool:
    """Affiche la barre laterale complete et retourne l'etat du toggle LLM (RAG complet ou non)."""
    with st.sidebar:
        st.header("Configuration & Documents")

        _render_upload_and_index_section()
        st.divider()
        _render_vectorstore_status_section()
        st.divider()

        llm_enabled = st.toggle("Activer le modele LLM (RAG complet)", value=False)

        if llm_enabled:
            st.caption("Mode actuel : **RAG complet (Ollama / Mistral)**")
        else:
            st.caption("Mode actuel : **Recherche semantique brute (sans LLM)**")

    return llm_enabled


# --- Point d'entree de l'application -----------------------------------------------

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
