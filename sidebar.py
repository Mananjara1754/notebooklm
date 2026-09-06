"""Barre laterale : upload/indexation des documents et selection du mode LLM."""
import streamlit as st

from ingestion import process_and_index_files


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
    """Affiche la barre laterale complete et retourne l'etat du toggle LLM."""
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
