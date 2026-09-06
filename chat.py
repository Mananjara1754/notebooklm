"""Zone de chat : historique, recherche semantique et reponses RAG."""
import streamlit as st

from langchain_core.prompts import PromptTemplate

from llm_config import RAG_PROMPT_TEMPLATE, get_ollama_llm
from text_utils import remove_emojis


def _source_label(doc) -> str:
    source_name = doc.metadata.get("source", "Fichier inconnu")
    page_info = f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
    return f"*Source : `{source_name}`{page_info}*"


def _render_sources_expander(results):
    with st.expander("Sources consultees pour cette reponse"):
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Extrait {i}** — {_source_label(doc)}")
            st.code(remove_emojis(doc.page_content.strip()), language="text")


def render_chat_history():
    """Reaffiche les messages precedemment echanges dans la session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(remove_emojis(message["content"]))
            if "sources" in message and message["sources"]:
                _render_sources_expander(message["sources"])


def _answer_with_semantic_search_only(prompt):
    """Mode sans LLM : renvoie directement les extraits les plus pertinents."""
    with st.spinner("Recherche des extraits pertinents..."):
        results = st.session_state.vectorstore.similarity_search(prompt, k=4)

        if not results:
            response = "Aucun extrait pertinent trouve dans la base vectorielle."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return

        formatted_response = "### Extraits semantiques pertinents :\n\n"
        for i, doc in enumerate(results, 1):
            cleaned_content = remove_emojis(doc.page_content.strip())
            formatted_response += f"**Extrait {i}** — {_source_label(doc)}\n"
            formatted_response += f"```text\n{cleaned_content}\n```\n\n"

        st.markdown(formatted_response)
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})


def _answer_with_rag(prompt):
    """Mode RAG complet : contexte + generation via Ollama / Mistral."""
    with st.spinner("Recherche des contextes et generation de la reponse avec Mistral (Ollama)..."):
        try:
            results = st.session_state.vectorstore.similarity_search(prompt, k=4)

            if not results:
                response = "Aucun extrait pertinent trouve dans les documents indexes pour repondre a la question."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            context_text = "\n\n---\n\n".join([remove_emojis(doc.page_content) for doc in results])

            prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            formatted_prompt = prompt_template.format(context=context_text, question=prompt)

            llm = get_ollama_llm()

            if hasattr(llm, "invoke"):
                raw_response = llm.invoke(formatted_prompt)
                response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            else:
                response_text = str(llm(formatted_prompt))

            cleaned_response = remove_emojis(response_text)
            st.markdown(cleaned_response)
            _render_sources_expander(results)

            st.session_state.messages.append({
                "role": "assistant",
                "content": cleaned_response,
                "sources": results
            })
        except Exception as e:
            error_msg = (
                f"Erreur Ollama : {e}\n\n"
                "Verifiez que le serveur Ollama est demarre (ollama serve) "
                "et que le modele mistral est disponible (ollama list)."
            )
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


def handle_user_prompt(llm_enabled: bool):
    """Recupere la question de l'utilisateur et declenche le mode de reponse adapte."""
    prompt = st.chat_input("Posez une question sur vos documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            response = "Veuillez d'abord charger et indexer des documents dans la barre laterale."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        elif not llm_enabled:
            _answer_with_semantic_search_only(prompt)
        else:
            _answer_with_rag(prompt)
