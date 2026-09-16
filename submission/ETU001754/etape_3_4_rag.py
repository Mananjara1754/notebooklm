"""Etapes 3 & 4 : Recherche dans la base de donnees vectorielle et generation RAG.

Ce module regroupe :
- la connexion au LLM local (Ollama / Mistral)
- le prompt RAG strict (reponse basee uniquement sur le contexte retrouve)
- la recherche semantique (mode "brut", sans LLM)
- la generation de reponse RAG complete (recherche + LLM)
- l'affichage de l'historique de chat et des sources consultees
"""
import re

import streamlit as st
from langchain_core.prompts import PromptTemplate

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        from langchain_community.chat_models import ChatOllama as Ollama


# --- Nettoyage du texte (identique a l'etape d'ingestion) --------------------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "☀-⛿"
    "✀-➿"
    "]+",
    flags=re.UNICODE
)


def remove_emojis(text: str) -> str:
    """Supprime les emojis d'un texte pour garder un affichage propre."""
    if not text:
        return ""
    return _EMOJI_PATTERN.sub("", text)


# --- Modele LLM local ----------------------------------------------------------

@st.cache_resource
def get_ollama_llm():
    # Connexion a l'instance Ollama locale (modele Mistral) sur le port 11434
    return Ollama(model="mistral", base_url="http://localhost:11434")


# Prompt RAG strict pour forcer le LLM a s'appuyer uniquement sur le contexte retrouve
RAG_PROMPT_TEMPLATE = """Tu es un assistant specialise dans l'analyse de documents.
Reponds a la question posee en utilisant EXCLUSIVEMENT le contexte fourni ci-dessous.
Si la reponse ne se trouve pas dans le contexte, indique clairement : "Je ne trouve pas cette information dans les documents fournis."
N'utilise aucun emoji dans ta reponse. Ne tente pas d'inventer ou de speculer sur la reponse.

Contexte :
{context}

Question :
{question}

Reponse :"""


# --- Affichage des sources -------------------------------------------------------

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


# --- Mode 1 : recherche semantique brute (sans LLM) -------------------------------

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


# --- Mode 2 : RAG complet (recherche + generation LLM) -----------------------------

def _answer_with_rag(prompt):
    """Mode RAG complet : recherche de contexte + generation via Ollama / Mistral."""
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


# --- Point d'entree appele par l'interface -------------------------------------------

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
