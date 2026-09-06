"""Modeles d'embeddings/LLM et prompt RAG (avec fallbacks pour la compatibilite des versions)."""
import streamlit as st

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        from langchain_community.chat_models import ChatOllama as Ollama


@st.cache_resource
def get_embedding_model():
    # Modele d'embeddings leger et efficace execute 100% en local
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_ollama_llm():
    # Connexion a l'instance Ollama locale sur le port 11434
    return Ollama(model="mistral", base_url="http://localhost:11434")


# Prompt RAG strict pour forcer le LLM a s'appuyer uniquement sur le contexte
RAG_PROMPT_TEMPLATE = """Tu es un assistant specialise dans l'analyse de documents.
Reponds a la question posee en utilisant EXCLUSIVEMENT le contexte fourni ci-dessous.
Si la reponse ne se trouve pas dans le contexte, indique clairement : "Je ne trouve pas cette information dans les documents fournis."
N'utilise aucun emoji dans ta reponse. Ne tente pas d'inventer ou de speculer sur la reponse.

Contexte :
{context}

Question :
{question}

Reponse :"""
