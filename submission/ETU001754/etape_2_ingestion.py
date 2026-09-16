"""Etape 2 : Ingestion des sources dans la base de donnees vectorielle.

Ce module regroupe tout le pipeline d'ingestion :
- lecture des fichiers uploades (PDF, TXT, MD)
- nettoyage du texte (suppression des emojis)
- decoupage en chunks (chunking)
- vectorisation et indexation dans ChromaDB
"""
import os
import re
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Nettoyage du texte -----------------------------------------------------

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


# --- Modele d'embeddings -----------------------------------------------------

@st.cache_resource
def get_embedding_model():
    # Modele d'embeddings leger et efficace, execute 100% en local (pas d'appel API externe)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- Chargement des documents -------------------------------------------------

def _load_documents(uploaded_files, temp_dir):
    """Ecrit les fichiers uploades sur disque et charge leur contenu texte."""
    documents = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            docs = PyMuPDFLoader(file_path).load()
        elif file_extension in [".txt", ".md"]:
            docs = TextLoader(file_path, encoding="utf-8").load()
        else:
            st.warning(f"Format non supporte : {uploaded_file.name}")
            continue

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            doc.page_content = remove_emojis(doc.page_content)

        documents.extend(docs)

    return documents


# --- Chunking + indexation ----------------------------------------------------

def process_and_index_files(uploaded_files):
    """Extrait le texte des fichiers, le decoupe en chunks et l'indexe dans ChromaDB."""
    with tempfile.TemporaryDirectory() as temp_dir:
        all_documents = _load_documents(uploaded_files, temp_dir)

    if not all_documents:
        return None, 0

    # Decoupage du texte (Chunking) :
    # - chunk_size=1000 : environ 150-200 mots, ce qui conserve une unite semantique complete
    #   (un paragraphe) sans diluer le contexte transmis au LLM.
    # - chunk_overlap=200 : chevauchement de 20% pour ne pas couper une idee ou une phrase
    #   en plein milieu entre deux chunks consecutifs.
    # - separators : on privilegie les coupures naturelles (paragraphe, ligne, mot) avant
    #   de couper au caractere pres en dernier recours.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # Ingestion dans la base vectorielle ChromaDB (embeddings + stockage)
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, len(chunks)
