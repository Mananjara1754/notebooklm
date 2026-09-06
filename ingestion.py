"""Extraction, chunking et indexation vectorielle des documents charges."""
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from llm_config import get_embedding_model
from text_utils import remove_emojis


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


def process_and_index_files(uploaded_files):
    """Extrait le texte des fichiers, le decoupe en chunks et l'indexe dans ChromaDB."""
    with tempfile.TemporaryDirectory() as temp_dir:
        all_documents = _load_documents(uploaded_files, temp_dir)

    if not all_documents:
        return None, 0

    # Decoupage du texte (Chunking) :
    # - chunk_size=1000 : environ 150-200 mots, ce qui conserve une unite semantique complete (un paragraphe) sans diluer le contexte.
    # - chunk_overlap=200 : chevauchement de 20% pour ne pas couper une idee ou une phrase en plein milieu entre deux chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # Ingestion dans la base vectorielle ChromaDB
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, len(chunks)
