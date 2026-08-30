import os
import re
import tempfile
import streamlit as st

# --- Imports des bibliothèques LangChain, VectorStore et LLM ---
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Fallbacks pour les imports afin de garantir la compatibilité entre versions de LangChain
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        from langchain_community.chat_models import ChatOllama as Ollama


# --- UTILITAIRE DE NETTOYAGE D'ÉMOJIS ---
def remove_emojis(text: str) -> str:
    """
    Filtre et supprime tous les émojis d'un texte (y compris le contenu des documents indexés).
    Garantit une affichage 100% textuel et épuré.
    """
    if not text:
        return ""
    emoji_pattern = re.compile(
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
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="NotebookLM Local", layout="wide")

# Injection CSS propre pour Poppins (ciblée uniquement sur le texte pour éviter de casser les icônes Streamlit)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, p, span, h1, h2, h3, h4, h5, h6, label, input, textarea, button {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* En-tête principal élégant */
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

    /* Bouton d'action principal */
    div.stButton > button[kind="primary"] {
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }

    /* Expander de sources */
    .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 0.9rem;
    }

    /* Message de chat */
    .stChatMessage {
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# En-tête de l'application
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">NotebookLM Local</h1>
        <p class="subtitle">Analyse et interrogation sémantique de vos documents locaux en toute confidentialité.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- INITIALISATION DES ÉTATS DE SESSION (session_state) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed_files_summary" not in st.session_state:
    st.session_state.indexed_files_summary = []


# --- MODÈLES (Cache Streamlit) ---
@st.cache_resource
def get_embedding_model():
    """
    Initialise et met en cache le modèle de vectorisation Hugging Face.
    Utilise sentence-transformers/all-MiniLM-L6-v2 (100% local).
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_ollama_llm():
    """
    Initialise et met en cache le LLM Ollama local avec le modèle Mistral.
    Se connecte à l'instance Ollama locale sur http://localhost:11434.
    """
    return Ollama(model="mistral", base_url="http://localhost:11434")


# --- ÉTAPE 4 : TEMPLATE DE PROMPT POUR MODE RAG ---
RAG_PROMPT_TEMPLATE = """Tu es un assistant spécialisé dans l'analyse de documents.
Réponds à la question posée en utilisant EXCLUSIVEMENT le contexte fourni ci-dessous.
Si la réponse ne se trouve pas dans le contexte, indique clairement : "Je ne trouve pas cette information dans les documents fournis."
N'utilise aucun émoji dans ta réponse. Ne tente pas d'inventer ou de spéculer sur la réponse.

Contexte :
{context}

Question :
{question}

Réponse :"""


# --- ÉTAPE 2 : PIPELINE D'INGESTION ---
def process_and_index_files(uploaded_files):
    """
    Traite les fichiers uploadés, extrait leur texte, les découpe en chunks,
    puis les indexe dans ChromaDB avec leurs métadonnées.
    """
    all_documents = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. EXTRACTION selon l'extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == ".pdf":
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            else:
                st.warning(f"Format non supporté pour le fichier {uploaded_file.name}")
                continue

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                # Nettoyage automatique des émojis présents dans le document source
                doc.page_content = remove_emojis(doc.page_content)
            
            all_documents.extend(docs)

    if not all_documents:
        return None, 0

    # 2. CHUNKING (Découpage en fragments)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # 3. VECTORISATION ET STOCKAGE CHROMADB
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, len(chunks)


# --- SIDEBAR : CONFIGURATION ET UPLOAD ---
with st.sidebar:
    st.header("Configuration & Documents")
    
    # Zone d'upload multi-fichiers
    uploaded_files = st.file_uploader(
        "Chargez vos documents", 
        type=["pdf", "md", "txt"], 
        accept_multiple_files=True
    )
    
    # Bouton d'indexation
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
                        st.success(f"Indexation réussie : {total_chunks} fragments enregistrés dans la base vectorielle.")
                    else:
                        st.error("Aucun document n'a pu être extrait.")
                except Exception as e:
                    st.error(f"Erreur lors de l'indexation : {e}")
        else:
            st.warning("Veuillez charger au moins un document avant d'indexer.")
            
    st.divider()
    
    # Affichage de l'état de la base vectorielle
    if st.session_state.vectorstore is not None:
        st.subheader("État de la base vectorielle")
        st.info(f"Documents indexés : {len(st.session_state.indexed_files_summary)}")
        with st.expander("Voir les détails des fichiers indexés"):
            for summary in st.session_state.indexed_files_summary:
                st.write(f"- {summary}")

    st.divider()
    
    # Toggle pour activer/désactiver le LLM
    llm_enabled = st.toggle("Activer le modèle LLM (RAG complet)", value=False)
    
    if llm_enabled:
        st.caption("Mode actuel : **RAG complet (Ollama / Mistral)**")
    else:
        st.caption("Mode actuel : **Recherche sémantique brute (sans LLM)**")


# --- ZONE PRINCIPALE : INTERFACE DE CHAT ---

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(remove_emojis(message["content"]))
        if "sources" in message and message["sources"]:
            with st.expander("Sources consultées pour cette réponse"):
                for i, doc in enumerate(message["sources"], 1):
                    source_name = doc.metadata.get("source", "Fichier inconnu")
                    page_info = f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
                    st.markdown(f"**Extrait {i}** — *Source : `{source_name}`{page_info}*")
                    st.code(remove_emojis(doc.page_content.strip()), language="text")

# Champ de saisie utilisateur
if prompt := st.chat_input("Posez une question sur vos documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            response = "Veuillez d'abord charger et indexer des documents dans la barre latérale."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            if not llm_enabled:
                # --- ÉTAPE 3 : MODE RECHERCHE SÉMANTIQUE BRUTE ---
                with st.spinner("Recherche des extraits pertinents..."):
                    results = st.session_state.vectorstore.similarity_search(prompt, k=4)
                    
                    if not results:
                        response = "Aucun extrait pertinent trouvé dans la base vectorielle."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        formatted_response = "### Extraits sémantiques pertinents :\n\n"
                        for i, doc in enumerate(results, 1):
                            source_name = doc.metadata.get("source", "Fichier inconnu")
                            page_info = f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
                            cleaned_content = remove_emojis(doc.page_content.strip())
                            
                            formatted_response += f"**Extrait {i}** — *Source : `{source_name}`{page_info}*\n"
                            formatted_response += f"```text\n{cleaned_content}\n```\n\n"
                        
                        st.markdown(formatted_response)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
            else:
                # --- ÉTAPE 4 : MODE RAG COMPLET (OLLAMA / MISTRAL) ---
                with st.spinner("Recherche des contextes et génération de la réponse avec Mistral (Ollama)..."):
                    try:
                        results = st.session_state.vectorstore.similarity_search(prompt, k=4)
                        
                        if not results:
                            response = "Aucun extrait pertinent trouvé dans les documents indexés pour répondre à la question."
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
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
                            
                            with st.expander("Sources consultées pour cette réponse"):
                                for i, doc in enumerate(results, 1):
                                    source_name = doc.metadata.get("source", "Fichier inconnu")
                                    page_info = f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
                                    st.markdown(f"**Extrait {i}** — *Source : `{source_name}`{page_info}*")
                                    st.code(remove_emojis(doc.page_content.strip()), language="text")

                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": cleaned_response,
                                "sources": results
                            })
                    except Exception as e:
                        error_msg = f"Erreur Ollama : {e}\n\nVérifiez que le serveur Ollama est démarré (ollama serve) et que le modèle mistral est disponible (ollama list)."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
