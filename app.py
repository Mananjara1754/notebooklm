import os
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


# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="NotebookLM Local", page_icon="📚", layout="wide")

st.title("Clone local de NotebookLM 📚")
st.markdown("Interrogez vos documents localement (sans aucune API externe).")

# --- INITIALISATION DES ÉTATS DE SESSION (session_state) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed_files_summary" not in st.session_state:
    st.session_state.indexed_files_summary = []


# --- MODELES (Cache Streamlit) ---
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
# Instruction stricte : le LLM doit répondre EXCLUSIVEMENT en se basant sur le contexte fourni.
# Si l'information est absente, il doit l'indiquer clairement plutôt qu'inventer.
RAG_PROMPT_TEMPLATE = """Tu es un assistant spécialisé dans l'analyse de documents.
Réponds à la question posée en utilisant EXCLUSIVEMENT le contexte fourni ci-dessous.
Si la réponse ne se trouve pas dans le contexte, indique clairement : "Je ne trouve pas cette information dans les documents fournis."
Ne tente pas d'inventer ou de spéculer sur la réponse.

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

    # Fichiers temporaires pour la lecture par PyMuPDFLoader / TextLoader
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

            # Conservation de la métadonnée 'source' (nom du fichier)
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            
            all_documents.extend(docs)

    if not all_documents:
        return None, 0

    # 2. CHUNKING (Découpage en fragments)
    # Justification : chunk_size=1000 préserve le contexte sémantique ; 
    # chunk_overlap=200 évite la perte d'information aux frontières de découpe.
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
                        st.success(f"✅ Indexation réussie ! {total_chunks} chunks créés et stockés dans ChromaDB.")
                    else:
                        st.error("Aucun document n'a pu être extrait.")
                except Exception as e:
                    st.error(f"Erreur lors de l'indexation : {e}")
        else:
            st.warning("Veuillez charger au moins un document avant d'indexer.")
            
    st.divider()
    
    # Affichage de débogage pour valider l'Étape 2
    if st.session_state.vectorstore is not None:
        st.subheader("📊 État de la base ChromaDB")
        st.info(f"Documents indexés : {len(st.session_state.indexed_files_summary)}")
        with st.expander("Voir les détails des fichiers indexés"):
            for summary in st.session_state.indexed_files_summary:
                st.write(f"- {summary}")

    st.divider()
    
    # Toggle pour activer/désactiver le LLM
    llm_enabled = st.toggle("Activer le modèle LLM (RAG complet)", value=False)
    
    if llm_enabled:
        st.caption("Mode actuel : **RAG complet (Ollama / Mistral)** 🤖")
    else:
        st.caption("Mode actuel : **Recherche Sémantique brute (sans LLM)** 🔍")


# --- ZONE PRINCIPALE : INTERFACE DE CHAT ---

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Transparence des sources conservée dans l'historique pour l'étape 4
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources utilisées pour cette réponse"):
                for i, doc in enumerate(message["sources"], 1):
                    source_name = doc.metadata.get("source", "Fichier inconnu")
                    page_info = f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
                    st.markdown(f"**Extrait {i}** — *Source : `{source_name}`{page_info}*")
                    st.code(doc.page_content.strip(), language="text")

# Champ de saisie utilisateur
if prompt := st.chat_input("Posez une question sur vos documents..."):
    # Ajout de la question à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Bloc de réponse de l'assistant
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            response = "⚠️ Veuillez d'abord charger et indexer des documents dans la barre latérale."
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
                        formatted_response = "### 🔍 Extraits sémantiques pertinents trouvés :\n\n"
                        for i, doc in enumerate(results, 1):
                            source_name = doc.metadata.get("source", "Fichier inconnu")
                            page_info = f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
                            
                            formatted_response += f"**Extrait {i}** — *Source : `{source_name}`{page_info}*\n"
                            formatted_response += f"```text\n{doc.page_content.strip()}\n```\n\n"
                        
                        st.markdown(formatted_response)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
            else:
                # --- ÉTAPE 4 : MODE RAG COMPLET (OLLAMA / MISTRAL) ---
                with st.spinner("Recherche des contextes et génération de la réponse avec Mistral (Ollama)..."):
                    try:
                        # 1. Récupération des chunks pertinents
                        results = st.session_state.vectorstore.similarity_search(prompt, k=4)
                        
                        if not results:
                            response = "Aucun extrait pertinent trouvé dans les documents indexés pour répondre à la question."
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            # Concaténation des chunks pour construire le contexte
                            context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
                            
                            # 2. Ingénierie de prompt avec PromptTemplate
                            prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                            formatted_prompt = prompt_template.format(context=context_text, question=prompt)
                            
                            # 3. Appel du LLM local Ollama (Mistral)
                            llm = get_ollama_llm()
                            
                            if hasattr(llm, "invoke"):
                                raw_response = llm.invoke(formatted_prompt)
                                response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
                            else:
                                response_text = str(llm(formatted_prompt))
                            
                            # 4. Affichage de la réponse générée par le LLM
                            st.markdown(response_text)
                            
                            # 5. Transparence (Exigence du sujet) : st.expander dépliable avec les sources
                            with st.expander("📚 Sources utilisées pour cette réponse"):
                                for i, doc in enumerate(results, 1):
                                    source_name = doc.metadata.get("source", "Fichier inconnu")
                                    page_info = f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
                                    st.markdown(f"**Extrait {i}** — *Source : `{source_name}`{page_info}*")
                                    st.code(doc.page_content.strip(), language="text")

                            # Sauvegarde dans l'historique avec les sources rattachées
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text,
                                "sources": results
                            })
                    except Exception as e:
                        error_msg = f"❌ **Erreur Ollama** : {e}\n\nVérifiez que le serveur Ollama tourne (`ollama serve`) et que le modèle `mistral` est présent (`ollama list`)."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
