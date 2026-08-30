# Projet : Clone local de NotebookLM (TP RAG)

## Objectif
Construire une application Streamlit permettant à un utilisateur de charger des documents (PDF, Markdown, TXT), de les indexer dans une base vectorielle locale, puis d'interroger cette base — soit en pure recherche sémantique, soit via un LLM local (Ollama) contraint par le contexte récupéré.

**Contrainte stricte : aucun appel à une API externe (pas d'OpenAI, pas de Claude API, pas de service cloud). Tout doit tourner en local.**

## État actuel du setup
- Ollama est installé (v0.33.2) et le modèle `mistral` est téléchargé et fonctionnel (testé via `ollama run mistral`).
- Les étapes 1 à 3 doivent être terminées et validées avant de commencer l'étape 4.
- L'étape 4 (mode RAG complet) peut maintenant être implémentée normalement (voir section correspondante ci-dessous, qui remplace le stub).

## Stack imposée
- **UI** : Streamlit
- **Orchestration** : langchain, langchain-community
- **Vector store** : chromadb
- **Embeddings** : HuggingFaceEmbeddings (sentence-transformers), donc 100% local, pas d'API
- **Extraction PDF** : PyMuPDFLoader
- **Extraction texte/Markdown** : TextLoader
- **LLM local (étape 4 uniquement, plus tard)** : Ollama (mistral ou qwen2.5-coder)

## Fichiers attendus
- `app.py` — application complète, **commentée proprement** (c'est une exigence du rendu final)
- `requirements.txt` — avec versions figées des libs installées, pour éviter tout décalage d'API entre versions de LangChain

## Étape 1 — Squelette de l'interface Streamlit
- `st.sidebar` contenant :
  - `st.file_uploader` (multi-fichiers, types acceptés : pdf, md, txt)
  - un bouton d'indexation ("Indexer les documents")
  - un `st.toggle()` pour activer/désactiver le LLM (représente le passage entre mode "Recherche Sémantique" et mode "RAG complet")
- Zone principale :
  - interface conversationnelle avec `st.chat_message()` et `st.chat_input()`
  - historique de conversation conservé dans `st.session_state`
- À ce stade, aucune logique d'indexation ni de recherche n'est branchée — juste l'UI fonctionnelle.
- **Validation attendue** : je dois pouvoir lancer `streamlit run app.py`, uploader un fichier (sans effet pour l'instant), taper un message dans le chat et le voir apparaître dans l'historique.

## Étape 2 — Pipeline d'ingestion
Déclenché quand l'utilisateur a uploadé des fichiers et cliqué sur le bouton d'indexation.
1. **Extraction** : lire les fichiers uploadés avec `PyMuPDFLoader` (PDF) ou `TextLoader` (txt/md).
2. **Chunking** : découper le texte extrait (utiliser `RecursiveCharacterTextSplitter` ou équivalent). **Choisis une taille de chunk et un overlap, et justifie ce choix dans un commentaire dans le code** — c'est une exigence explicite de l'énoncé du TP, pas un détail à passer sous silence.
3. **Vectorisation** : encoder les chunks avec `HuggingFaceEmbeddings` et les stocker dans ChromaDB. Conserve le nom du fichier source dans les métadonnées de chaque chunk (obligatoire pour l'étape 3).
- **Validation attendue** : après indexation, je dois pouvoir vérifier (par exemple via un print/log temporaire ou un affichage debug) que la base ChromaDB contient bien des documents avec leurs métadonnées.

## Étape 3 — Mode "Recherche Sémantique" (toggle désactivé)
- Quand l'utilisateur pose une question et que le toggle LLM est désactivé : **aucun appel à un modèle génératif**, quel qu'il soit.
- Interroger la base vectorielle (`similarity_search` ou retriever équivalent) pour récupérer les fragments les plus proches sémantiquement de la question.
- Afficher dans le chat : le contenu brut de chaque chunk retourné + le nom du fichier source (récupéré depuis les métadonnées).
- **Validation attendue** : uploader un document réel, l'indexer, poser une question en rapport avec son contenu, toggle désactivé, et voir apparaître les extraits pertinents avec leur source.

## Étape 4 — Mode "RAG complet" (toggle activé)
Ollama tourne en local avec le modèle `mistral` disponible (API sur `http://localhost:11434`, compatible avec `langchain-community` via `Ollama`/`ChatOllama`).

1. **Récupération du contexte** : réutilise la même recherche que l'étape 3 (`similarity_search` / retriever) pour obtenir les chunks pertinents.
2. **Ingénierie de prompt** : construis un `PromptTemplate` avec un texte d'instruction strict, ordonnant au modèle de répondre **exclusivement** à partir du `{context}` fourni, et de dire clairement s'il ne trouve pas la réponse dans les documents plutôt que d'inventer. Injecte `{context}` (les chunks concaténés) et `{question}` dans ce template.
3. **Appel du LLM local** : envoie le prompt final au modèle `mistral` via Ollama (pas d'appel réseau externe — uniquement `localhost:11434`).
4. **Affichage de la réponse** dans le chat comme les autres messages (`st.chat_message`).
5. **Transparence (exigence du sujet)** : sous la réponse, ajoute un `st.expander("Sources utilisées")` dépliable qui montre les extraits de chunks + noms de fichiers ayant servi de contexte — exactement comme à l'étape 3, mais replié par défaut.

- **Validation attendue** : toggle activé, poser une question sur un document indexé, obtenir une réponse générée par mistral, et pouvoir déplier les sources qui l'ont justifiée. Toggle désactivé, revérifier que le mode étape 3 (sans LLM) fonctionne toujours normalement — les deux modes doivent cohabiter sans se casser mutuellement.
- Si le premier appel est lent (chargement du modèle en mémoire), c'est normal — les appels suivants sont plus rapides tant qu'Ollama garde le modèle chargé.

## Consignes de travail
- Avance étape par étape (1 → 2 → 3), pas tout en un bloc. Après chaque étape, indique-moi comment la tester avant de passer à la suivante.
- Si une API LangChain a changé de nom/signature selon la version installée, demande-moi la version exacte plutôt que de deviner.
- Commente le code en français, clairement, en gardant à l'esprit que ce fichier sera rendu tel quel.