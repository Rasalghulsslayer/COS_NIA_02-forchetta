from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

PROMPT_MODES = {
    "Chat Standard": "Réponds de manière naturelle.",
    
    "🎙️ Audio (Podcast)": (
        "Adopte le persona d'un animateur de podcast dynamique. "
        "Ton but est de créer un script oral engageant. "
        "Ne mets pas de balises de mise en scène (musique, applaudissements), juste le texte parlé."
        "Ne lis SURTOUT PAS la ponctuation."
    ),
    
    "🧠 Visual card": (
        "Tu es un expert en visualisation de données. "
        "Ta mission : Créer une carte mentale hiérarchique avec Mermaid.js. "
        "RÈGLES DE SYNTAXE STRICTES :\n"
        "1. Utilise UNIQUEMENT la syntaxe 'graph TD' (et non 'mindmap').\n"
        "2. IMPORTANT : Mets TOUS les textes des nœuds entre guillemets doubles. Ex: id[\"Mon Texte Ici\"]\n"
        "3. Ne mets JAMAIS de parenthèses () directement dans le texte sans guillemets.\n"
        "4. Réponds UNIQUEMENT avec le bloc de code ```mermaid ... ```.\n\n"
        "Exemple valide :\n"
        "```mermaid\n"
        "graph TD\n"
        "  A[\"Sujet Principal\"] --> B[\"Idée 1 (Détail)\"]\n"
        "  A --> C[\"Idée 2 : L'exemple\"]\n"
        "```"
    ),
    
    "📝 Flash card": (
        "Ton but est de créer un outil de mémorisation. "
        "Tu DOIS répondre UNIQUEMENT au format JSON strict (liste d'objets). "
        "Format attendu : [{{'question': '...', 'reponse': '...'}}, {{'question': '...', 'reponse': '...'}}]"
    ),
    
    "📊 Slides (PPTX)": (
        "Ton but est de synthétiser pour une présentation. "
        "Tu DOIS répondre UNIQUEMENT au format JSON strict. "
        "Format attendu : [{{'titre': '...', 'points': ['...', '...']}}, ...]"
    )
}

def contextualize_question(input_question, chat_history):
    """
    Reformule la question utilisateur en intégrant le contexte de l'historique.
    """
    if not chat_history:
        return input_question
    
    last_exchanges = chat_history[-3:] 
    history_str = ""
    for msg in last_exchanges:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 200: 
            content = content[:200] + "..."
        history_str += f"{role}: {content}\n"
    
    llm = Ollama(model="deepseek-r1:8b", temperature=0.1)
    
    reformulation_prompt = (
        "Tu es un expert en linguistique. Ta tâche est de réécrire la dernière question de l'utilisateur "
        "pour qu'elle soit autonome et compréhensible sans l'historique de la conversation.\n\n"
        "--- HISTORIQUE ---\n"
        f"{history_str}\n"
        "------------------\n"
        f"Dernière question de l'Utilisateur : '{input_question}'\n\n"
        "CONSIGNES :\n"
        "1. Remplace les pronoms (il, elle, ça, le...) par les noms précis présents dans l'historique.\n"
        "2. Si la question est déjà claire, ne change rien.\n"
        "3. Réponds UNIQUEMENT par la question reformulée. Pas de politesse, pas de balises.\n"
    )
    
    try:
        response = llm.invoke(reformulation_prompt)
        if "</think>" in response:
            response = response.split("</think>")[-1]
        return response.replace('"', '').strip()
    except Exception as e:
        print(f"Reformulation Error: {e}")
        return input_question

def initialize_rag_chain_dynamic(selected_files, user_data, custom_prompt=None):
    profil = user_data.get("profil", {})
    user_info = {
        "name": user_data.get("auth", {}).get("username_display", "Étudiant"),
        "role": profil.get("role", "Cadet"),
        "level": profil.get("niveau", "Intermédiaire"),
        "goal": profil.get("objectif", "Apprendre"),
        "tone": profil.get("preferences_apprentissage", {}).get("ton", "neutre")
    }

    all_pages = []
    for pdf_path in selected_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_pages.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")

    if not all_pages: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="deepseek-r1:8b", temperature=0)
    
    format_instr = custom_prompt if custom_prompt else "Réponds normalement."
    
    system_prompt = (
        f"Tu es un assistant pour {user_info['name']} ({user_info['role']}). "
        f"Niveau: {user_info['level']}. Objectif: {user_info['goal']}. Ton: {user_info['tone']}. "
        "\n\n--- FORMAT ---\n" + format_instr + "\n"
        "--- CONTEXTE ---\n{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, chain)