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
    "🎙️ Résumé Audio (Podcast)": "Script podcast dynamique sans ponctuation lue.",
    "🧠 Carte Mentale": "Bloc code Mermaid graph TD uniquement.",
    "📝 Fiches de Révision": "JSON strict liste objets question/reponse.",
    "📊 Diapositives (PPTX)": "JSON strict liste diapositives titre/points."
}

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
        except: pass

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