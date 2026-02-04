import streamlit as st
import os
import json
import glob
import re
import hashlib

# --- IMPORTATIONS LANGCHAIN (Standards) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Spaceflight Institute", page_icon="🚀", layout="wide")
st.title("🤖 Spaceflight I(A)nstitute - Accès Sécurisé")

# Configuration Proxy
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# --- GESTION DES DOSSIERS ---
base_folder = "data"
cours_folder = os.path.join(base_folder, "cours")
users_folder = os.path.join(base_folder, "users")

# Création automatique des dossiers
for folder in [base_folder, cours_folder, users_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 1. FONCTIONS DE SÉCURITÉ ET GESTION UTILISATEURS ---

def hash_password(password):
    """Transforme un mot de passe en empreinte SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_filepath(username):
    """Génère un nom de fichier standardisé"""
    safe_name = re.sub(r'[^a-z0-9]', '', username.lower())
    return os.path.join(users_folder, f"{safe_name}.json")

def create_user(username, password, level, tone):
    filepath = get_user_filepath(username)
    
    if os.path.exists(filepath):
        return False, "Cet utilisateur existe déjà."
    
    # Structure JSON sécurisée : Auth séparé du Profil
    data = {
        "auth": {
            "username_display": username,
            "password_hash": hash_password(password) # Stockage sécurisé
        },
        "profil": {
            "niveau": level,
            "preferences_apprentissage": {
                "ton": tone,
                "contenu_prefere": "mixte"
            }
        }
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True, "Compte créé avec succès !"
    except Exception as e:
        return False, f"Erreur d'écriture : {e}"

def verify_credentials(username, password):
    """Vérifie le couple user/password"""
    filepath = get_user_filepath(username)
    
    if not os.path.exists(filepath):
        return None, "Utilisateur inconnu."
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        stored_hash = data["auth"].get("password_hash")
        input_hash = hash_password(password)
        
        if stored_hash == input_hash:
            return data, "Succès"
        else:
            return None, "Mot de passe incorrect."
    except Exception as e:
        return None, f"Erreur lecture fichier : {e}"

# --- 2. SÉLECTION INTELLIGENTE DES FICHIERS (Ta version) ---
def get_relevant_files(prompt, pdf_folder_path):
    all_pdfs = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not prompt or not all_pdfs:
        return all_pdfs, True # True = Recherche globale

    # Nettoyage simple du prompt pour extraire des mots-clés
    mots_vides = ["le", "la", "les", "de", "du", "des", "un", "une", "est", "sont", "comment", "quoi"]
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
    keywords = [word for word in cleaned_prompt.split() if word not in mots_vides and len(word) > 2]
    
    selected_files = []
    for pdf_path in all_pdfs:
        filename = os.path.basename(pdf_path).lower()
        if any(kw in filename for kw in keywords):
            selected_files.append(pdf_path)
            
    # Cas 2 : Aucun mot clé trouvé -> Fallback global
    if not selected_files:
        return all_pdfs, True
    
    # Cas 3 : Sélection précise
    return list(set(selected_files)), False

# --- 3. INITIALISATION RAG (Adapté à la structure JSON sécurisée) ---
def initialize_rag_chain_dynamic(selected_files, user_data):
    
    # Lecture dans la section "profil" et "auth" du nouveau JSON
    profil = user_data.get("profil", {})
    user_name = user_data.get("auth", {}).get("username_display", "Étudiant")
    user_level = profil.get("niveau", "Intermédiaire")
    ai_tone = profil.get("preferences_apprentissage", {}).get("ton", "neutre")

    # Chargement PDF
    all_pages = []
    for pdf_path in selected_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_pages.extend(loader.load())
        except Exception as e:
            print(f"Erreur fichier {pdf_path}: {e}")

    if not all_pages:
        return None

    # Vectorisation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Modèle
    llm = Ollama(model="deepseek-r1:8b")
    
    # Prompt personnalisé
    system_prompt = (
        f"Tu es un tuteur personnel pour {user_name}. "
        f"Niveau de l'élève : {user_level}. "
        f"Ton style pédagogique doit être : {ai_tone}. "
        "Utilise le contexte fourni pour répondre. Si tu ne sais pas, dis-le."
        "\n\n"
        "{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    
    chain = create_stuff_documents_chain(llm, prompt_template)
    rag = create_retrieval_chain(retriever, chain)
    return rag

# --- GESTION SESSION ---
if "user_session" not in st.session_state:
    st.session_state["user_session"] = None

# --- INTERFACE SIDEBAR (AUTH SÉCURISÉE) ---
with st.sidebar:
    st.header("🔒 Authentification")
    
    # Cas 1 : Utilisateur Connecté
    if st.session_state["user_session"]:
        user_name = st.session_state["user_session"]["auth"]["username_display"]
        st.success(f"Connecté : **{user_name}**")
        
        if st.button("Se déconnecter"):
            st.session_state["user_session"] = None
            st.rerun()
            
        st.divider()
        st.header("📚 Bibliothèque")
        uploaded_file = st.file_uploader("Ajouter un cours (PDF)", type="pdf")
        if uploaded_file:
            file_path = os.path.join(cours_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Cours ajouté !")

    # Cas 2 : Utilisateur Non Connecté
    else:
        tab_login, tab_signup = st.tabs(["Connexion", "Créer compte"])
        
        with tab_login:
            with st.form("login_form"):
                u_input = st.text_input("Identifiant")
                p_input = st.text_input("Mot de passe", type="password")
                if st.form_submit_button("Entrer"):
                    data, msg = verify_credentials(u_input, p_input)
                    if data:
                        st.session_state["user_session"] = data
                        st.success("Connexion réussie !")
                        st.rerun()
                    else:
                        st.error(msg)
                        
        with tab_signup:
            with st.form("signup_form"):
                new_user = st.text_input("Nouvel Identifiant")
                new_pass = st.text_input("Nouveau mot de passe", type="password")
                st.markdown("**Préférences :**")
                new_level = st.select_slider("Niveau", options=["Débutant", "Intermédiaire", "Expert"])
                new_tone = st.selectbox("Style IA", ["Strict & Concis", "Pédagogique & Illustré", "Socratique", "Fun & Détendu"])
                
                if st.form_submit_button("S'inscrire"):
                    if new_user and new_pass:
                        ok, msg = create_user(new_user, new_pass, new_level, new_tone)
                        if ok:
                            st.success("Compte créé ! Connectez-vous.")
                        else:
                            st.error(msg)
                    else:
                        st.warning("Tout remplir SVP.")

# --- ZONE PRINCIPALE (PROTECTION) ---
if not st.session_state["user_session"]:
    st.info("👋 Veuillez vous connecter dans la barre latérale pour accéder à l'assistant.")
    st.stop()

# --- HISTORIQUE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ZONE DE CHAT (Ta logique de tri conservée) ---
if prompt := st.chat_input("Posez votre question sur les cours..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # 1. Sélection intelligente des fichiers
        relevant_files, is_global_search = get_relevant_files(prompt, cours_folder)
        
        # Feedback UI
        files_names = [os.path.basename(f) for f in relevant_files]
        
        if is_global_search:
            st.warning("⚠️ Recherche globale (aucun fichier spécifique détecté dans le titre).")
            with st.expander("Voir les fichiers utilisés"):
                st.write(files_names)
        else:
            st.success(f"🎯 Ciblage réussi sur : {', '.join(files_names)}")

        # 2. Lancement du RAG avec le profil utilisateur connecté
        if relevant_files:
            with st.spinner("Analyse en cours..."):
                try:
                    rag_chain = initialize_rag_chain_dynamic(relevant_files, st.session_state["user_session"])
                
                    if rag_chain:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        # Nettoyage DeepSeek
                        if "</think>" in answer:
                            answer = answer.split("</think>")[-1].strip()
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erreur technique : {e}")
        else:
            st.error("Aucun document disponible dans la bibliothèque.")