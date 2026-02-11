import streamlit as st
import os
import json
import glob
import re
import hashlib
from collections import Counter

# --- IMPORTATIONS LANGCHAIN (Standards) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Spaceflight Institute", page_icon="🚀", layout="wide")
st.title("🤖 Spaceflight Institute - Recherche Intelligente")

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# --- GESTION DES DOSSIERS ---
base_folder = "data"
cours_folder = os.path.join(base_folder, "cours")
users_folder = os.path.join(base_folder, "users")

for folder in [base_folder, cours_folder, users_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 1. FONCTIONS AUTHENTIFICATION (Identique précédent) ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_filepath(username):
    safe_name = re.sub(r'[^a-z0-9]', '', username.lower())
    return os.path.join(users_folder, f"{safe_name}.json")

def create_user(username, password, level, tone):
    filepath = get_user_filepath(username)
    if os.path.exists(filepath): return False, "Utilisateur existant."
    data = {
        "auth": {"username_display": username, "password_hash": hash_password(password)},
        "profil": {"niveau": level, "preferences_apprentissage": {"ton": tone}}
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True, "Succès"
    except: return False, "Erreur écriture"

def verify_credentials(username, password):
    filepath = get_user_filepath(username)
    if not os.path.exists(filepath): return None, "Inconnu"
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        if data["auth"]["password_hash"] == hash_password(password): return data, "Succès"
        return None, "Mot de passe faux"
    except: return None, "Erreur fichier"

# --- 2. FONCTIONS FICHIERS & CONTENU ---

def get_relevant_files(prompt, pdf_folder_path):
    """
    Tente de trouver des fichiers par nom. 
    Retourne TOUS les fichiers si pas de correspondance précise (pour laisser le RAG chercher dedans).
    """
    all_pdfs = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not prompt or not all_pdfs: return all_pdfs, True 

    mots_vides = ["le", "la", "les", "de", "du", "des", "un", "une", "fichier", "cours", "pdf", "sur"]
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
    keywords = [word for word in cleaned_prompt.split() if word not in mots_vides and len(word) > 2]
    
    selected_files = []
    for pdf_path in all_pdfs:
        filename = os.path.basename(pdf_path).lower()
        if any(kw in filename for kw in keywords):
            selected_files.append(pdf_path)
            
    # Si on trouve des fichiers par nom, on est contents
    if selected_files:
        return list(set(selected_files)), False
    
    # Sinon, on renvoie TOUT pour que le RAG cherche le contenu "puits de potentiel" partout
    return all_pdfs, True


# 1. On définit la structure de réponse attendue (JSON strict)
class FileRequest(BaseModel):
    veut_telecharger: bool = Field(description="Vrai si l'utilisateur demande explicitement un document, un article, un pdf ou une source complète.")
    nom_fichier_cible: str = Field(description="Le nom exact du fichier PDF parmi la liste fournie qui correspond le mieux à la demande. Vide si aucun fichier ne correspond.")
    raisonnement: str = Field(description="Pourquoi ce fichier a été choisi.")

def smart_file_router(user_prompt, folder_path):
    """
    Utilise l'IA pour décider quel fichier proposer, basé sur le sens de la phrase
    plutôt que sur des mots-clés.
    """
    # 1. Récupérer la liste des fichiers réels
    all_pdfs = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.pdf"))]
    
    if not all_pdfs:
        return None

    # 2. Préparer le LLM (On utilise le même modèle)
    llm = Ollama(model="mistral", temperature=0) # Température 0 pour être logique et strict
    
    # 3. Le Prompt du "Bibliothécaire"
    router_prompt = (
        "Tu es un bibliothécaire intelligent gérant une base de données de fichiers PDF. "
        "Ta mission est UNIQUEMENT d'analyser si l'utilisateur veut récupérer un fichier spécifique.\n"
        "Voici la liste EXACTE des fichiers disponibles dans ta bibliothèque :\n"
        f"{json.dumps(all_pdfs)}\n\n"
        "Analyses la demande de l'utilisateur. Si l'utilisateur demande un document, un article, un cours, ou une référence "
        "qui semble correspondre au SUJET d'un des fichiers, tu dois l'identifier.\n"
        "Exemple : Si l'utilisateur demande 'L'article de Palessandro' et que tu as 'Physique_Review_2024.pdf', et que tu sais ou devines que c'est lié, tu le sélectionnes.\n"
        "Attention : Réponds UNIQUEMENT au format JSON strict."
    )
    
    # 4. Création de la chaîne
    parser = JsonOutputParser(pydantic_object=FileRequest)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", router_prompt),
        ("human", "{input}\n\nFormat JSON attendu:\n{format_instructions}")
    ])
    
    chain = prompt_template | llm | parser
    
    try:
        # Lancement de l'analyse
        result = chain.invoke({
            "input": user_prompt,
            "format_instructions": parser.get_format_instructions()
        })
        
        # 5. Vérification du résultat
        if result["veut_telecharger"] and result["nom_fichier_cible"] in all_pdfs:
            # On retourne le chemin complet
            return os.path.join(folder_path, result["nom_fichier_cible"])
            
    except Exception as e:
        print(f"Erreur du routeur intelligent : {e}")
        # En cas d'erreur (si le LLM hallucine le JSON), on ne fait rien par sécurité
        return None
        
    return None

# --- 3. INITIALISATION RAG ---
def initialize_rag_chain_dynamic(selected_files, user_data):
    profil = user_data.get("profil", {})
    user_name = user_data.get("auth", {}).get("username_display", "Étudiant")
    
    all_pages = []
    for pdf_path in selected_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_pages.extend(loader.load())
        except: pass

    if not all_pages: return None

    # Vectorisation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    # On demande 4 morceaux de contexte
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="mistral")
    
    system_prompt = (
        f"Tu es un tuteur pour {user_name}. Niveau : {profil.get('niveau')}. "
        f"Ton : {profil.get('preferences_apprentissage', {}).get('ton')}. "
        "Utilise le contexte pour répondre."
        "\n\n{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = create_stuff_documents_chain(llm, prompt_template)
    
    # IMPORTANT : create_retrieval_chain renvoie le contexte ("source_documents") dans la réponse
    rag = create_retrieval_chain(retriever, chain)
    return rag

# --- GESTION SESSION ---
if "user_session" not in st.session_state: st.session_state["user_session"] = None

# --- SIDEBAR (LOGIN) ---
with st.sidebar:
    st.header("🔒 Authentification")
    if st.session_state["user_session"]:
        st.success(f"Connecté : **{st.session_state['user_session']['auth']['username_display']}**")
        if st.button("Se déconnecter"):
            st.session_state["user_session"] = None
            st.rerun()
        st.divider()
        uploaded = st.file_uploader("Ajouter PDF", type="pdf")
        if uploaded:
            with open(os.path.join(cours_folder, uploaded.name), "wb") as f: f.write(uploaded.getbuffer())
            st.success("Ajouté !")
    else:
        t1, t2 = st.tabs(["Log", "Sign"])
        with t1:
            with st.form("l"):
                u, p = st.text_input("User"), st.text_input("Pass", type="password")
                if st.form_submit_button("Go"):
                    d, m = verify_credentials(u, p)
                    if d: st.session_state["user_session"] = d; st.rerun()
                    else: st.error(m)
        with t2:
            with st.form("s"):
                u, p = st.text_input("New User"), st.text_input("New Pass", type="password")
                l, t = st.select_slider("Niveau", ["A", "B", "C"]), st.selectbox("Style", ["Cool", "Strict"])
                if st.form_submit_button("Créer"):
                    create_user(u, p, l, t)
                    st.success("Crée !")

# --- ZONE PRINCIPALE ---
if not st.session_state["user_session"]: st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# --- ZONE DE CHAT ---

# 1. Vérification de sécurité
if not st.session_state["user_session"]:
    st.info("👋 Veuillez vous connecter dans la barre latérale pour accéder à l'assistant.")
    st.stop()

# 2. Affichage de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Boucle principale (Quand l'utilisateur tape Entrée)
if prompt := st.chat_input("Posez votre question (ex: 'Donne moi l'article de Palessandro')..."):
    
    # A. Afficher le message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Réponse de l'Assistant
    with st.chat_message("assistant"):
        
        found_file_path = None
        
        # --- ÉTAPE 1 : ROUTEUR INTELLIGENT ---
        # On affiche un petit statut pendant que l'IA cherche le fichier
        with st.status("🔍 Recherche dans la bibliothèque...", expanded=False) as status:
            found_file_path = smart_file_router(prompt, cours_folder)
            
            if found_file_path:
                status.update(label="Document trouvé !", state="complete")
            else:
                status.update(label="Analyse du contenu...", state="complete")

        # Si le routeur a trouvé un fichier, on affiche le bouton IMMÉDIATEMENT
        if found_file_path:
            filename = os.path.basename(found_file_path)
            st.success(f"📂 J'ai trouvé ce document pour vous : **{filename}**")
            
            with open(found_file_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Télécharger {filename}",
                    data=f.read(),
                    file_name=filename,
                    mime="application/pdf"
                )
            
            # On ajoute une note invisible pour que l'IA sache qu'elle a déjà donné le fichier
            prompt += f" (Note système : Tu as déjà proposé le fichier {filename} en téléchargement. Confirme-le poliment.)"

        # --- ÉTAPE 2 : RAG (Réponse textuelle) ---
        # On récupère les fichiers pour le contexte (méthode classique)
        relevant_files, is_global = get_relevant_files(prompt, cours_folder)
        
        if relevant_files:
            with st.spinner("Rédaction de la réponse..."):
                try:
                    # On lance la chaîne RAG avec le profil utilisateur
                    rag_chain = initialize_rag_chain_dynamic(relevant_files, st.session_state["user_session"])
                
                    if rag_chain:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        # Nettoyage des balises <think> de DeepSeek
                        if "</think>" in answer:
                            answer = answer.split("</think>")[-1].strip()
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")
        else:
            st.warning("Je n'ai aucun document pour répondre à cette demande.")