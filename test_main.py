import streamlit as st
import os
import json
import streamlit_mermaid as st_mermaid

# Importations de NOS modules
import utils
from modules import auth, files, rag, generators

# 1. Configuration Initiale
utils.init_folders()
utils.setup_page(st)

# 2. Gestion Session
if "user_session" not in st.session_state: st.session_state["user_session"] = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔒 Authentification")
    
    if st.session_state["user_session"]:
        user_info = st.session_state["user_session"]
        username = user_info['auth']['username_display']
        role = user_info["profil"].get("role", "Astronaute")
        st.success(f"{role} : **{username}**")
        
        # --- BLOC PROFIL (Utilisation des fonctions auth) ---
        with st.expander("👤 Dossier Personnel"):
            # ... (Copiez ici le code de formulaire de profil que nous avons fait précédemment) ...
            # Pour faire court dans cet exemple :
            if st.button("Se déconnecter"):
                st.session_state["user_session"] = None
                st.rerun()

        st.divider()
        st.header("⚙️ Mode de Sortie")
        selected_mode = st.radio("Format :", list(rag.PROMPT_MODES.keys()))
        
        st.divider()
        uploaded = st.file_uploader("Ajouter PDF", type="pdf")
        if uploaded:
            with open(os.path.join(utils.COURS_FOLDER, uploaded.name), "wb") as f: 
                f.write(uploaded.getbuffer())
            st.success("Archivé !")

    else:
        # Onglets Connexion / Inscription
        t1, t2 = st.tabs(["Connexion", "Inscription"])
        with t1:
            with st.form("login"):
                u = st.text_input("User"); p = st.text_input("Pass", type="password")
                if st.form_submit_button("Go"):
                    d, m = auth.verify_credentials(u, p)
                    if d: st.session_state["user_session"] = d; st.rerun()
                    else: st.error(m)
        with t2:
            with st.form("sign"):
                # Champs d'inscription...
                u = st.text_input("New User"); p = st.text_input("New Pass", type="password")
                # ... Ajoutez les selectbox ici
                if st.form_submit_button("Créer"):
                    # auth.create_user(...)
                    pass

# --- MAIN CHAT ---
if not st.session_state["user_session"]: st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Votre question..."):
    # 1. Affichage User
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # 2. Contextualisation (NOUVEAU)
        real_prompt = prompt
        if len(st.session_state.messages) > 1:
            with st.status("🧠 Analyse du contexte...", expanded=False) as status:
                history = st.session_state.messages[:-1]
                new_prompt = rag.contextualize_question(prompt, history)
                if new_prompt != prompt:
                    real_prompt = new_prompt
                    status.write(f"Question reformulée : {real_prompt}")
                    status.update(label="Contexte compris !", state="complete")
                else:
                    status.update(label="Pas de contexte nécessaire.", state="complete")

        # 3. Smart Router (Sur prompt original ou reformulé, au choix)
        file_path = files.smart_file_router(real_prompt, utils.COURS_FOLDER)
        if file_path:
            fname = os.path.basename(file_path)
            st.success(f"Document trouvé : {fname}")
            with open(file_path, "rb") as f:
                st.download_button("Télécharger", f, file_name=fname)
            real_prompt += f" (Note: Fichier {fname} proposé.)"

        # 4. RAG (Sur le prompt reformulé)
        rel_files, _ = files.get_relevant_files(real_prompt, utils.COURS_FOLDER)
        if rel_files:
            with st.spinner("Analyse..."):
                custom_instr = rag.PROMPT_MODES[selected_mode] if selected_mode != "Chat Standard" else None
                chain = rag.initialize_rag_chain_dynamic(rel_files, st.session_state["user_session"], custom_instr)
                
                if chain:
                    res = chain.invoke({"input": real_prompt})
                    raw = res["answer"]
                    
                    # Nettoyage
                    clean_txt = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
                    
                    # Affichage selon mode
                    if selected_mode == "Chat Standard":
                        st.markdown(clean_txt)
                    elif selected_mode == "🧠 Carte Mentale":
                        code = generators.clean_mermaid_code(clean_txt)
                        try:
                            st_mermaid.st_mermaid(code, height="500px")
                        except:
                            st.error("Erreur d'affichage du graphique.")
                            with st.expander("Voir le code"): st.code(code)
                    # ... autres modes ...
                    
                    st.session_state.messages.append({"role": "assistant", "content": clean_txt})