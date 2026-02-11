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

# --- SIDEBAR (LOGIN & PROFIL) ---
with st.sidebar:
    st.header("🔒 Authentification")
    
    # CAS 1 : UTILISATEUR CONNECTÉ
    if st.session_state["user_session"]:
        user_info = st.session_state["user_session"]
        username = user_info['auth']['username_display']
        
        # Récupération du rôle pour l'affichage (ex: Astronaute Nayel)
        display_role = user_info["profil"].get("role", "Astronaute")
        st.success(f"{display_role} : **{username}**")
        
        # --- BLOC UNIQUE : MODIFIER MON PROFIL ---
        with st.expander("👤 Dossier Personnel"):
            st.caption("Mettez à jour vos accréditations.")
            
            # --- 1. RECUPERATION DONNEES ACTUELLES (Avec Fallback) ---
            raw_level = user_info["profil"].get("niveau", "Intermédiaire")
            raw_tone = user_info["profil"].get("preferences_apprentissage", {}).get("ton", "Neutre")
            raw_role = user_info["profil"].get("role", "Cadet")
            raw_goal = user_info["profil"].get("objectif", "")

            # --- 2. LISTES D'OPTIONS ---
            list_niveaux = ["Débutant", "Intermédiaire", "Avancé", "Expert"]
            list_styles = ["Cool", "Strict", "Neutre", "Scientifique", "Militaire"]
            list_roles = ["Cadet", "Pilote", "Ingénieur", "Scientifique", "Commandant", "Touriste"]

            # --- 3. LOGIQUE DE SECURITE (Pour anciens profils) ---
            # Niveau
            if raw_level not in list_niveaux:
                mapping_niveau = {"A": "Débutant", "B": "Intermédiaire", "C": "Avancé"}
                default_level = mapping_niveau.get(raw_level, "Intermédiaire")
            else:
                default_level = raw_level
            
            # Style
            default_tone = raw_tone if raw_tone in list_styles else "Neutre"
            
            # Rôle
            default_role = raw_role if raw_role in list_roles else "Cadet"

            # --- 4. FORMULAIRE ---
            with st.form("update_profile_full"):
                st.markdown("**Identité**")
                new_role = st.selectbox("Spécialité", list_roles, index=list_roles.index(default_role))
                new_goal = st.text_input("Objectif de Mission", value=raw_goal, placeholder="Ex: Certification Moteur Raptor")
                
                st.markdown("**Préférences IA**")
                new_level = st.select_slider("Niveau d'expertise", options=list_niveaux, value=default_level)
                new_tone = st.selectbox("Ton de l'IA", list_styles, index=list_styles.index(default_tone))
                
                if st.form_submit_button("💾 Mettre à jour le dossier"):
                    updated_data = auth.update_user_profile(username, new_level, new_tone, new_role, new_goal)
                    if updated_data:
                        st.session_state["user_session"] = updated_data
                        st.toast("Dossier mis à jour !", icon="✅")
                        st.rerun()
                    else:
                        st.error("Erreur mise à jour.")

        st.divider()
        
        # --- SÉLECTION DU MODE ---
        st.header("⚙️ Mode de Sortie")
        selected_mode = st.radio(
            "Format de la réponse :",
            list(rag.PROMPT_MODES.keys()),
            key="mode_radio_main" 
        )
        
        st.divider()
        
        if st.button("Se déconnecter", key="logout_btn_main"):
            st.session_state["user_session"] = None
            st.rerun()
            
        st.divider()
        
        uploaded = st.file_uploader("Ajouter PDF", type="pdf", key="pdf_uploader_main")
        if uploaded:
            save_path = os.path.join(utils.COURS_FOLDER, uploaded.name)
            with open(save_path, "wb") as f: 
                f.write(uploaded.getbuffer())
            st.success("Document archivé !")

    # CAS 2 : UTILISATEUR NON CONNECTÉ
    else:
        t1, t2 = st.tabs(["Connexion", "Inscription"])
        
        with t1:
            with st.form("login_form"):
                u = st.text_input("Matricule (User)")
                p = st.text_input("Code d'accès (Pass)", type="password")
                if st.form_submit_button("S'identifier"):
                    d, m = auth.verify_credentials(u, p)
                    if d: 
                        st.session_state["user_session"] = d
                        st.rerun()
                    else: 
                        st.error(m)
        
        with t2:
            st.markdown("Rejoindre le **Spaceflight Institute**")
            with st.form("signup_form"):
                u = st.text_input("Nouveau Matricule")
                p = st.text_input("Créer Code d'accès", type="password")
                
                c1, c2 = st.columns(2)
                with c1:
                    role_signup = st.selectbox("Spécialité", ["Cadet", "Pilote", "Ingénieur", "Scientifique", "Commandant", "Touriste"])
                with c2:
                    level_signup = st.selectbox("Niveau", ["Débutant", "Intermédiaire", "Avancé", "Expert"])
                
                goal_signup = st.text_input("Objectif Principal", placeholder="Ex: Apprendre la physique orbitale")
                tone_signup = st.selectbox("Style de l'assistant", ["Cool", "Strict", "Neutre", "Scientifique", "Militaire"])
                
                if st.form_submit_button("Initialiser le profil"):
                    created, msg = auth.create_user(u, p, level_signup, tone_signup, role_signup, goal_signup)
                    if created:
                        st.success("Profil créé ! Connectez-vous.")
                    else:
                        st.error(msg)

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

        # 3. Smart Router 
        file_path = files.smart_file_router(real_prompt, utils.COURS_FOLDER)
        if file_path:
            fname = os.path.basename(file_path)
            st.success(f"Document trouvé : {fname}")
            with open(file_path, "rb") as f:
                st.download_button("Télécharger", f, file_name=fname)
            real_prompt += f" (Note: Fichier {fname} proposé.)"

        # 4. RAG 
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