import streamlit as st
import os
import json
import glob
import re
import random
import tempfile
from datetime import datetime, timedelta
from itertools import cycle

from streamlit_calendar import calendar
from fpdf import FPDF # <--- Il faut avoir fait pip install fpdf

# --- IMPORTATIONS LANGCHAIN ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Spaceflight Institute", page_icon="🚀", layout="wide")
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# --- 2. STYLES CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
        table { width: 100%; border-collapse: separate; border-radius: 10px; overflow: hidden; border: 1px solid #4A4A4A; background-color: #1E1E1E; color: #E0E0E0; font-family: 'Courier New', Courier, monospace; }
        thead tr th { background: linear-gradient(90deg, #2b5876 0%, #4e4376 100%); color: #ffffff; font-weight: bold; text-transform: uppercase; padding: 15px; border-bottom: 2px solid #FFD700; }
        tbody tr td { padding: 12px; border-bottom: 1px solid #333333; }
        tbody tr:hover { background-color: #2C2C2C; cursor: default; }
        td:first-child { font-weight: bold; color: #4db8ff; }
        .stButton button { width: 100%; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()
st.title("🤖 Spaceflight I(A)nstitute")

# --- 3. GESTION DOSSIERS ---
base_folder = "data"
cours_folder = os.path.join(base_folder, "cours")
users_folder = os.path.join(base_folder, "users")

for folder in [base_folder, cours_folder, users_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 4. FONCTIONS UTILITAIRES ---

def clean_text_for_pdf(text):
    """
    Nettoie le texte pour le PDF :
    1. Remplace les émojis connus par du texte lisible.
    2. Supprime tous les autres caractères non supportés (évite les '?').
    """
    # Remplacement manuel des émojis utilisés dans l'app
    replacements = {
        "⚡": "[FLASH]",
        "📝": "[EXO]",
        "📚": "[COURS]",
        "🚀": "",
        "🏋️": "[SPORT]",
        "🧠": "",
        "🥪": ""
    }
    
    for emoji, text_replace in replacements.items():
        text = text.replace(emoji, text_replace)
        
    # Encodage en Latin-1 en IGNORANT les erreurs (supprime les caractères inconnus au lieu de mettre ?)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_planning_pdf(events, username):
    """Génère le PDF du planning"""
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, clean_text_for_pdf(f'PLANNING DE MISSION : {username}'), 0, 1, 'C')
            self.ln(10)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    days_order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
    # Mapping date -> jour
    date_to_day = {
        "2024-01-01": "Lundi", "2024-01-02": "Mardi", "2024-01-03": "Mercredi",
        "2024-01-04": "Jeudi", "2024-01-05": "Vendredi"
    }
    
    # Organisation des données
    organized_data = {day: [] for day in days_order}
    for event in events:
        date_str = event['start'].split('T')[0]
        day_name = date_to_day.get(date_str)
        if day_name:
            organized_data[day_name].append(event)

    for day in days_order:
        # Titre Jour
        pdf.set_fill_color(200, 220, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, day.upper(), 1, 1, 'L', fill=True)
        
        day_events = organized_data[day]
        day_events.sort(key=lambda x: x['start'])

        if not day_events:
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, "Repos / Libre", 1, 1, 'C')
        
        for evt in day_events:
            start_h = evt['start'].split('T')[1][:5]
            end_h = evt['end'].split('T')[1][:5]
            title = evt['extendedProps']['fullTitle']
            
            # Nettoyage strict pour le PDF
            title_clean = clean_text_for_pdf(title)
            
            pdf.set_font('Arial', '', 10)
            # Rouge pour le sport (#D63031), Noir sinon
            if evt.get('backgroundColor') == "#D63031":
                pdf.set_text_color(200, 0, 0)
            else:
                pdf.set_text_color(0, 0, 0)
                
            pdf.cell(35, 8, f"{start_h} - {end_h}", 1, 0, 'C')
            pdf.cell(0, 8, f" {title_clean}", 1, 1, 'L')
            
        pdf.ln(5)

    # Sauvegarde en mémoire
    try:
        # FPDF renvoie le contenu binaire si dest='S'
        return pdf.output(dest='S').encode('latin-1') 
    except:
        # Fallback si erreur encodage
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        with open(temp_file.name, "rb") as f:
            return f.read()

def get_user_list():
    files = glob.glob(os.path.join(users_folder, "*.json"))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]

def create_user(username, level, tone):
    filename = f"{username.lower().replace(' ', '_')}.json"
    filepath = os.path.join(users_folder, filename)
    data = {
        "utilisateur": {
            "prenom": username,
            "niveau": level,
            "preferences_apprentissage": {"ton": tone, "contenu_prefere": "mixte"}
        }
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return filename

def load_user_preferences(username_file):
    filepath = os.path.join(users_folder, f"{username_file}.json")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def update_user_schedule(username_file, schedule_data):
    filepath = os.path.join(users_folder, f"{username_file}.json")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["utilisateur"]["disponibilites"] = schedule_data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erreur update: {e}")
        return False

# --- 5. LOGIQUE PLANNING ---

def calculate_free_slots(busy_schedule):
    """Calcul strict : Lundi-Vendredi, 08h-18h"""
    free_slots_data = [] 
    day_start_str = "08:00"
    day_end_str = "18:00"
    fmt = "%H:%M"
    lunch_break = {"debut": "12:00", "fin": "13:00", "activite": "PAUSE DEJEUNER"}
    work_days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
    
    for day in work_days:
        activities = busy_schedule.get(day, []).copy()
        activities.append(lunch_break)
        activities.sort(key=lambda x: x['debut'])
        
        cursor = datetime.strptime(day_start_str, fmt)
        end_of_day = datetime.strptime(day_end_str, fmt)
        
        for activity in activities:
            try:
                s = activity['debut'] if len(activity['debut']) == 5 else f"0{activity['debut']}"
                e = activity['fin'] if len(activity['fin']) == 5 else f"0{activity['fin']}"
                act_start = datetime.strptime(s, fmt)
                act_end = datetime.strptime(e, fmt)
                
                if act_start > cursor:
                    free_slots_data.append({
                        "day": day,
                        "start": cursor.strftime(fmt),
                        "end": act_start.strftime(fmt),
                        "duration": (act_start - cursor).total_seconds() / 60
                    })
                cursor = max(cursor, act_end)
            except ValueError: continue

        if cursor < end_of_day:
            free_slots_data.append({
                "day": day,
                "start": cursor.strftime(fmt),
                "end": end_of_day.strftime(fmt),
                "duration": (end_of_day - cursor).total_seconds() / 60
            })
    return free_slots_data

def generate_revision_plan(user_config, pdf_folder):
    """Génération Python pure (Instantané)"""
    user_info = user_config.get("utilisateur", {})
    busy_schedule = user_info.get("disponibilites", {}) 
    
    try:
        strict_slots = calculate_free_slots(busy_schedule)
    except Exception as e:
        return f"Erreur calcul : {e}"

    if not strict_slots:
        return "Aucun temps libre trouvé !"

    files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    course_titles = [os.path.basename(f).replace('.pdf', '').replace('_', ' ') for f in files]
    if not course_titles: course_titles = ["Culture Spatiale", "Mécanique du Vol"]

    random.shuffle(course_titles)
    course_cycle = cycle(course_titles)

    csv_output = ""
    last_course = ""

    for slot in strict_slots:
        duration = slot['duration']
        current_course = next(course_cycle)
        if current_course == last_course:
            current_course = next(course_cycle)
        last_course = current_course

        if duration < 30: prefix = "⚡ Flash Quiz"
        elif duration < 90: prefix = "📝 Exercices"
        else: prefix = "📚 Cours Magistral"

        full_title = f"{prefix} {current_course}"
        csv_output += f"{slot['day']}|{slot['start']}|{slot['end']}|{full_title}\n"

    return csv_output

def parse_schedule_to_events(llm_response, busy_schedule):
    calendar_events = []
    day_map = {"Lundi": "2024-01-01", "Mardi": "2024-01-02", "Mercredi": "2024-01-03",
               "Jeudi": "2024-01-04", "Vendredi": "2024-01-05", "Samedi": "2024-01-06", "Dimanche": "2024-01-07"}

    # 1. Activités Fixes
    if busy_schedule:
        for day, activities in busy_schedule.items():
            date_base = day_map.get(day, "2024-01-01")
            for act in activities:
                s = act['debut'] if len(act['debut']) == 5 else f"0{act['debut']}"
                e = act['fin'] if len(act['fin']) == 5 else f"0{act['fin']}"
                calendar_events.append({
                    "title": act['activite'],
                    "start": f"{date_base}T{s}:00", "end": f"{date_base}T{e}:00",
                    "backgroundColor": "#D63031", "borderColor": "#B83227",
                    "extendedProps": {"fullTitle": act['activite']}
                })

    # 2. Pause
    for day_name, date_base in day_map.items():
        if day_name not in ["Samedi", "Dimanche"]:
            calendar_events.append({
                "title": "PAUSE",
                "start": f"{date_base}T12:00:00", "end": f"{date_base}T13:00:00",
                "backgroundColor": "#FDCB6E", "borderColor": "#E1B12C", "textColor": "#000000",
                "extendedProps": {"fullTitle": "Pause Déjeuner"}
            })

    # 3. Révisions
    try:
        lines = llm_response.strip().split('\n')
        for line in lines:
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    day_str, start_time, end_time, full_title = parts[0], parts[1], parts[2], parts[3]
                    date_base = day_map.get(day_str, "2024-01-01")
                    bg_color = "#6c5ce7" if "⚡" in full_title or "Quiz" in full_title else "#0984e3"
                    calendar_events.append({
                        "title": full_title,
                        "start": f"{date_base}T{start_time}:00", "end": f"{date_base}T{end_time}:00",
                        "backgroundColor": bg_color, "borderColor": "#74b9ff",
                        "extendedProps": {"fullTitle": full_title}
                    })
    except Exception as e: print(e)
    return calendar_events

# --- 6. RAG ---
def get_relevant_files(prompt, pdf_folder_path):
    all_pdfs = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not prompt or not all_pdfs: return all_pdfs, True
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
    keywords = [w for w in cleaned_prompt.split() if len(w) > 2]
    selected_files = [f for f in all_pdfs if any(kw in os.path.basename(f).lower() for kw in keywords)]
    return list(set(selected_files)) if selected_files else all_pdfs, not bool(selected_files)

def initialize_rag_chain_dynamic(selected_files, user_config):
    user_info = user_config.get("utilisateur", {})
    all_pages = []
    for pdf_path in selected_files:
        try: all_pages.extend(PyPDFLoader(pdf_path).load())
        except: pass
    if not all_pages: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    
    llm = Ollama(model="mistral")
    system_prompt = f"Tu es tuteur pour {user_info.get('prenom')}. Style: {user_info.get('preferences_apprentissage', {}).get('ton')}. Contexte:\n{{context}}"
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    return create_retrieval_chain(vectorstore.as_retriever(), create_stuff_documents_chain(llm, prompt_template))

# --- 7. SIDEBAR ---
with st.sidebar:
    st.header("👤 Espace Membre")
    existing_users = get_user_list()
    mode = st.radio("Option", ["Connexion", "Nouveau Profil"], label_visibility="collapsed")
    current_user_data = None
    
    if mode == "Connexion":
        if existing_users:
            selected_user = st.selectbox("Choisir un profil", existing_users)
            current_user_data = load_user_preferences(selected_user)
            if current_user_data:
                st.info(f"Bonjour {current_user_data['utilisateur']['prenom']}")
                st.divider()
                st.subheader("📅 Mon Emploi du temps")
                uploaded_schedule = st.file_uploader("Disponibilités (.json)", type="json")
                if uploaded_schedule:
                    if update_user_schedule(selected_user, json.load(uploaded_schedule)):
                        st.success("Mise à jour réussie !")
                        current_user_data = load_user_preferences(selected_user)
        else: st.warning("Créez un profil.")
    else: 
        with st.form("new_user"):
            new_name = st.text_input("Prénom")
            new_level = st.select_slider("Niveau", ["Débutant", "Intermédiaire", "Expert"])
            new_tone = st.selectbox("Style", ["Strict", "Pédagogique", "Fun"])
            if st.form_submit_button("Créer") and new_name:
                create_user(new_name, new_level, new_tone)
                st.rerun()

    st.divider()
    st.header("📚 Bibliothèque")
    uploaded_file = st.file_uploader("Ajouter PDF", type="pdf")
    if uploaded_file:
        with open(os.path.join(cours_folder, uploaded_file.name), "wb") as f: f.write(uploaded_file.getbuffer())
        st.success("Ajouté !")

# --- 8. LOGIQUE PRINCIPALE ---
if not current_user_data:
    st.info("👈 Connectez-vous d'abord.")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []

# --- A. AFFICHAGE HISTORIQUE (Calendrier + PDF ici pour qu'ils restent !) ---
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "calendar_data" in message:
            # 1. Calendrier
            cal_opts = {
                "editable": False, "headerToolbar": {"left": "", "center": "title", "right": ""},
                "firstDay": 1, "initialView": "timeGridWeek", "initialDate": "2024-01-01",
                "slotMinTime": "08:00:00", "slotMaxTime": "18:00:00", "allDaySlot": False,
                "locale": "fr", "height": "auto", "weekends": False
            }
            calendar(events=message["calendar_data"], options=cal_opts, key=f"cal_{idx}")
            
            # 2. Bouton PDF (Généré à la volée pour rester actif)
            try:
                username = current_user_data["utilisateur"]["prenom"]
                pdf_bytes = create_planning_pdf(message["calendar_data"], username)
                st.download_button(
                    label="📄 Télécharger le Planning (PDF)",
                    data=pdf_bytes,
                    file_name=f"Planning_{username}.pdf",
                    mime="application/pdf",
                    key=f"pdf_btn_{idx}"
                )
            except Exception as e:
                st.error(f"Erreur PDF: {e}")

            # 3. Détails
            with st.expander("📋 Liste détaillée"):
                revs = [e for e in message["calendar_data"] if e.get("backgroundColor") in ["#0984e3", "#6c5ce7"]]
                revs.sort(key=lambda x: x['start'])
                days = {"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi", "Thursday": "Jeudi", "Friday": "Vendredi"}
                for r in revs:
                    s = datetime.strptime(r['start'], "%Y-%m-%dT%H:%M:%S")
                    e_dt = datetime.strptime(r['end'], "%Y-%m-%dT%H:%M:%S")
                    d_fr = days.get(s.strftime("%A"), s.strftime("%A"))
                    st.markdown(f"**{d_fr} {s.strftime('%H:%M')} - {e_dt.strftime('%H:%M')}** : {r['extendedProps']['fullTitle']}")

# --- B. INPUT ---
if prompt := st.chat_input("Votre message (ou 'Planifie ma semaine')"):
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if any(kw in prompt.lower() for kw in ["planif", "agenda", "emploi du temps"]):
            st.caption("🗓️ Génération en cours...")
            schedule = current_user_data.get("utilisateur", {}).get("disponibilites", {})
            raw_csv = generate_revision_plan(current_user_data, cours_folder)
            events = parse_schedule_to_events(raw_csv, schedule)
            
            if isinstance(events, set): events = list(events)
            
            # On sauvegarde le résultat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "✅ Voici votre planning optimisé (Lundi-Vendredi, 08h-18h) :",
                "calendar_data": events
            })
            # IMPORTANT : On force le rechargement pour que la boucle du haut (Historique) affiche tout
            st.rerun()
        
        else:
            # Mode RAG
            files, _ = get_relevant_files(prompt, cours_folder)
            if files:
                with st.spinner("Recherche..."):
                    rag = initialize_rag_chain_dynamic(files, current_user_data)
                    if rag:
                        ans = rag.invoke({"input": prompt})["answer"]
                        if "</think>" in ans: ans = ans.split("</think>")[-1].strip()
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
            else:
                st.warning("Pas de documents trouvés.")