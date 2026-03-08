import json
import os
import random
from itertools import cycle
import glob
from datetime import datetime

from streamlit_calendar import calendar
from fpdf import FPDF
import tempfile

import utils


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
        

def update_user_schedule(username_file, schedule_data):
    filepath = os.path.join(utils.USERS_FOLDER, f"{username_file}.json")
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