import os
import json
import hashlib
import re
from utils import USERS_FOLDER

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_filepath(username):
    safe_name = re.sub(r'[^a-z0-9]', '', username.lower())
    return os.path.join(USERS_FOLDER, f"{safe_name}.json")

def create_user(username, password, level, tone, role, goal):
    filepath = get_user_filepath(username)
    if os.path.exists(filepath): return False, "Utilisateur existant."
    
    data = {
        "auth": {"username_display": username, "password_hash": hash_password(password)},
        "profil": {
            "niveau": level,
            "role": role,
            "objectif": goal,
            "preferences_apprentissage": {"ton": tone}
        }
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True, "Succès"
    except: return False, "Erreur écriture"

def update_user_profile(username, new_level, new_tone, new_role, new_goal):
    filepath = get_user_filepath(username)
    if not os.path.exists(filepath): return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data["profil"]["niveau"] = new_level
        data["profil"]["preferences_apprentissage"]["ton"] = new_tone
        data["profil"]["role"] = new_role
        data["profil"]["objectif"] = new_goal
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4) 
        return data
    except Exception as e:
        print(f"Erreur update: {e}")
        return None

def verify_credentials(username, password):
    filepath = get_user_filepath(username)
    if not os.path.exists(filepath): return None, "Inconnu"
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        if data["auth"]["password_hash"] == hash_password(password): return data, "Succès"
        return None, "Mot de passe faux"
    except: return None, "Erreur fichier"