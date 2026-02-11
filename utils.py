import os

# Constantes de chemins
BASE_FOLDER = "data"
COURS_FOLDER = os.path.join(BASE_FOLDER, "cours")
USERS_FOLDER = os.path.join(BASE_FOLDER, "users")

def init_folders():
    """Crée les dossiers nécessaires au démarrage."""
    for folder in [BASE_FOLDER, COURS_FOLDER, USERS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

# Configuration de la page (à appeler au début du main)
def setup_page(st):
    st.set_page_config(page_title="Spaceflight Institute", page_icon="🚀", layout="wide")
    st.title("🤖 Spaceflight Institute - Recherche Intelligente")
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"