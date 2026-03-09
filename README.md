# COS_NIA_02
R2D2
Prérequis
Avant de commencer, assurez-vous d'avoir installé :
Python 3.8
Ollama :
    linux:
    curl -fsSL https://ollama.com/install.sh | sh

    windows:
    irm https://ollama.com/install.ps1 | iex

    mac :
    curl -fsSL https://ollama.com/install.sh | sh

1. Cloner le projet et se préparer
Ouvrez un terminal et placez-vous dans le répertoire du projet :

2. Installer les dépendances Python
pip install -r requirements.txt

3. installer le modèle
ollama pull deepseek-r1:8b
d'autres modèles peuvent être utilisés en adaptant le code. Ce modèle nécessite environ 6Gb de RAM et tourne "assez" bien sur ordinateur. Le code n'est pas forcément optimisé pour marcher avec ce modèle particulier

Lancement de l'appli
Pour faire fonctionner l'assistant, vous devez ouvrir deux terminaux :

Terminal 1 : Serveur d'IA
Lancez le moteur Ollama pour qu'il puisse répondre aux requêtes :
ollama serve

Terminal 2 : Interface Utilisateur
Lancez l'interface Streamlit avec la configuration réseau appropriée :

streamlit run main.py --server.headless true --server.address 0.0.0.0

Une fois lancé, l'assistant sera accessible dans votre navigateur à l'adresse : http://localhost:8501 (ou l'IP de votre machine).


📁 Structure du projet
main.py : Interface graphique streamlit, et logique du main chat
requirements.txt : Liste des dépendances Python
README.md : Ce guide d'utilisation.
app.bat : tentative d'executable pour lancer directement l'appli sous windows
start.sh : même chose sous linux
utils.py : quelques trucs utiles comme les path des différents directory

Modules :
    auth.py : gestion des comptes utilisateurs et de l'authentification
    files.py : gestion de la base de données et recherche intelligente
    generators.py : génération de contenu autre que le texte
    rag.py : Le rag en lui même
    schedule.py : module de gestion emploi du temps

data :
    cours :
        liste base de donnée des cours sur lesquels se base le rag
    users :
        liste des comptes et infos des utilisateurs

generated :
    directory d'arrivée des contenus générés

💡 Notes importantes
Headless mode : L'option --server.headless true est activée pour éviter l'ouverture automatique du navigateur (utile pour les serveurs distants ou Docker).

Accès réseau : L'adresse 0.0.0.0 permet au serveur d'écouter sur toutes les interfaces réseau de la machine