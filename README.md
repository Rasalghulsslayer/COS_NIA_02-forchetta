# 🚀 COS_NIA_02 - Assistant IA R2D2

## 📋 Prérequis

Avant de commencer, assurez-vous d'avoir installé **Python 3.8** et **Ollama**.

- **Mac / Linux :** `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows :** `irm https://ollama.com/install.ps1 | iex`

## 🛠️ Installation

1. **Cloner le projet :** Ouvrez un terminal et placez-vous dans le répertoire du projet.
2. **Installer les dépendances Python :**
   ```bash
   pip install -r requirements.txt
   ```

````

3. **Installer le modèle IA :**

```bash
ollama pull deepseek-r1:8b

```

> _Note : Ce modèle nécessite environ 6 Go de RAM. D'autres modèles peuvent être utilisés en adaptant le code._

## 🚀 Lancement de l'application

L'assistant nécessite deux terminaux fonctionnant en parallèle :

**Terminal 1 (Serveur d'IA) :**

```bash
ollama serve

```

**Terminal 2 (Interface Utilisateur) :**

```bash
streamlit run main.py --server.headless true --server.address 0.0.0.0

```

L'interface sera accessible dans votre navigateur à l'adresse : [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501) (ou via l'IP de votre machine).

## 📁 Structure du projet

**Fichiers principaux :**

- `main.py` : Interface graphique Streamlit et logique du chat.
- `requirements.txt` : Liste des dépendances Python.
- `utils.py` : Fonctions utilitaires (chemins de répertoires, etc.).
- `app.bat` / `start.sh` : Scripts de lancement direct (Windows / Linux).

**Modules :**

- `auth.py` : Gestion des comptes et de l'authentification.
- `files.py` : Gestion de la base de données et recherche intelligente.
- `generators.py` : Génération de contenu alternatif (hors texte).
- `rag.py` : Moteur RAG (Retrieval-Augmented Generation).
- `schedule.py` : Module de gestion de l'emploi du temps.

**Dossiers de données :**

- `data/cours/` : Base de données des cours sur lesquels se base le RAG.
- `data/users/` : Informations et paramètres des comptes utilisateurs.
- `generated/` : Répertoire de destination des contenus générés.

## 💡 Notes importantes

- **Mode Headless :** L'option `--server.headless true` évite l'ouverture automatique du navigateur (utile pour les serveurs distants ou Docker).
- **Accès réseau :** L'adresse `0.0.0.0` permet au serveur d'écouter sur toutes les interfaces réseau de la machine.

```

```
````
