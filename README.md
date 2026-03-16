# 🚀 COS_NIA_02 - AI Assistant R2D2

## 📋 Prerequisites

Before starting, make sure you have **Python 3.8** and **Ollama** installed.

- **Mac / Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows:** `irm https://ollama.com/install.ps1 | iex`

## 🛠️ Installation

1. **Clone the project:** Open a terminal and navigate to the project directory.
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install the AI model:**

```bash
ollama pull deepseek-r1:8b


```

> _Note: This model requires about 6GB of RAM. Other models can be used by adapting the code._

## 🚀 Launching the application

The assistant requires two terminals running in parallel:

**Terminal 1 (AI Server):**

```bash
ollama serve


```

**Terminal 2 (User Interface):**

```bash
streamlit run main.py --server.headless true --server.address 0.0.0.0


```

The interface will be accessible in your browser at: [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501) (or via your machine's IP).

## 📁 Project structure

**Main files:**

- `main.py` : Streamlit graphical interface and chat logic.
- `requirements.txt` : List of Python dependencies.
- `utils.py` : Utility functions (directory paths, etc.).
- `app.bat` / `start.sh` : Direct launch scripts (Windows / Linux).

**Modules:**

- `auth.py` : Account and authentication management.
- `files.py` : Database management and intelligent search.
- `generators.py` : Alternative content generation (non-text).
- `rag.py` : RAG (Retrieval-Augmented Generation) engine.
- `schedule.py` : Schedule management module.

**Data folders:**

- `data/cours/` : Course database used by the RAG.
- `data/users/` : User accounts information and settings.
- `generated/` : Destination directory for generated content.

## 💡 Important notes

- **Headless mode:** The `--server.headless true` option prevents the browser from opening automatically (useful for remote servers or Docker).
- **Network access:** The `0.0.0.0` address allows the server to listen on all of the machine's network interfaces.
