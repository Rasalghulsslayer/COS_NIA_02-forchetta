#!/bin/bash

# 1. Lancer Ollama si besoin
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# 2. Ouvrir le navigateur Windows depuis WSL
# On le fait avant ou juste après le lancement
explorer.exe "http://localhost:8501"

# 3. Lancer Streamlit (sans le mode headless)
streamlit run main.py --server.address 0.0.0.0