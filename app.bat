@echo off
set DISTRO=Ubuntu

echo Lancement du serveur RAG et d'Ollama...

:: Lance Ollama et Streamlit dans WSL en arrière-plan
wsl -d %DISTRO% bash -c "ollama serve > /dev/null 2>&1 & streamlit run main.py --server.address 0.0.0.0" &

:: Attend 5 secondes que le serveur démarre
timeout /t 5

:: Ouvre le navigateur sur Windows
start http://localhost:8501

pause