@echo off
title Lancement de Spaceflight Institute 🚀

echo ==========================================
echo      DEMARRAGE DU SYSTEME IA (WSL)
echo ==========================================

:: Etape 1 : On prépare l'ouverture du navigateur dans 8 secondes
:: (Le temps que Streamlit démarre).
:: Le ">nul" sert à cacher le compte à rebours pour que ce soit propre.
start "" /B cmd /c "timeout /nobreak /t 8 >nul & start http://localhost:8501"

:: Etape 2 : On lance WSL et Streamlit
:: Note : J'ai ajouté --server.address 0.0.0.0 pour forcer l'accessibilité
wsl -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate pie; if ! pgrep -x 'ollama' > /dev/null; then echo 'Démarrage Ollama...'; ollama serve > /dev/null 2>&1 & sleep 5; fi; cd ~/Sup/COS_NIA_02/; echo 'Lancement Streamlit...'; streamlit run main.py --server.port 8501 --server.address 0.0.0.0"

pause