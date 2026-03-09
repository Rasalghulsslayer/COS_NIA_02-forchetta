@echo off
title Assistant IA - Reparation et Lancement
setlocal enabledelayedexpansion

:: --- FIX 1 : GESTION DES CHEMINS UNC/RESEAU ---
:: pushd est plus puissant que cd, il cree un lecteur virtuel si besoin
pushd "%~dp0"
echo [INFO] Repertoire de travail reel : %CD%

:: --- FIX 2 : VERIFICATION DES FICHIERS ---
if not exist "requirements.txt" (
    echo [ERREUR] Fichier requirements.txt introuvable dans %CD%
    echo Verifiez que le .bat est bien a cote de vos fichiers .py
    pause
    exit
)

:: --- FIX 3 : FORCE L'INSTALLATION (Meme si PATH est casse) ---
echo [1/4] Installation des dependances...
:: On utilise "python -m pip" car "pip" tout seul ne marche pas chez toi
python -m pip install --user -r requirements.txt --no-warn-script-location

:: --- FIX 4 : OLLAMA (Test Windows puis Test Bash) ---
echo [2/4] Tentative de lancement de Ollama...
:: On teste si ollama.exe est dans le dossier standard de Windows
if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
    start /min "" "%LocalAppData%\Programs\Ollama\ollama.exe" serve
) else (
    :: Si pas trouve, on tente via ton bash qui semble fonctionner
    start /min cmd /k "bash -c 'ollama serve'"
)
timeout /t 5 /nobreak > nul

:: --- FIX 5 : LANCEMENT STREAMLIT ---
echo [3/4] Ouverture du navigateur...
start "" "http://localhost:8501"

echo [4/4] Demarrage de l'interface...
:: Utilisation de "python -m streamlit" pour contourner le probleme de PATH
python -m streamlit run main.py --server.headless true --server.address 0.0.0.0

if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] Streamlit n'a pas pu demarrer.
    echo Essayez de taper manuellement : python -m pip install streamlit
    pause
)

:: On libere le chemin a la fin
popd
pause