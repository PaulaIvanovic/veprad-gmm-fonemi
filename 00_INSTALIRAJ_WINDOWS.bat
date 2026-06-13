@echo off
chcp 65001 >nul
echo === Instalacija Python okoline ===
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Gotovo. Sada pokreni 01_BRZI_TEST_WINDOWS.bat
pause
