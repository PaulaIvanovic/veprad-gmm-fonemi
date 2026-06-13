@echo off
chcp 65001 >nul
call .venv\Scripts\activate
python scripts\run_sm04_supervised.py
python scripts\check_outputs.py
pause
