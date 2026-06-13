@echo off
chcp 65001 >nul
call .venv\Scripts\activate
python scripts\00_prepare_archives.py
python scripts\00_validate_dataset.py
pause
