@echo off
chcp 65001 >nul
call .venv\Scripts\activate
python scripts\run_all_audio_weak.py --max_files 200 --n_components 4 --max_frames_per_phone 10000
python scripts\check_outputs.py
pause
