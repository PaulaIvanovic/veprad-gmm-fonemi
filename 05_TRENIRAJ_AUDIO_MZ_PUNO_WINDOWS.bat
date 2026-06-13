@echo off
chcp 65001 >nul
call .venv\Scripts\activate
python scripts\run_all_audio_weak.py --n_components 8 --max_frames_per_phone 30000
python scripts\check_outputs.py
pause
