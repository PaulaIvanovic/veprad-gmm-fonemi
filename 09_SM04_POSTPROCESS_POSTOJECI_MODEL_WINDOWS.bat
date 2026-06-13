@echo off
chcp 65001 >nul
setlocal
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python

echo === POSTPROCESS POSTOJECEG SM04 MODELA ===
echo Ovo NE trenira novi model. Samo bira prior/smoothing na validation skupu i evaluira test.

%PY% scripts\10_eval_gmm_advanced.py --features_index data/features_sm04/features_index.csv --model_path models/gmm_mfcc39_sm04.joblib --split val --out_dir results_sm04_existing_postprocess_val --grid_search --prior_grid 0,0.25,0.5,0.75,1.0 --smooth_grid 1,3,5,7,9,11
if errorlevel 1 goto err

%PY% scripts\10_eval_gmm_advanced.py --features_index data/features_sm04/features_index.csv --model_path models/gmm_mfcc39_sm04.joblib --split test --out_dir results_sm04_existing_postprocess_test --best_params_json results_sm04_existing_postprocess_val\best_params.json --save_predictions
if errorlevel 1 goto err

echo.
echo GOTOVO. Pogledaj results_sm04_existing_postprocess_test\classification_report_test.txt
echo Ako je accuracy veci od 0.538859, koristi taj rezultat.
pause
exit /b 0

:err
echo.
echo DOSLO JE DO GRESKE. Kopiraj zadnjih 20 redova i posalji ChatGPT-u.
pause
exit /b 1
