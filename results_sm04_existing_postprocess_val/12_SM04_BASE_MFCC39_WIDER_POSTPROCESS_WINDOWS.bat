@echo off
chcp 65001 >nul
setlocal
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo === SM04 BASE MFCC39 WIDER POSTPROCESS GRID ===
echo Ovo NE trenira novi model. Samo proba siri prior/smoothing grid na najboljem osnovnom MFCC39 modelu.
echo Cilj: vidjeti moze li se rezultat 0.624123 jos malo popraviti bez novog treninga.
echo.

"%PY%" -c "import pandas, numpy, sklearn, matplotlib, tqdm, joblib; print('Python OK')"
if errorlevel 1 goto err

echo === 1/2 Wider validation grid na osnovnom modelu ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04/features_index.csv --model_path models/gmm_mfcc39_sm04.joblib --split val --out_dir results_sm04_existing_postprocess_WIDE_val --grid_search --prior_grid 0,0.25,0.5,0.75,1,1.25,1.5,2 --smooth_grid 1,3,5,7,9,11,13,15,17
if errorlevel 1 goto err

echo === 2/2 TEST evaluacija s WIDE best_params ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04/features_index.csv --model_path models/gmm_mfcc39_sm04.joblib --split test --out_dir results_sm04_existing_postprocess_WIDE_test --best_params_json results_sm04_existing_postprocess_WIDE_val\best_params.json --save_predictions
if errorlevel 1 goto err

echo.
echo GOTOVO. Pogledaj:
echo   results_sm04_existing_postprocess_WIDE_val\best_params.json
echo   results_sm04_existing_postprocess_WIDE_test\classification_report_test.txt
echo.
echo Ako je accuracy veci od 0.624123, koristi ovaj rezultat. Ako nije, ostani na 0.624123.
pause
exit /b 0

:err
echo.
echo DOSLO JE DO GRESKE. Kopiraj zadnjih 20 redova i posalji ChatGPT-u.
pause
exit /b 1
