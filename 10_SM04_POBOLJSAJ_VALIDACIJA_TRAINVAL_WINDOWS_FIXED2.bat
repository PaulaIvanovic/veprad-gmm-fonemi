@echo off
setlocal EnableExtensions
set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto havepy
set "PY=python"
:havepy

echo === SM04 IMPROVE FIXED2: extended features + validation + trainval final ===
echo Using Python: %PY%
echo.

"%PY%" -c "import pandas, numpy, sklearn, librosa, soundfile, matplotlib, tqdm, joblib; print('Python OK')"
if errorlevel 1 goto packages_err

if exist data\manifests\manifest_sm04.csv goto manifest_ok
echo ERROR: data\manifests\manifest_sm04.csv ne postoji.
echo Prvo pokreni: 03_TRENIRAJ_SM04_NAJBOLJE_WINDOWS.bat
pause
exit /b 1
:manifest_ok

if exist scripts\10_extract_features_plus.py goto scripts_ok
echo ERROR: scripts\10_extract_features_plus.py ne postoji.
echo Nisi kopirala cijeli accuracy patch u projekt.
pause
exit /b 1
:scripts_ok

echo === 1/6 Extended features: 20 MFCC + delta + delta-delta + spectral ===
"%PY%" scripts\10_extract_features_plus.py --manifest_csv data/manifests/manifest_sm04.csv --out_dir data/features_sm04_plus --index_csv data/features_sm04_plus/features_index.csv --frame_length_ms 20 --hop_length_ms 8 --n_mfcc 20 --extra_spectral --delta_all
if errorlevel 1 goto err

echo === 2/6 Train model on TRAIN only ===
"%PY%" scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index.csv --out_model models/gmm_sm04_plus_trainonly.joblib --train_split train --n_components 12 --max_frames_per_phone 50000 --min_frames_per_phone 50 --max_iter 250 --n_init 3 --reg_covar 0.0001
if errorlevel 1 goto err

echo === 3/6 Validation accuracy tracking / grid search ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index.csv --model_path models/gmm_sm04_plus_trainonly.joblib --split val --out_dir results_sm04_plus_validation_tracking --grid_search --prior_grid 0,0.25,0.5,0.75,1.0 --smooth_grid 1,3,5,7,9,11
if errorlevel 1 goto err

echo === 4/6 Create TRAIN+VAL feature index ===
"%PY%" scripts\10_make_trainval_index.py --features_index data/features_sm04_plus/features_index.csv --out_csv data/features_sm04_plus/features_index_trainval.csv
if errorlevel 1 goto err

echo === 5/6 Final training on TRAIN+VAL ===
"%PY%" scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index_trainval.csv --out_model models/gmm_sm04_plus_TRAINVAL_FINAL.joblib --train_split train --n_components 12 --max_frames_per_phone 80000 --min_frames_per_phone 50 --max_iter 300 --n_init 3 --reg_covar 0.0001
if errorlevel 1 goto err

echo === 6/6 Final TEST evaluation ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index_trainval.csv --model_path models/gmm_sm04_plus_TRAINVAL_FINAL.joblib --split test --out_dir results_sm04_plus_TRAINVAL_FINAL --best_params_json results_sm04_plus_validation_tracking\best_params.json --save_predictions
if errorlevel 1 goto err

echo.
echo GOTOVO.
echo Pogledaj:
echo   results_sm04_plus_validation_tracking\validation_grid.csv
echo   results_sm04_plus_TRAINVAL_FINAL\classification_report_test.txt
echo   results_sm04_plus_TRAINVAL_FINAL\confusion_matrix_test.png
echo.
echo Usporedi s trenutnim najboljim rezultatom: 0.624123
pause
exit /b 0

:packages_err
echo.
echo Nedostaju Python paketi ili se ne koristi .venv.
echo Pokreni: 00_INSTALIRAJ_WINDOWS.bat
pause
exit /b 1

:err
echo.
echo DOSLO JE DO GRESKE. Kopiraj zadnjih 20 redova i posalji ChatGPT-u.
pause
exit /b 1
