@echo off
chcp 65001 >nul
setlocal
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo === SM04 LONG16 SVOJA VALIDACIJA ===
echo Ovo je verzija koja NE koristi best_params iz 9 ili 10.
echo Trenirat ce 16-komponentni train-only model, naci svoje prior/smoothing parametre na VAL,
echo zatim trenirati finalni TRAIN+VAL model i evaluirati TEST.
echo Usporedi s trenutnim najboljim: 0.624123.
echo.

"%PY%" -c "import pandas, numpy, sklearn, librosa, soundfile, matplotlib, tqdm, joblib; print('Python OK')"
if errorlevel 1 goto err

if not exist "data\features_sm04_plus\features_index.csv" (
  echo Plus znacajke ne postoje, prvo ih radim...
  "%PY%" scripts\10_extract_features_plus.py --manifest_csv data/manifests/manifest_sm04.csv --out_dir data/features_sm04_plus --index_csv data/features_sm04_plus/features_index.csv --frame_length_ms 20 --hop_length_ms 8 --n_mfcc 20 --extra_spectral --delta_all
  if errorlevel 1 goto err
)

if not exist "data\features_sm04_plus\features_index_trainval.csv" (
  echo Radim train+val index...
  "%PY%" scripts\10_make_trainval_index.py --features_index data/features_sm04_plus/features_index.csv --out_csv data/features_sm04_plus/features_index_trainval.csv
  if errorlevel 1 goto err
)

echo === 1/4 Train-only LONG16 model za vlastitu validaciju ===
"%PY%" scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index.csv --out_model models/gmm_sm04_plus_trainonly_LONG16_OWNVAL.joblib --train_split train --n_components 16 --max_frames_per_phone 70000 --min_frames_per_phone 50 --max_iter 350 --n_init 4 --reg_covar 0.0001
if errorlevel 1 goto err

echo === 2/4 WIDER validation grid za LONG16 ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index.csv --model_path models/gmm_sm04_plus_trainonly_LONG16_OWNVAL.joblib --split val --out_dir results_sm04_plus_LONG16_OWN_validation_tracking --grid_search --prior_grid 0,0.25,0.5,0.75,1,1.25,1.5,2 --smooth_grid 1,3,5,7,9,11,13,15
if errorlevel 1 goto err

echo === 3/4 Finalni TRAIN+VAL LONG16 trening ===
"%PY%" scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index_trainval.csv --out_model models/gmm_sm04_plus_TRAINVAL_LONG16_OWNVAL.joblib --train_split train --n_components 16 --max_frames_per_phone 120000 --min_frames_per_phone 50 --max_iter 400 --n_init 4 --reg_covar 0.0001
if errorlevel 1 goto err

echo === 4/4 TEST evaluacija s vlastitim LONG16 best_params ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index_trainval.csv --model_path models/gmm_sm04_plus_TRAINVAL_LONG16_OWNVAL.joblib --split test --out_dir results_sm04_plus_TRAINVAL_LONG16_OWNVAL --best_params_json results_sm04_plus_LONG16_OWN_validation_tracking\best_params.json --save_predictions
if errorlevel 1 goto err

echo.
echo GOTOVO. Pogledaj:
echo   results_sm04_plus_LONG16_OWN_validation_tracking\best_params.json
echo   results_sm04_plus_TRAINVAL_LONG16_OWNVAL\classification_report_test.txt
echo.
echo Ako je accuracy veci od 0.624123, koristi ovaj rezultat. Ako nije, ostani na 0.624123.
pause
exit /b 0

:err
echo.
echo DOSLO JE DO GRESKE. Kopiraj zadnjih 20 redova i posalji ChatGPT-u.
pause
exit /b 1
