@echo off
chcp 65001 >nul
setlocal
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python

echo === DUGI/AGRESIVNI SM04 TRENING ===
echo Ovo pokreci samo ako imas vremena. Moze trajati 1-4 sata.
echo Veci GMM ne mora uvijek biti bolji, ali vrijedi probati.
echo.

if not exist data\features_sm04_plus\features_index.csv (
  echo Plus znacajke ne postoje, prvo ih radim...
  %PY% scripts\10_extract_features_plus.py --manifest_csv data/manifests/manifest_sm04.csv --out_dir data/features_sm04_plus --index_csv data/features_sm04_plus/features_index.csv --frame_length_ms 20 --hop_length_ms 8 --n_mfcc 20 --extra_spectral --delta_all
  if errorlevel 1 goto err
)

if not exist data\features_sm04_plus\features_index_trainval.csv (
  %PY% scripts\10_make_trainval_index.py --features_index data/features_sm04_plus/features_index.csv --out_csv data/features_sm04_plus/features_index_trainval.csv
  if errorlevel 1 goto err
)

if not exist results_sm04_plus_validation_tracking\best_params.json (
  %PY% scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index.csv --out_model models/gmm_sm04_plus_trainonly.joblib --train_split train --n_components 16 --max_frames_per_phone 60000 --min_frames_per_phone 50 --max_iter 300 --n_init 3 --reg_covar 0.0001
  if errorlevel 1 goto err
  %PY% scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index.csv --model_path models/gmm_sm04_plus_trainonly.joblib --split val --out_dir results_sm04_plus_validation_tracking --grid_search --prior_grid 0,0.25,0.5,0.75,1.0 --smooth_grid 1,3,5,7,9,11
  if errorlevel 1 goto err
)

echo === Finalni dugi trening: 16 komponenti, 120000 frameova po fonemu ===
%PY% scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index_trainval.csv --out_model models/gmm_sm04_plus_TRAINVAL_LONG16.joblib --train_split train --n_components 16 --max_frames_per_phone 120000 --min_frames_per_phone 50 --max_iter 400 --n_init 4 --reg_covar 0.0001
if errorlevel 1 goto err

%PY% scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index_trainval.csv --model_path models/gmm_sm04_plus_TRAINVAL_LONG16.joblib --split test --out_dir results_sm04_plus_TRAINVAL_LONG16 --best_params_json results_sm04_plus_validation_tracking\best_params.json --save_predictions
if errorlevel 1 goto err

echo.
echo GOTOVO. Pogledaj results_sm04_plus_TRAINVAL_LONG16\classification_report_test.txt
pause
exit /b 0

:err
echo.
echo DOSLO JE DO GRESKE. Kopiraj zadnjih 20 redova i posalji ChatGPT-u.
pause
exit /b 1
