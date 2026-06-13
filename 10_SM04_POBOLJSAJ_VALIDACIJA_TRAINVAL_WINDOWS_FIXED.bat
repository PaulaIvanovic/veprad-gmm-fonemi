@echo off
setlocal EnableExtensions
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo === SM04 IMPROVE: more features + validation + trainval final ===
echo Using Python: %PY%
echo.

"%PY%" -c "import pandas, numpy, sklearn, librosa, soundfile, matplotlib, tqdm, joblib; print('Python OK')"
if errorlevel 1 (
  echo.
  echo Packages missing. Installing from requirements.txt...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto err
)

if not exist data\manifests\manifest_sm04.csv (
  echo Manifest missing, building manifest...
  "%PY%" scripts\01_build_manifest.py --wav_dir data/raw/wav_sm04 --label_dir data/labels/lab_sm04 --transcript_dir data/transcripts/txt_sm04 --out_csv data/manifests/manifest_sm04.csv --split_by utterance --speaker_regex ^(sm\d{2}) --test_size 0.2 --val_size 0.1 --only_with_labels
  if errorlevel 1 goto err
)

echo === 1/6 Extracting extended features ===
"%PY%" scripts\10_extract_features_plus.py --manifest_csv data/manifests/manifest_sm04.csv --out_dir data/features_sm04_plus --index_csv data/features_sm04_plus/features_index.csv --frame_length_ms 20 --hop_length_ms 8 --n_mfcc 20 --extra_spectral --delta_all
if errorlevel 1 goto err

echo === 2/6 Train model on TRAIN only ===
"%PY%" scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index.csv --out_model models/gmm_sm04_plus_trainonly.joblib --train_split train --n_components 12 --max_frames_per_phone 50000 --min_frames_per_phone 50 --max_iter 250 --n_init 3 --reg_covar 0.0001
if errorlevel 1 goto err

echo === 3/6 Validation grid search ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index.csv --model_path models/gmm_sm04_plus_trainonly.joblib --split val --out_dir results_sm04_plus_validation_tracking --grid_search --prior_grid 0,0.25,0.5,0.75,1.0 --smooth_grid 1,3,5,7,9,11
if errorlevel 1 goto err

echo === 4/6 Create TRAIN+VAL index ===
"%PY%" scripts\10_make_trainval_index.py --features_index data/features_sm04_plus/features_index.csv --out_csv data/features_sm04_plus/features_index_trainval.csv
if errorlevel 1 goto err

echo === 5/6 Final training on TRAIN+VAL ===
"%PY%" scripts\04_train_gmm.py --features_index data/features_sm04_plus/features_index_trainval.csv --out_model models/gmm_sm04_plus_TRAINVAL_FINAL.joblib --train_split train --n_components 12 --max_frames_per_phone 80000 --min_frames_per_phone 50 --max_iter 300 --n_init 3 --reg_covar 0.0001
if errorlevel 1 goto err

echo === 6/6 Final TEST evaluation ===
"%PY%" scripts\10_eval_gmm_advanced.py --features_index data/features_sm04_plus/features_index_trainval.csv --model_path models/gmm_sm04_plus_TRAINVAL_FINAL.joblib --split test --out_dir results_sm04_plus_TRAINVAL_FINAL --best_params_json results_sm04_plus_validation_tracking\best_params.json --save_predictions
if errorlevel 1 goto err

echo.
echo DONE.
echo Check:
echo   results_sm04_plus_validation_tracking\validation_grid.csv
echo   results_sm04_plus_TRAINVAL_FINAL\classification_report_test.txt
echo   results_sm04_plus_TRAINVAL_FINAL\confusion_matrix_test.png
pause
exit /b 0

:err
echo.
echo ERROR. Send the last 20 lines to ChatGPT.
pause
exit /b 1
