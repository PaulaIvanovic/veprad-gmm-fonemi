@echo off
chcp 65001 >nul
call .venv\Scripts\activate
echo === 1/5 manifest za brzi test ===
python scripts\01_build_manifest.py --wav_dir data/raw/wav_sm04_demo --label_dir data/labels/lab_sm04_demo --transcript_dir data/transcripts/txt_sm04_demo --out_csv data/manifests/manifest_quick.csv --split_by utterance --speaker_regex "^(sm\d{2})" --test_size 0.25 --val_size 0 --only_with_labels --max_files 16
if errorlevel 1 goto error
echo === 2/5 MFCC znacajke ===
python scripts\03_extract_features.py --manifest_csv data/manifests/manifest_quick.csv --out_dir data/features_quick --index_csv data/features_quick/features_index.csv --frame_length_ms 20 --hop_length_ms 8 --n_mfcc 13
if errorlevel 1 goto error
echo === 3/5 GMM trening - brzi test ===
python scripts\04_train_gmm.py --features_index data/features_quick/features_index.csv --out_model models/gmm_mfcc39_QUICK_TEST.joblib --n_components 1 --max_frames_per_phone 300 --min_frames_per_phone 5 --max_iter 10 --n_init 1
if errorlevel 1 goto error
echo === 4/5 Evaluacija ===
python scripts\05_evaluate.py --features_index data/features_quick/features_index.csv --model_path models/gmm_mfcc39_QUICK_TEST.joblib --split test --out_dir results_QUICK_TEST --save_predictions
if errorlevel 1 goto error
echo === 5/5 Klasifikacija jednog audio primjera ===
python scripts\06_classify_audio.py --audio_path data/raw/wav_sm04_demo/sm04010103201.wav --model_path models/gmm_mfcc39_QUICK_TEST.joblib --out_csv results_QUICK_TEST/classified_segments_sample.csv --out_png results_QUICK_TEST/classified_segments_sample.png --frame_length_ms 20 --hop_length_ms 8
python scripts\check_outputs.py
echo.
echo BRZI TEST GOTOV. Ako vidis OK kod rezultata, projekt radi.
pause
exit /b 0
:error
echo.
echo DOSLO JE DO GRESKE. Slikaj ovaj prozor i posalji mi screenshot.
pause
exit /b 1
