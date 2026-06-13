@echo off
chcp 65001 >nul
setlocal
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo === FINALNI GRAF PREDIKCIJA KROZ VRIJEME ===
echo Koristi finalni frame_predictions_test.csv iz results_sm04_plus_TRAINVAL_LONG16_OWNVAL.
echo.

"%PY%" -c "import pandas, matplotlib; print('Python OK')"
if errorlevel 1 goto err

if not exist "results_sm04_plus_TRAINVAL_LONG16_OWNVAL\frame_predictions_test.csv" (
  echo Ne postoji results_sm04_plus_TRAINVAL_LONG16_OWNVAL\frame_predictions_test.csv
  echo Provjeri da si pokrenula 11B i da je evaluacija spremila predikcije.
  goto err
)

"%PY%" scripts\13_plot_final_timeline.py --csv results_sm04_plus_TRAINVAL_LONG16_OWNVAL\frame_predictions_test.csv --out_png results_sm04_plus_TRAINVAL_LONG16_OWNVAL\final_predicted_phonemes_timeline.png --column pred --smooth_window 5
if errorlevel 1 goto err

"%PY%" scripts\13_plot_final_timeline.py --csv results_sm04_plus_TRAINVAL_LONG16_OWNVAL\frame_predictions_test.csv --out_png results_sm04_plus_TRAINVAL_LONG16_OWNVAL\final_true_vs_pred_timeline.png --column both --smooth_window 5
if errorlevel 1 goto err

echo.
echo GOTOVO. Napravljene su slike:
echo   results_sm04_plus_TRAINVAL_LONG16_OWNVAL\final_predicted_phonemes_timeline.png
echo   results_sm04_plus_TRAINVAL_LONG16_OWNVAL\final_true_vs_pred_timeline.png
echo.
echo Za prezentaciju koristi prvu ako zelis prikaz izlaza modela,
echo a drugu samo ako zelis pokazati usporedbu stvarnih i predvidenih fonema.
pause
exit /b 0

:err
echo.
echo DOSLO JE DO GRESKE. Posalji zadnjih 15 redova ispisa.
pause
exit /b 1
