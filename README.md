# Klasifikacija kratkotrajnih segmenata govora pomoću MFCC i GMM na VEPRAD bazi

Autorica: Paula Ivanović

Ovaj projekt trenira fonemski klasifikator nad VEPRAD govorom:

1. WAV govor se reže na kratke okvire.
2. Iz svakog okvira računaju se MFCC značajke.
3. Dodaju se delta i delta-delta koeficijenti.
4. Za svaki fonem trenira se poseban Gaussian Mixture Model (GMM).
5. Novi okvir se klasificira kao fonem čiji GMM daje najveću log-vjerojatnost.
6. Model se evaluira accuracyjem, classification reportom i matricom zabune.

## 1. Struktura projekta

```text
veprad_gmm_fonemi_CIJELI_PROJEKT/
  README.md
  KAKO_POKRENUTI_I_OBJASNJENJE.md
  requirements.txt
  config.yaml

  data/
    archives/                         # OVDJE stavljaš RAR arhive
      wav_sm04(1).rar
      lab_sm04(1).rar
      txt_sm04(1).rar
      audio_m(2).rar
      audio_z(2).rar
      text(1).rar

    raw/
      wav_sm04/                       # WAV za SM04 supervised trening
      audio_m/                        # muški govornici m01-m11
      audio_z/                        # ženski govornici z01-z14

    labels/
      lab_sm04/                       # prave fonemske oznake za SM04
      uniform_audio_mz/               # približne oznake iz TXT+DCT

    transcripts/
      txt_sm04/                       # SM04 transkripti
      text/                           # transkripti za audio_m/audio_z

    dict/
      VEPRAD_W.DCT.txt                # fonetski rječnik riječ -> fonemi

    manifests/                        # CSV popisi datoteka i splitova
    features_sm04/                    # MFCC značajke za SM04
    features_audio_mz_weak/           # MFCC značajke za audio_m/audio_z weak mod

  src/veprad_gmm/                     # Python modul projekta
    features.py                       # MFCC + delta + delta-delta
    labels.py                         # čitanje .lab oznaka
    g2p_hr.py                         # DCT/G2P za približne oznake
    model.py                          # GMM modeli
    plotting.py                       # matrica zabune i timeline
    io_utils.py

  scripts/
    00_prepare_archives.py            # raspakirava RAR u prave mape
    00_validate_dataset.py            # provjera datoteka
    01_build_manifest.py              # radi manifest CSV
    02_make_uniform_phone_labels.py   # TXT+DCT -> približne .lab oznake
    03_extract_features.py            # MFCC ekstrakcija
    04_train_gmm.py                   # treniranje GMM modela
    05_evaluate.py                    # evaluacija
    06_classify_audio.py              # klasifikacija jednog WAV-a
    07_classify_folder.py             # klasifikacija cijele mape
    run_sm04_supervised.py            # glavni supervised pipeline
    run_all_audio_weak.py             # audio_m/audio_z približni pipeline

  models/                             # istrenirani modeli
  results_sm04/                       # rezultati glavnog pipelinea
  results_audio_mz_weak/              # rezultati približnog pipelinea
```

## 2. Koje datoteke su potrebne

Za ispravan supervised projekt minimalno trebaju:

```text
wav_sm04(1).rar     # WAV zvuk
lab_sm04(1).rar     # fonemske oznake start-end-fonem
VEPRAD_W.DCT.txt    # fonetski rječnik, uključen u ZIP
```

Preporučeno je imati i:

```text
txt_sm04(1).rar     # tekstualne transkripcije
```

Za dodatni veći, ali približni trening možeš koristiti:

```text
audio_m(2).rar      # muški govornici
audio_z(2).rar      # ženski govornici
text(1).rar         # transkripti za audio_m/audio_z
```

Važno: `audio_m` i `audio_z` su samo WAV audio. Za pravi supervised trening trebale bi i odgovarajuće `.lab` oznake za te govornike. Ako ih nema, projekt može napraviti približne fonemske oznake iz TXT+DCT metodom uniformnog trajanja fonema, ali to nije jednako dobro kao prave fonemske granice.

## 3. Instalacija

Windows:

```bat
cd veprad_gmm_fonemi_CIJELI_PROJEKT
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
cd veprad_gmm_fonemi_CIJELI_PROJEKT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ako automatsko raspakiravanje RAR arhiva ne radi, instaliraj 7-Zip ili ručno raspakiraj arhive u mape navedene u strukturi.

## 4. Raspakiravanje podataka

Stavi arhive u:

```text
data/archives/
```

Zatim pokreni:

```bash
python scripts/00_prepare_archives.py
python scripts/00_validate_dataset.py
```

Ako nemaš 7-Zip/unrar, ručno raspakiraj:

```text
wav_sm04(1).rar  -> data/raw/wav_sm04/
lab_sm04(1).rar  -> data/labels/lab_sm04/
txt_sm04(1).rar  -> data/transcripts/txt_sm04/
audio_m(2).rar   -> data/raw/audio_m/
audio_z(2).rar   -> data/raw/audio_z/
text(1).rar      -> data/transcripts/text/
```

## 5. Glavni način pokretanja: SM04 supervised

Ovo je najispravniji način jer koristi prave `.lab` fonemske oznake.

```bash
python scripts/run_sm04_supervised.py
```

Dobit ćeš:

```text
models/gmm_mfcc39_sm04.joblib
results_sm04/classification_report_test.txt
results_sm04/confusion_matrix_test.png
results_sm04/frame_predictions_test.csv
results_sm04/classified_segments_sample.csv
results_sm04/classified_segments_sample.png
```

## 6. Veći audio_m/audio_z način, ali s približnim oznakama

Ovo koristi `audio_m`, `audio_z`, `text` i `VEPRAD_W.DCT.txt`. Korisno je jer koristi puno više govornika, ali oznake su približne jer nema pravih `.lab` granica za te audio datoteke.

Brzi test:

```bash
python scripts/run_all_audio_weak.py --max_files 200
```

Puni trening:

```bash
python scripts/run_all_audio_weak.py
```

Rezultati:

```text
models/gmm_mfcc39_audio_mz_weak.joblib
results_audio_mz_weak/classification_report_test.txt
results_audio_mz_weak/confusion_matrix_test.png
```

## 7. Ručno pokretanje pipelinea

SM04 supervised ručno:

```bash
python scripts/01_build_manifest.py --wav_dir data/raw/wav_sm04 --label_dir data/labels/lab_sm04 --transcript_dir data/transcripts/txt_sm04 --out_csv data/manifests/manifest_sm04.csv --split_by utterance --speaker_regex "^(sm\d{2})" --only_with_labels

python scripts/03_extract_features.py --manifest_csv data/manifests/manifest_sm04.csv --out_dir data/features_sm04 --index_csv data/features_sm04/features_index.csv --frame_length_ms 20 --hop_length_ms 8 --n_mfcc 13

python scripts/04_train_gmm.py --features_index data/features_sm04/features_index.csv --out_model models/gmm_mfcc39_sm04.joblib --n_components 8 --max_frames_per_phone 30000

python scripts/05_evaluate.py --features_index data/features_sm04/features_index.csv --model_path models/gmm_mfcc39_sm04.joblib --split test --out_dir results_sm04 --save_predictions
```

Klasifikacija novog WAV-a:

```bash
python scripts/06_classify_audio.py --audio_path data/raw/wav_sm04/NEKI_FILE.wav --model_path models/gmm_mfcc39_sm04.joblib --out_csv results_moj_audio.csv --out_png results_moj_audio.png
```

## 8. Koliko traje trening

Procjena za normalan laptop:

- demo iz ZIP-a: 1-3 minute
- SM04 supervised: otprilike 15-45 minuta
- audio_m + audio_z weak: otprilike 45-120 minuta

Vrijeme jako ovisi o procesoru, disku, broju WAV datoteka, `n_components` i `max_frames_per_phone`. Ako je presporo, smanji:

```bash
--n_components 4 --max_frames_per_phone 10000
```

Ako želiš bolji model i imaš vremena, koristi:

```bash
--n_components 8 --max_frames_per_phone 30000
```

## 9. Što predati

U seminarskom/projektu možeš predati:

```text
README.md
KAKO_POKRENUTI_I_OBJASNJENJE.md
src/
scripts/
config.yaml
requirements.txt
results_sm04/classification_report_test.txt
results_sm04/confusion_matrix_test.png
```

Ne moraš predavati cijelu VEPRAD bazu ako je velika; dovoljno je objasniti gdje se ona stavlja i priložiti rezultate.
