"""
=========================================================
EXTRACCIÓN DE FEATURES — BASE DE FRAIWAN
=========================================================
Replica el mismo pipeline que ICBHI:
  - Frecuencia: 4 kHz
  - Ventana:    5 s (no solapadas)
  - Umbral de silencio: 1e-5 (igual que ICBHI corregido)
  - 41 features idénticas

El diagnóstico se extrae del nombre del archivo:
  BP1_Asthma,I E W,P L L,70,M.wav
  └── BP1: ID paciente
  └── Asthma: diagnóstico (entre "_" y la primera ",")

Las 6 clases consideradas:
  Healthy, Asthma, BRON, COPD, Pneumonia, Heart Failure
=========================================================
"""

import os
import numpy as np
import librosa
import pandas as pd
from scipy.stats import skew, kurtosis

# =========================================================
# RUTAS
# =========================================================

audio_dir   = r"C:\Users\josem\Desktop\tfg\AudioFiles"
output_path = r"C:\Users\josem\Desktop\tfg\features_fraiwan.csv"

# =========================================================
# PARÁMETROS (idénticos a ICBHI)
# =========================================================

TARGET_SR        = 4000
WINDOW_SECONDS   = 5
window_size      = TARGET_SR * WINDOW_SECONDS
N_MFCC           = 13
SILENCE_THRESHOLD = 1e-5

# =========================================================
# MAPEO DE DIAGNÓSTICOS
# =========================================================
# Las 6 clases consideradas en Fraiwan.
# Asegura que las etiquetas son IDÉNTICAS a las de ICBHI
# (en mayúsculas las dejará el notebook de modelos).
# =========================================================

diagnosis_map = {
    "asthma"          : "Asthma",
    "bron"            : "BRON",
    "bronchiectasis"  : "BRON",
    "bronchitis"      : "BRON",
    "copd"            : "COPD",
    "heart failure"   : "Heart Failure",
    "heartfailure"    : "Heart Failure",
    "n"               : "Healthy",
    "normal"          : "Healthy",
    "healthy"         : "Healthy",
    "pneumonia"       : "Pneumonia",
    "plueral effusion": "Pneumonia",
    "pleural effusion": "Pneumonia",
    "lung fibrosis"   : "Pneumonia",
}

# =========================================================
# PARSER DEL NOMBRE DE ARCHIVO
# =========================================================

def parse_filename(filename):
    """Extrae el id de paciente y la etiqueta cruda de diagnóstico."""
    nombre = os.path.splitext(filename)[0]
    if "_" not in nombre:
        return None, None
    patient_id, resto = nombre.split("_", 1)
    if "," in resto:
        diag_raw = resto.split(",", 1)[0]
    else:
        diag_raw = resto
    return patient_id, diag_raw.strip()

def normalizar_diagnostico(diag_raw):
    if diag_raw is None:
        return None
    return diagnosis_map.get(diag_raw.lower().strip(), None)

# =========================================================
# FEATURES DE ENTROPÍA
# =========================================================

def shannon_entropy(signal):
    n      = len(signal)
    sigma  = np.std(signal)
    bw     = 3.5 * sigma / (n ** (1/3))
    bins   = max(int((np.max(signal) - np.min(signal)) / (bw + 1e-12)), 10)
    hist, _= np.histogram(signal, bins=bins, density=True)
    p      = (hist + 1e-12) / np.sum(hist + 1e-12)
    return -np.sum(p * np.log2(p))

def log_energy_entropy(signal):
    hist, _= np.histogram(signal, bins=64, density=True)
    p      = (hist + 1e-12) / np.sum(hist + 1e-12)
    return -np.sum((np.log(p)) ** 2)

def spectral_entropy(signal):
    S   = np.abs(librosa.stft(signal, n_fft=2048)) ** 2
    psd = np.sum(S, axis=1)
    psd = psd / (np.sum(psd) + 1e-12)
    return -np.sum(psd * np.log2(psd + 1e-12))

# =========================================================
# EXTRACCIÓN DE FEATURES
# =========================================================

def extraer_features(window, sr):
    feats = {}

    feats['shannon_entropy']    = shannon_entropy(window)
    feats['log_energy_entropy'] = log_energy_entropy(window)
    feats['spectral_entropy']   = spectral_entropy(window)

    mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        feats[f'mfcc_{i+1}_std']  = np.std(mfccs[i])

    zcr = librosa.feature.zero_crossing_rate(window)
    feats['zcr_mean'] = np.mean(zcr)
    feats['zcr_std']  = np.std(zcr)

    rms = librosa.feature.rms(y=window)
    feats['rms_mean'] = np.mean(rms)
    feats['rms_std']  = np.std(rms)

    sc = librosa.feature.spectral_centroid(y=window, sr=sr)
    feats['spectral_centroid_mean'] = np.mean(sc)
    feats['spectral_centroid_std']  = np.std(sc)

    sb = librosa.feature.spectral_bandwidth(y=window, sr=sr)
    feats['spectral_bandwidth_mean'] = np.mean(sb)
    feats['spectral_bandwidth_std']  = np.std(sb)

    sr_feat = librosa.feature.spectral_rolloff(y=window, sr=sr)
    feats['spectral_rolloff_mean'] = np.mean(sr_feat)
    feats['spectral_rolloff_std']  = np.std(sr_feat)

    feats['skewness'] = skew(window)
    feats['kurtosis'] = kurtosis(window)

    return feats

# =========================================================
# PROCESAMIENTO
# =========================================================

resultados   = []
desconocidos = []
errores      = []
descartados_silencio = []

print(f"Carpeta de audios: {audio_dir}")
print(f"Umbral de silencio: {SILENCE_THRESHOLD}")
print("-" * 60)

for file in sorted(os.listdir(audio_dir)):

    if not file.lower().endswith(".wav"):
        continue

    file_path = os.path.join(audio_dir, file)

    patient_id, diag_raw = parse_filename(file)
    diagnostico          = normalizar_diagnostico(diag_raw)

    if diagnostico is None:
        desconocidos.append((file, diag_raw))
        continue

    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)

        window_count = 0

        for start in range(0, len(y), window_size):
            end    = start + window_size
            window = y[start:end]

            if len(window) < window_size:
                continue
            if np.mean(window ** 2) < SILENCE_THRESHOLD:
                continue

            feats = extraer_features(window, sr)
            feats['archivo']     = file
            feats['diagnostico'] = diagnostico
            feats['ventana']     = window_count

            resultados.append(feats)
            window_count += 1

        if window_count == 0:
            descartados_silencio.append(file)
        print(f"  {file} -> {diagnostico} ({window_count} ventanas)")

    except Exception as e:
        errores.append((file, str(e)))
        print(f"  Error en {file}: {e}")

# =========================================================
# RESUMEN Y AVISOS
# =========================================================

print()
print("=" * 60)
print("RESUMEN")
print("=" * 60)

if desconocidos:
    print(f"\nDiagnósticos NO reconocidos: {len(desconocidos)}")
    for f, d in desconocidos[:20]:
        print(f"   {f}  ->  '{d}'")
    if len(desconocidos) > 20:
        print(f"   ... y {len(desconocidos) - 20} más")
    print("   >>> Añade estas variantes a `diagnosis_map` y relanza.")

if descartados_silencio:
    print(f"\nArchivos con todas las ventanas silenciosas: {len(descartados_silencio)}")

if errores:
    print(f"\nArchivos con error: {len(errores)}")
    for f, e in errores:
        print(f"   {f}: {e}")

# =========================================================
# GUARDAR CSV
# =========================================================

if len(resultados) == 0:
    print("\nERROR: no se ha generado ninguna fila. Revisa rutas y mapeo.")
else:
    df = pd.DataFrame(resultados)

    meta_cols    = ['archivo', 'diagnostico', 'ventana']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]

    df.to_csv(output_path, index=False, sep=';', decimal=',')

    print(f"\n{'=' * 60}")
    print(f"CSV guardado en: {output_path}")
    print(f"Features por ventana:  {len(feature_cols)}")
    print(f"Total ventanas:        {len(df)}")
    print(f"Archivos únicos:       {df['archivo'].nunique()}")
    print(f"\nDistribución por ventanas:\n{df['diagnostico'].value_counts()}")
    print(f"\nDistribución por archivos únicos:")
    print(df.groupby('diagnostico')['archivo'].nunique())
