"""
leer_canales.py
--------------------------------------------------------------
Lee UN archivo .fif (Epochs de MNE) y te imprime:
  - los nombres de los 63 electrodos EN ORDEN
  - la forma real de los datos (confirma las muestras del modelo)
  - la frecuencia de muestreo y si ya trae montaje

Uso:
    python leer_canales.py "sub-069_ses-5_task-95ByBP_eeg_clean-epo.fif"
"""
import sys
import mne

ruta = sys.argv[1] if len(sys.argv) > 1 else "sub-069_ses-5_task-95ByBP_eeg_clean-epo.fif"

epochs = mne.read_epochs(ruta, preload=False, verbose="ERROR")

print("Archivo:", ruta)
print("N.º de canales:", len(epochs.ch_names))
print("¿Trae montaje embebido?:", epochs.get_montage() is not None)
print("Frecuencia de muestreo (Hz):", epochs.info["sfreq"])

datos = epochs.get_data()          # (n_epochs, n_canales, n_muestras)
print("Forma de los datos (n_epochs, n_canales, n_muestras):", datos.shape)
print("  -> muestras por época:", datos.shape[-1], "  (el modelo espera ~375)")

print("\n--- Nombres en orden (lista lista para pegar en CH_NAMES) ---")
print("CH_NAMES =", epochs.ch_names)
