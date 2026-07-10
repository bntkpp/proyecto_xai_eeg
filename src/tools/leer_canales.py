"""
tools/leer_canales.py
--------------------------------------------------------------
Lee UN archivo .fif (Epochs de MNE) y te imprime:
  - los nombres de los electrodos EN ORDEN
  - la forma real de los datos (confirma las muestras del modelo)
  - la frecuencia de muestreo y si ya trae montaje

Uso:
    python src/tools/leer_canales.py "data/sub-069_ses-5_task-95ByBP_eeg_clean-epo.fif"
--------------------------------------------------------------
"""
import sys
from pathlib import Path

import mne

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta = str(_PROJECT_ROOT / "data" / "sub-069_ses-5_task-95ByBP_eeg_clean-epo.fif")

    epochs = mne.read_epochs(ruta, preload=False, verbose="ERROR")

    print("Archivo:", ruta)
    print("N.º de canales:", len(epochs.ch_names))
    print("¿Trae montaje embebido?:", epochs.get_montage() is not None)
    print("Frecuencia de muestreo (Hz):", epochs.info["sfreq"])

    datos = epochs.get_data()
    print("Forma de los datos (n_epochs, n_canales, n_muestras):", datos.shape)
    print("  -> muestras por época:", datos.shape[-1], "  (el modelo espera ~375)")

    print("\n--- Nombres en orden (lista lista para pegar en CH_NAMES) ---")
    print("CH_NAMES =", epochs.ch_names)


if __name__ == "__main__":
    main()
