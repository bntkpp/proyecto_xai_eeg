"""
config.py
====================================================================
Configuración centralizada del proyecto: rutas, constantes y logging.
Es el ÚNICO lugar donde se calculan rutas absolutas — todos los demás
módulos importan desde aquí en vez de recalcular Path(__file__)...
Esto evita el bug real que tuvimos: rutas relativas distintas en cada
archivo (parent.parent vs parent.parent.parent) que se rompen si algo
se mueve de carpeta.
====================================================================
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------
# RUTAS DEL PROYECTO
# --------------------------------------------------------------------
# Este archivo vive en <raíz>/src/config.py
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

WEIGHTS_PATH = ROOT_DIR / "eegnet_fold1.pth"
BODY_MODEL_PATH = DATA_DIR / "xgboost_cuerpo.json"
BODY_DATASET_PATH = DATA_DIR / "dataset_elite_biovid.csv"
CALIBRATOR_PATH = DATA_DIR / "calibrador_isotonico.pkl"
ENV_FILE = ROOT_DIR / ".env"

# --------------------------------------------------------------------
# CONSTANTES DEL MODELO EEG (EEGNet)
# --------------------------------------------------------------------
N_CLASSES = 4
N_CHANNELS = 63
N_SAMPLES = 375          # muestras esperadas (1.5 s @ 250 Hz)
SFREQ = 250               # Hz
IG_STEPS = 50              # pasos de Integrated Gradients

# El modelo NO se entrenó con Voltios crudos: sin este escalado satura y
# predice siempre "Sin Dolor". Recuperado por forense del propio .pth
# (running_var de la primera BatchNorm). Incertidumbre ~±15%.
EEG_SCALE = 8.7e4
RAW_VOLT_MAXABS = 1e-2    # bajo este valor, asumimos que el tensor viene en Voltios crudos

# Temperature scaling: ajustado por NLL en la mitad de los sujetos y
# validado en la otra mitad. No cambia la clase predicha, solo el % mostrado.
# OJO: confianzas > 55% siguen algo sobreestimadas.
CONF_TEMPERATURE = 0.903

MAPA_DOLOR = {
    0: "Sin Dolor (NRS 0-2)",
    1: "Dolor Leve (NRS 4)",
    2: "Dolor Moderado (NRS 6)",
    3: "Dolor Severo (NRS 8)",
}

CH_NAMES = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    "TP9", "TP10",
]
assert len(CH_NAMES) == N_CHANNELS, f"CH_NAMES tiene {len(CH_NAMES)}, deben ser {N_CHANNELS}"

# Grupos de electrodos por región (para el texto clínico).
CENTRAL_SOMATOSENSORY = {"C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FCz", "CP1", "CPz", "CP2"}
EOG_CHANNELS = {"VEO", "HEO", "HEOR", "HEOL", "VEOU", "VEOL", "EOG"}
FRONTAL_OCULAR = {"Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"}
TEMPORAL_MUSCLE = {"T7", "T8", "FT7", "FT8", "TP7", "TP8", "TP9", "TP10"}
# Canales que NO van en el cuero cabelludo (se excluyen solo al graficar el topomap).
NO_SCALP_CHANNELS = EOG_CHANNELS | {"EKG", "ECG", "EMG", "M1", "M2"}

# --------------------------------------------------------------------
# CONSTANTES DE FUSIÓN (Pain Index — Hito V)
# --------------------------------------------------------------------
def _cargar_env(path: Path = ENV_FILE) -> None:
    """Lector .env mínimo (sin dependencias). Solo setea claves aún no definidas."""
    if not path.exists():
        return
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        k, v = linea.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_cargar_env()

W_CUERPO_DEFAULT = float(os.getenv("PI_W_CUERPO", "0.5"))
W_CEREBRO_DEFAULT = float(os.getenv("PI_W_CEREBRO", "0.5"))

CLASES_BIOVID = np.array([0, 1, 2, 3, 4], dtype=float)
NRS_POR_CLASE_BIOVID = np.array([0.0, 2.5, 5.0, 7.5, 10.0])

BODY_META_COLS = ["subject_id", "subject_name", "class_id", "class_name",
                  "sample_id", "sample_name"]

# Mapea la etiqueta de evento real del .fif (protocolo de estimulación)
# a la clase BioVid correspondiente (0=BL1 sin dolor ... 4=PA4 extremo).
# Confirmado con el usuario: NRS_2/4/6/8 = PA1/PA2/PA3/PA4.
NRS_LABEL_TO_BIOVID_CLASS = {
    "NRS_0": 0,   # BL1 - sin dolor (si existe en los datos)
    "NRS_2": 1,   # PA1
    "NRS_4": 2,   # PA2
    "NRS_6": 3,   # PA3
    "NRS_8": 4,   # PA4
}


# --------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Logger consistente para todo el proyecto (reemplaza los `print` sueltos)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
