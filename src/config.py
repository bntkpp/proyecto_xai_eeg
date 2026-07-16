import logging
import os
from pathlib import Path
import numpy as np

# --------------------------------------------------------------------
# RUTAS DEL PROYECTO
# --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

EEG_DATA_DIR = DATA_DIR / "eeg"
AUTONOMIC_DATA_DIR = DATA_DIR / "autonomic"

# Rutas estáticas
EEG_MODEL_PATH = MODELS_DIR / "eegnet_model.pth"
AUTONOMIC_MODEL_PATH = MODELS_DIR / "xgboost_autonomic.json"
CALIBRATOR_PATH = MODELS_DIR / "isotonic_calibrator.pkl"
LOGO_PATH = ASSETS_DIR / "logo.jpeg"
ENV_FILE = ROOT_DIR / ".env"

# Alias para los motores
WEIGHTS_PATH = EEG_MODEL_PATH
BODY_MODEL_PATH = AUTONOMIC_MODEL_PATH
BODY_DATASET_PATH = AUTONOMIC_DATA_DIR / "dataset_elite_biovid.csv" 

# --------------------------------------------------------------------
# CONSTANTES DEL MODELO EEG
# --------------------------------------------------------------------
N_CLASSES = 4
N_CHANNELS = 61          
N_SAMPLES = 375          
SFREQ = 250.0            

EEG_SCALE = 8.7e4
RAW_VOLT_MAXABS = 1e-2    
CONF_TEMPERATURE = 0.903

MAPA_DOLOR = {0: "Sin Dolor (NRS 0-2)", 1: "Dolor Leve (NRS 4)", 2: "Dolor Moderado (NRS 6)", 3: "Dolor Severo (NRS 8)"}
BIOVID_LABELS = ["BL1 (sin dolor)", "PA1 (leve)", "PA2 (moderado)", "PA3 (fuerte)", "PA4 (extremo)"]

CH_NAMES = [
    'Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 
    'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9', 'Pz', 'P1', 
    'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 
    'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 
    'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2'
]

CENTRAL_SOMATOSENSORY = {"C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FCz", "CP1", "CPz", "CP2"}
EOG_CHANNELS = {"VEO", "HEO", "HEOR", "HEOL", "VEOU", "VEOL", "EOG"}
FRONTAL_OCULAR = {"Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"}
TEMPORAL_MUSCLE = {"T7", "T8", "FT7", "FT8", "TP7", "TP8", "TP9", "TP10"}

NO_SCALP_CHANNELS = EOG_CHANNELS | {"EKG", "ECG", "EMG", "M1", "M2"}
IG_STEPS = 50

# --------------------------------------------------------------------
# CONSTANTES DE FUSIÓN Y CUERPO
# --------------------------------------------------------------------
def _cargar_env(path: Path = ENV_FILE) -> None:
    if not path.exists(): return
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea: continue
        k, v = linea.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

_cargar_env()

W_CUERPO_DEFAULT = float(os.getenv("PI_W_CUERPO", "0.5"))
W_CEREBRO_DEFAULT = float(os.getenv("PI_W_CEREBRO", "0.5"))

CLASES_BIOVID = np.array([0, 1, 2, 3, 4], dtype=float)
NRS_POR_CLASE_BIOVID = np.array([0.0, 2.5, 5.0, 7.5, 10.0])

# ¡Aquí está la variable que faltaba!
BODY_META_COLS = ["subject_id", "subject_name", "class_id", "class_name", "sample_id", "sample_name"]

N2_WINDOW_MS = (200, 350)
P300_WINDOW_MS = (250, 500)
GAMMA_BAND_HZ = (30.0, 80.0)

NRS_LABEL_TO_BIOVID_CLASS = {"NRS_0": 0, "NRS_2": 1, "NRS_4": 2, "NRS_6": 3, "NRS_8": 4}

# --------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
