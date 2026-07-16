"""
inference_backend.py
====================================================================
MOTOR DE INFERENCIA (Backend) — Hito 4 / Actividad 3
XAI y Mapeo Cerebral en Tiempo Real con EEGNet + Captum + MNE

Arquitectura RECONSTRUIDA leyendo tu `eegnet_fold1.pth`, así que
`load_state_dict(..., strict=True)` calza sin errores. Hechos verificados:
    - 63 electrodos     (spatial_conv.0.weight = (16,1,63,1))
    - 4 clases          (classifier.1.weight   = (4,176))
    - Kernel temporal 125  (= fs/2  ->  fs ~ 250 Hz)
    - Entrada de tiempo FIJA: 176 = 16 x 11 -> ~375 muestras (1.5 s @ 250 Hz)

Soporta dos fuentes de datos:
    A) Archivos .fif epocados de MNE  (TU CASO REAL)  -> nombres y montaje
       salen del propio archivo, no hay que escribir CH_NAMES a mano.
    B) Tensores sueltos .npy / .pt    -> usa la lista CH_NAMES de abajo.
====================================================================
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

# ====================================================================
# RUTAS (robustas: se calculan desde la ubicación de este archivo)
# ====================================================================
_HERE = Path(__file__).resolve().parent          # .../proyecto_xai_eeg/src
ROOT  = _HERE.parent                              # .../proyecto_xai_eeg
WEIGHTS_PATH = str(ROOT / "eegnet_fold1.pth")     # pon el modelo en la raíz del proyecto
DATA_DIR     = str(ROOT / "data")                 # pon los .fif aquí

# ====================================================================
# CONFIGURACIÓN
# ====================================================================
N_CLASSES  = 4
N_CHANNELS = 63
N_SAMPLES  = 375        # muestras de tiempo esperadas (verifícalo con leer_canales.py)
SFREQ      = 250        # Hz
IG_STEPS   = 50         # pasos de Integrated Gradients (baja a ~25 si necesitas velocidad)

# --------------------------------------------------------------------
# NORMALIZACIÓN DE ENTRADA  (CRÍTICO — ver nota)
# --------------------------------------------------------------------
# El modelo NO se entrenó con Voltios crudos. Si se le pasan datos en
# Voltios (~1e-6) sin escalar, SATURA y predice SIEMPRE la clase 0
# ("Sin Dolor"): la app daría una constante sin sentido.
#
# EEG_SCALE se RECUPERÓ por forense del propio .pth: la varianza guardada
# en `temporal_conv.1.running_var` (BatchNorm) implica que el entrenamiento
# escaló la entrada por ~8.7e4 (equivale a dividir por un std de dataset
# ~11 µV; deja la salida de BN1 en media≈0, var≈1). Es PROVISIONAL: hay
# ~±15% de incertidumbre en la constante exacta porque estos sujetos no son
# idénticos a los de entrenamiento. Si algún día aparece el script de
# entrenamiento, sustituye esto por su normalización exacta.
EEG_SCALE = 8.7e4
# Umbral para decidir si un tensor ya viene normalizado (.npy/.pt) o en
# Voltios crudos: si su magnitud máxima es < este valor, asumimos Voltios.
_RAW_VOLT_MAXABS = 1e-2

# --------------------------------------------------------------------
# CALIBRACIÓN DE CONFIANZA  (temperature scaling)
# --------------------------------------------------------------------
# El % de confianza de un softmax crudo no es una probabilidad honesta.
# CONF_TEMPERATURE se ajustó minimizando NLL sobre una mitad de los sujetos
# y se validó en la otra mitad (sin solapamiento). NO cambia la clase
# predicha (T es un escalar), solo el % mostrado.
#
# Hallazgo: el modelo YA estaba bien calibrado a nivel global (ECE ~3.5%,
# T≈0.90), así que el ajuste es pequeño. OJO: las confianzas ALTAS (>55%)
# están algo SOBREESTIMADAS (dice ~81% y acierta ~72%); un único T no corrige
# eso del todo. Para uso clínico, leer una confianza alta con cautela.
CONF_TEMPERATURE = 0.903

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAPA_DOLOR = {
    0: "Sin Dolor (NRS 0-2)",
    1: "Dolor Leve (NRS 4)",
    2: "Dolor Moderado (NRS 6)",
    3: "Dolor Severo (NRS 8)",
}

# --------------------------------------------------------------------
# CH_NAMES: solo se usa con la FUENTE B (.npy/.pt). Con .fif los nombres
# salen del archivo. Placeholder 10-10 para que el código corra.
# --------------------------------------------------------------------
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

# Grupos por región (POR NOMBRE de electrodo). Ver tabla del plan, sección 4.
CENTRAL_SOMATOSENSORY = {"C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                         "FCz", "CP1", "CPz", "CP2"}
# Canales OCULARES de verdad (electrodos EOG). Si el modelo se apoya en
# estos, la predicción se basa en los OJOS, no en el cerebro: es el caso
# más grave y debe alertarse explícitamente.
EOG_CHANNELS          = {"VEO", "HEO", "HEOR", "HEOL", "VEOU", "VEOL", "EOG"}
FRONTAL_OCULAR        = {"Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"}
TEMPORAL_MUSCLE       = {"T7", "T8", "FT7", "FT8", "TP7", "TP8", "TP9", "TP10"}


# ====================================================================
# ARQUITECTURA EEGNet (calza con tu .pth)
# ====================================================================
class EEGNet(nn.Module):
    def __init__(self, n_classes=4, channels=63, samples=375,
                 F1=8, D=2, F2=16, kern_length=125, dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False),   # (8,1,1,125)
            nn.BatchNorm2d(F1),
        )
        # OJO: el kernel (channels,1) COLAPSA los 63 electrodos -> Grad-CAM
        # tras esta capa NO sirve para pesos por electrodo.
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),       # (16,1,63,1)
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding="same", bias=False),  # (16,1,1,16)
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),                                       # (16,16,1,1)
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (samples // 32), n_classes),                        # (4,176) si samples=375
        )

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x


# ====================================================================
# CARGA DEL MODELO  (UNA SOLA VEZ — ver app_streamlit.py @st.cache_resource)
# ====================================================================
def load_model(weights_path=WEIGHTS_PATH, device=DEVICE):
    model = EEGNet(n_classes=N_CLASSES, channels=N_CHANNELS, samples=N_SAMPLES).to(device)
    state = torch.load(weights_path, map_location=device)
    # state = state.get("model_state_dict", state)  # descomenta si guardaste un checkpoint completo
    model.load_state_dict(state, strict=True)
    model.eval()                                     # OBLIGATORIO
    return model


# ====================================================================
# FUENTE A: leer archivos .fif (Epochs de MNE)  -- TU CASO REAL
# ====================================================================
def leer_fif(path):
    """Carga un .fif epocado y devuelve el objeto Epochs de MNE.

    Si el archivo está a otra frecuencia que la del modelo (SFREQ=250 Hz),
    lo remuestrea automáticamente. Ej.: 1000 Hz / 1501 muestras -> 250 Hz / ~375.
    """
    import mne
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    if round(epochs.info["sfreq"]) != SFREQ:
        epochs.resample(SFREQ)
    # Asegura EXACTAMENTE N_SAMPLES muestras (el resample puede dejar 375 o 376).
    n_times = epochs.get_data(copy=False).shape[-1]
    if n_times > N_SAMPLES:
        # recorta las muestras sobrantes al final (mantiene el inicio de la época)
        tmin = epochs.times[0]
        tmax = epochs.times[N_SAMPLES - 1]
        epochs.crop(tmin=tmin, tmax=tmax)
    return epochs


def epoca_a_tensor(epochs, idx=0):
    """Extrae la época `idx` como tensor (n_canales, n_muestras)."""
    data = epochs.get_data(copy=True)               # (n_epochs, n_ch, n_times)
    return torch.as_tensor(data[idx], dtype=torch.float32)


def info_con_montaje(epochs):
    """Devuelve un info válido para plot_topomap; si el .fif no trae
    montaje, le pone el 10-20 estándar por nombre."""
    import mne
    info = epochs.info
    if epochs.get_montage() is None:
        info = info.copy()
        info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                         on_missing="warn")
    return info


# ====================================================================
# FUENTE B: construir info desde CH_NAMES (para .npy/.pt)
# ====================================================================
def build_info(sfreq=SFREQ):
    import mne
    info = mne.create_info(CH_NAMES, sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")
    return info


# ====================================================================
# UTILIDADES
# ====================================================================
def _normalizar(tensor_eeg):
    """Escala la entrada a la magnitud que el modelo espera (ver EEG_SCALE).

    Solo escala si el tensor parece estar en Voltios crudos (magnitud máxima
    < _RAW_VOLT_MAXABS). Así un .npy/.pt que ya venga normalizado no se
    re-escala por error. Devuelve el tensor escalado.
    """
    if float(tensor_eeg.abs().max()) < _RAW_VOLT_MAXABS:
        tensor_eeg = tensor_eeg * EEG_SCALE
    return tensor_eeg


def _prep_input(tensor_eeg):
    """Lleva el tensor a (1, 1, 63, T). Acepta (63,T), (1,63,T), (1,1,63,T)."""
    if not torch.is_tensor(tensor_eeg):
        tensor_eeg = torch.as_tensor(tensor_eeg, dtype=torch.float32)
    tensor_eeg = tensor_eeg.float()
    tensor_eeg = _normalizar(tensor_eeg)
    if tensor_eeg.dim() == 2:
        tensor_eeg = tensor_eeg.unsqueeze(0).unsqueeze(0)
    elif tensor_eeg.dim() == 3:
        tensor_eeg = tensor_eeg.unsqueeze(0)
    elif tensor_eeg.dim() != 4:
        raise ValueError(f"Forma no soportada: {tuple(tensor_eeg.shape)}")
    if tensor_eeg.shape[2] != N_CHANNELS:
        raise ValueError(f"Esperaba {N_CHANNELS} electrodos, recibí {tensor_eeg.shape[2]}. "
                         f"¿El tensor viene transpuesto?")
    if tensor_eeg.shape[3] != N_SAMPLES:
        raise ValueError(
            f"El modelo espera {N_SAMPLES} muestras de tiempo, recibí {tensor_eeg.shape[3]}.\n"
            f"  -> Si tu .fif está a otra frecuencia, remuestrea: epochs.resample({SFREQ}).\n"
            f"  -> O recorta a 1.5 s. (Esto se decide en el Hito 4.3.1)."
        )
    return tensor_eeg


def _texto_clinico(pesos_63, ch_names):
    idx_max = int(np.argmax(pesos_63))
    ch = ch_names[idx_max]
    if ch in EOG_CHANNELS:
        return (f"⚠️ ALERTA: el electrodo dominante es {ch}, un canal OCULAR (EOG). "
                f"La predicción se está apoyando en actividad de los ojos "
                f"(parpadeo/movimiento), NO en la corteza cerebral. "
                f"Resultado clínicamente NO fiable para esta época.")
    if ch in CENTRAL_SOMATOSENSORY:
        return (f"Activación somatosensorial en {ch} (córtex central). "
                f"Patrón fisiológico válido de procesamiento del dolor.")
    if ch in FRONTAL_OCULAR:
        return (f"⚠️ Advertencia: alta activación frontal en {ch}. "
                f"Posible artefacto ocular/parpadeo o actividad cognitiva — revisar.")
    if ch in TEMPORAL_MUSCLE:
        return (f"⚠️ Advertencia: activación temporal en {ch}. "
                f"Posible artefacto muscular (EMG).")
    return (f"Activación en región periférica (electrodo dominante: {ch}). "
            f"Interpretar con cautela.")


# ====================================================================
# FUNCIÓN PRINCIPAL DEL BACKEND
# ====================================================================
def predecir_proba_cerebro(tensor_eeg, model_cargado):
    """Devuelve el vector softmax COMPLETO de las 4 clases NRS (soft output).

    Lo necesita la capa de fusión multimodal (Hito V): la fusión combina las
    distribuciones de probabilidad de ambos modelos, no solo la clase final.
    Aplica la misma normalización y temperatura de calibración que la app.

    Retorna: np.ndarray de forma (4,) que suma 1.
    """
    x = _prep_input(tensor_eeg).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model_cargado(x) / CONF_TEMPERATURE, dim=1)
    return probs.cpu().numpy().reshape(-1)


def procesar_onda_eeg(tensor_eeg, model_cargado, ch_names=None):
    """
    Recibe: tensor EEG (63,T)/(1,1,63,T), el modelo ya cargado, y opcionalmente
            los nombres de canal reales (del .fif). Si no se pasan, usa CH_NAMES.
    Retorna: nivel_dolor(str), pesos_63(np.ndarray (63,)), explicacion(str), confianza(float)
    """
    if ch_names is None:
        ch_names = CH_NAMES
    x = _prep_input(tensor_eeg).to(DEVICE)

    # 1) PREDICCIÓN (sin gradientes)
    with torch.no_grad():
        logits = model_cargado(x)
        clase = int(logits.argmax(dim=1).item())          # argmax no depende de T
        probs = torch.softmax(logits / CONF_TEMPERATURE, dim=1)   # confianza calibrada
        confianza = float(probs[0, clase].item())
    nivel_dolor = MAPA_DOLOR.get(clase, "Desconocido")

    # 2) XAI con Integrated Gradients (atribución sobre la ENTRADA -> 63 pesos)
    ig = IntegratedGradients(model_cargado)
    inp = x.clone().requires_grad_(True)
    atribuciones = ig.attribute(inp, target=clase, n_steps=IG_STEPS)   # (1,1,63,T)
    pesos_63 = atribuciones.detach().abs().mean(dim=3).reshape(-1).cpu().numpy()
    assert pesos_63.shape == (N_CHANNELS,), f"pesos shape {pesos_63.shape}"

    rango = float(pesos_63.max() - pesos_63.min())
    pesos_63 = (pesos_63 - pesos_63.min()) / rango if rango > 0 else np.zeros_like(pesos_63)

    # 3) Texto clínico (por NOMBRE de electrodo)
    explicacion = _texto_clinico(pesos_63, ch_names)
    return nivel_dolor, pesos_63, explicacion, confianza


# ====================================================================
# AUTO-TEST:  python src/inference_backend.py
# ====================================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Modelo esperado en: {WEIGHTS_PATH}")
    try:
        modelo = load_model()
        print("Modelo cargado OK (state_dict calza).")
    except FileNotFoundError:
        print("No encontré el .pth; uso modelo sin entrenar solo para probar el flujo.")
        modelo = EEGNet(N_CLASSES, N_CHANNELS, N_SAMPLES).to(DEVICE).eval()

    dummy = torch.randn(1, 1, N_CHANNELS, N_SAMPLES)
    nivel, pesos, texto, conf = procesar_onda_eeg(dummy, modelo)
    print(f"Predicción : {nivel}  (confianza {conf:.0%})")
    print(f"Electrodo  : {CH_NAMES[int(np.argmax(pesos))]}")
    print(f"Explicación: {texto}")
