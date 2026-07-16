"""
backend/eeg_engine.py
====================================================================
Motor de inferencia EEG: arquitectura EEGNet, carga del modelo,
lectura de datos (.fif / tensores sueltos), predicción y XAI
(Integrated Gradients).

Este módulo NO importa Streamlit ni sabe que existe una interfaz
gráfica — se puede probar y usar de forma completamente independiente
(ver src/tools/smoke_test_eeg.py).
====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from captum.attr import GradientShap, IntegratedGradients, LayerAttribution, LayerGradCam

from src import config

logger = config.get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ResultadoEEG:
    """Resultado estructurado de una inferencia EEG (evita tuplas anónimas
    tipo `nivel, pesos, texto, conf = ...` que son frágiles ante refactors)."""
    nivel_dolor: str
    pesos_electrodo: np.ndarray   # (61,) normalizado 0-1
    explicacion: str
    confianza: float
    clase: int


class EEGNet(nn.Module):
    """Arquitectura reconstruida para calzar exactamente con eegnet_fold1.pth.

    Hechos verificados del checkpoint:
        - 61 electrodos    (spatial_conv.0.weight = (16,1,63,1))
        - 4 clases         (classifier.1.weight   = (4,176))
        - Kernel temporal 125 (= fs/2 -> fs ~ 250 Hz)
        - Entrada de tiempo fija: 176 = 16 x 11 -> ~375 muestras (1.5 s @ 250 Hz)
    """

    def __init__(self, n_classes: int = config.N_CLASSES, channels: int = config.N_CHANNELS,
                 samples: int = config.N_SAMPLES, F1: int = 8, D: int = 2, F2: int = 16,
                 kern_length: int = 125, dropout: float = 0.5):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        # OJO: el kernel (channels,1) COLAPSA los 63 electrodos -> Grad-CAM
        # tras esta capa NO sirve para pesos por electrodo (por eso usamos IG).
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding="same", bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (samples // 32), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return self.classifier(x)


# ====================================================================
# CARGA DEL MODELO
# ====================================================================
def load_model(weights_path: Path = config.WEIGHTS_PATH, device: torch.device = DEVICE) -> EEGNet:
    """Carga el checkpoint entrenado.

    Lanza FileNotFoundError con mensaje explícito si no existe.
    A propósito NO hace fallback silencioso a pesos aleatorios: una
    predicción clínica basada en un modelo sin entrenar debe ser
    imposible de obtener por accidente (Fase 0.1 del plan de mejoras).
    """
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en:\n  {weights_path}\n"
            f"Este archivo es OBLIGATORIO: sin él no hay predicción clínica válida.\n"
            f"Coloca 'eegnet_fold1.pth' en la raíz del proyecto."
        )
    model = EEGNet().to(device)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("Modelo EEGNet cargado OK desde %s", weights_path)
    return model


# ====================================================================
# LECTURA DE DATOS
# ====================================================================
def leer_fif(path: str | Path):
    """Lee un .fif de MNE (Epochs), remuestrea a SFREQ y recorta a N_SAMPLES.

    Adaptado a que los .fif reales del paciente vienen a 1000 Hz / 1501
    muestras, mientras el modelo espera 250 Hz / 375 muestras.
    """
    import mne
    epochs = mne.read_epochs(str(path), preload=True, verbose="ERROR")
    if round(epochs.info["sfreq"]) != config.SFREQ:
        epochs.resample(config.SFREQ)
    n_times = epochs.get_data(copy=False).shape[-1]
    if n_times > config.N_SAMPLES:
        tmin = epochs.times[0]
        tmax = epochs.times[config.N_SAMPLES - 1]
        epochs.crop(tmin=tmin, tmax=tmax)
    return epochs


def epoca_a_tensor(epochs, idx: int = 0) -> torch.Tensor:
    """Extrae la época `idx` como tensor (n_canales, n_muestras)."""
    data = epochs.get_data(copy=True)
    return torch.as_tensor(data[idx], dtype=torch.float32)


def info_con_montaje(epochs):
    """Devuelve un info válido para plot_topomap; si el .fif no trae
    montaje, le pone el 10-20 estándar por nombre."""
    import mne
    info = epochs.info
    if epochs.get_montage() is None:
        info = info.copy()
        info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")
    return info


def event_label_para_epoca(epochs, idx: int) -> Optional[str]:
    """Devuelve el nombre del evento real asociado a la época `idx`
    (ej. 'NRS_6'), leído del propio .fif — NO es una predicción, es la
    etiqueta del protocolo de estimulación. Devuelve None si el .fif
    no trae event_id útil."""
    if not epochs.event_id:
        return None
    codigo = epochs.events[idx, 2]
    inverso = {v: k for k, v in epochs.event_id.items()}
    return inverso.get(codigo)


def build_info(sfreq: int = config.SFREQ):
    """Construye un info de MNE desde CH_NAMES (fuente B: tensores sueltos)."""
    import mne
    info = mne.create_info(config.CH_NAMES, sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")
    return info


# ====================================================================
# PREPROCESAMIENTO
# ====================================================================
def _normalizar(tensor_eeg: torch.Tensor) -> torch.Tensor:
    """Escala a la magnitud que el modelo espera (ver EEG_SCALE en config).
    Solo escala si el tensor parece venir en Voltios crudos."""
    if float(tensor_eeg.abs().max()) < config.RAW_VOLT_MAXABS:
        tensor_eeg = tensor_eeg * config.EEG_SCALE
    return tensor_eeg


def _prep_input(tensor_eeg: torch.Tensor) -> torch.Tensor:
    """Lleva el tensor a (1, 1, 61, T). Acepta (61,T), (1,61,T), (1,1,61,T)."""
    if not torch.is_tensor(tensor_eeg):
        tensor_eeg = torch.as_tensor(tensor_eeg, dtype=torch.float32)
    tensor_eeg = _normalizar(tensor_eeg.float())
    if tensor_eeg.dim() == 2:
        tensor_eeg = tensor_eeg.unsqueeze(0).unsqueeze(0)
    elif tensor_eeg.dim() == 3:
        tensor_eeg = tensor_eeg.unsqueeze(0)
    elif tensor_eeg.dim() != 4:
        raise ValueError(f"Forma no soportada: {tuple(tensor_eeg.shape)}")
    if tensor_eeg.shape[2] != config.N_CHANNELS:
        raise ValueError(f"Esperaba {config.N_CHANNELS} electrodos, recibí {tensor_eeg.shape[2]}. "
                         f"¿El tensor viene transpuesto?")
    if tensor_eeg.shape[3] != config.N_SAMPLES:
        raise ValueError(
            f"El modelo espera {config.N_SAMPLES} muestras de tiempo, recibí {tensor_eeg.shape[3]}.\n"
            f"  -> Si tu .fif está a otra frecuencia, remuestrea: epochs.resample({config.SFREQ})."
        )
    return tensor_eeg


def _texto_clinico(pesos_63: np.ndarray, ch_names: Sequence[str]) -> str:
    ch = ch_names[int(np.argmax(pesos_63))]
    if ch in config.EOG_CHANNELS:
        return (f"⚠️ ALERTA: el electrodo dominante es {ch}, un canal OCULAR (EOG). "
                f"La predicción se apoya en actividad de los ojos, NO en la corteza cerebral. "
                f"Resultado clínicamente NO fiable para esta época.")
    if ch in config.CENTRAL_SOMATOSENSORY:
        return f"Activación somatosensorial en {ch} (córtex central). Patrón fisiológico válido de procesamiento del dolor."
    if ch in config.FRONTAL_OCULAR:
        return f"⚠️ Advertencia: alta activación frontal en {ch}. Posible artefacto ocular/parpadeo — revisar."
    if ch in config.TEMPORAL_MUSCLE:
        return f"⚠️ Advertencia: activación temporal en {ch}. Posible artefacto muscular (EMG)."
    return f"Activación en región periférica (electrodo dominante: {ch}). Interpretar con cautela."


# ====================================================================
# INFERENCIA
# ====================================================================
def predecir_proba_cerebro(tensor_eeg: torch.Tensor, model_cargado: EEGNet) -> np.ndarray:
    """Softmax completo de las 4 clases NRS (lo necesita la capa de fusión)."""
    x = _prep_input(tensor_eeg).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model_cargado(x) / config.CONF_TEMPERATURE, dim=1)
    return probs.cpu().numpy().reshape(-1)


def procesar_onda_eeg(tensor_eeg: torch.Tensor, model_cargado: EEGNet,
                      ch_names: Optional[Sequence[str]] = None) -> ResultadoEEG:
    """Predicción + explicabilidad (Integrated Gradients) para una época EEG."""
    ch_names = list(ch_names) if ch_names is not None else config.CH_NAMES
    x = _prep_input(tensor_eeg).to(DEVICE)

    with torch.no_grad():
        logits = model_cargado(x)
        clase = int(logits.argmax(dim=1).item())
        probs = torch.softmax(logits / config.CONF_TEMPERATURE, dim=1)
        confianza = float(probs[0, clase].item())
    nivel_dolor = config.MAPA_DOLOR.get(clase, "Desconocido")

    ig = IntegratedGradients(model_cargado)
    inp = x.clone().requires_grad_(True)
    atribuciones = ig.attribute(inp, target=clase, n_steps=config.IG_STEPS)
    pesos_63 = atribuciones.detach().abs().mean(dim=3).reshape(-1).cpu().numpy()
    assert pesos_63.shape == (config.N_CHANNELS,), f"pesos shape {pesos_63.shape}"

    rango = float(pesos_63.max() - pesos_63.min())
    pesos_63 = (pesos_63 - pesos_63.min()) / rango if rango > 0 else np.zeros_like(pesos_63)

    explicacion = _texto_clinico(pesos_63, ch_names)
    return ResultadoEEG(nivel_dolor, pesos_63, explicacion, confianza, clase)


# ====================================================================
# XAI EXTENDIDA — SHAP (GradientShap) + Grad-CAM
# ====================================================================
# Todo lo de aquí abajo usa exclusivamente las dimensiones del tensor
# de entrada (x.shape) y config.SFREQ/N_SAMPLES — NADA hardcodeado a
# 63 canales. Cuando se reemplace el checkpoint por el de 61 canales
# (sin VEO/HEOR), esto sigue funcionando sin cambios: basta con
# actualizar config.N_CHANNELS / config.CH_NAMES.

@dataclass(frozen=True)
class ResultadoXAIExtendido:
    """SHAP (GradientShap) + Grad-CAM, complementando a Integrated Gradients
    (ya calculado en `procesar_onda_eeg`). Se agrupan en una sola llamada
    para reusar la MISMA clase predicha en los tres métodos — comparar
    métodos que explican clases distintas no tendría sentido."""
    clase: int
    shap_por_canal: np.ndarray            # (n_channels,) normalizado 0-1
    gradcam_canal_tiempo: np.ndarray      # (n_channels, n_samples) normalizado 0-1
    gradcam_ventana_temporal: np.ndarray  # (n_samples,) normalizado 0-1


def calcular_shap_eeg(tensor_eeg: torch.Tensor, model_cargado: EEGNet,
                      n_baselines: int = 20, n_samples: int = 25,
                      clase: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Aproximación SHAP para EEGNet vía GradientShap (Erion et al. 2021).

    No existe un dataset de fondo de EEG crudo representativo (a diferencia
    del modelo corporal, que sí lo tiene), así que el 'fondo' se simula con
    ruido gaussiano alrededor de cero, escalado a la magnitud real de la
    señal de entrada. Complementa a Integrated Gradients (que usa un único
    baseline fijo en cero) con una explicación basada en distribución de
    baselines — son dos aproximaciones distintas al mismo concepto de
    Shapley values, útiles para comparar consistencia (ver Fase 1.4).
    """
    x = _prep_input(tensor_eeg).to(DEVICE)
    if clase is None:
        with torch.no_grad():
            clase = int(model_cargado(x).argmax(dim=1).item())

    escala_ruido = float(x.std()) or 1.0
    baselines = torch.randn(n_baselines, *x.shape[1:], device=DEVICE) * escala_ruido

    gs = GradientShap(model_cargado)
    atribuciones = gs.attribute(x, baselines=baselines, target=clase,
                                n_samples=n_samples, stdevs=0.05)
    pesos = atribuciones.detach().abs().mean(dim=3).reshape(-1).cpu().numpy()

    rango = float(pesos.max() - pesos.min())
    pesos = (pesos - pesos.min()) / rango if rango > 0 else np.zeros_like(pesos)
    return pesos, clase


def grad_cam_temporal(tensor_eeg: torch.Tensor, model_cargado: EEGNet,
                      clase: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Grad-CAM sobre `temporal_conv` (la PRIMERA capa convolucional).

    Se elige esta capa a propósito: `spatial_conv` (la siguiente) usa un
    kernel (n_channels, 1) que COLAPSA la dimensión de electrodo a 1 — si
    se hiciera Grad-CAM después de esa capa, se perdería toda resolución
    por canal. `temporal_conv` preserva electrodo × tiempo intactos
    (kernel (1, K) con padding='same'), así que el mapa resultante tiene
    la resolución completa (n_channels, n_samples) SIN necesidad de
    interpolar.
    """
    x = _prep_input(tensor_eeg).to(DEVICE)
    if clase is None:
        with torch.no_grad():
            clase = int(model_cargado(x).argmax(dim=1).item())

    gc = LayerGradCam(model_cargado, model_cargado.temporal_conv)
    mapa = gc.attribute(x, target=clase, relu_attributions=True)
    mapa = mapa.detach().cpu().numpy().reshape(mapa.shape[-2], mapa.shape[-1])  # (n_channels, n_samples)

    pico = float(mapa.max())
    if pico > 0:
        mapa = mapa / pico
    return mapa, clase


def grad_cam_ventana_temporal(tensor_eeg: torch.Tensor, model_cargado: EEGNet,
                              clase: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Grad-CAM sobre `separable_conv` (capa profunda, tras el pooling).

    Para esta altura de la red, `spatial_conv` YA colapsó los electrodos
    (dimensión de canal = 1) — este mapa NO tiene resolución por electrodo,
    solo temporal. Sirve para responder "¿qué ventana de tiempo dentro de
    la época fue más determinante?" (relevante para 1.4: contrastar contra
    ventanas esperadas de N2/P300/gamma en la literatura). Como el pooling
    reduce la resolución temporal (÷32), se interpola de vuelta al largo
    original de la época para poder graficarlo alineado con los otros XAI.
    """
    x = _prep_input(tensor_eeg).to(DEVICE)
    n_samples = x.shape[-1]
    if clase is None:
        with torch.no_grad():
            clase = int(model_cargado(x).argmax(dim=1).item())

    gc = LayerGradCam(model_cargado, model_cargado.separable_conv)
    mapa = gc.attribute(x, target=clase, relu_attributions=True)
    mapa = LayerAttribution.interpolate(mapa, (1, n_samples))
    serie = mapa.detach().cpu().numpy().reshape(-1)

    pico = float(serie.max())
    if pico > 0:
        serie = serie / pico
    return serie, clase


def calcular_xai_extendido(tensor_eeg: torch.Tensor, model_cargado: EEGNet) -> ResultadoXAIExtendido:
    """Calcula SHAP + los dos Grad-CAM de una sola vez, sobre la MISMA
    clase predicha (evita comparar explicaciones de clases distintas).

    Es más costoso que Integrated Gradients solo (3 pasadas backward
    adicionales) — en el frontend se deja gatillado por botón, no
    automático en cada tick del modo reproducción.
    """
    x = _prep_input(tensor_eeg).to(DEVICE)
    with torch.no_grad():
        clase = int(model_cargado(x).argmax(dim=1).item())

    shap_canal, _ = calcular_shap_eeg(tensor_eeg, model_cargado, clase=clase)
    gradcam_ct, _ = grad_cam_temporal(tensor_eeg, model_cargado, clase=clase)
    gradcam_vt, _ = grad_cam_ventana_temporal(tensor_eeg, model_cargado, clase=clase)

    return ResultadoXAIExtendido(
        clase=clase,
        shap_por_canal=shap_canal,
        gradcam_canal_tiempo=gradcam_ct,
        gradcam_ventana_temporal=gradcam_vt,
    )


# ====================================================================
# LIME-EEG — aproximación por OCLUSIÓN
# ====================================================================
# LIME clásico (lime.lime_tabular) asume features tabulares discretas e
# independientes — no aplica a una señal continua electrodo×tiempo. La
# adaptación estándar (usada en visión e interpretabilidad de señales) es
# la oclusión: silenciar una parte de la entrada y medir cuánto CAE la
# probabilidad de la clase predicha. Es perturbación real sobre el
# modelo (forward puro, sin gradientes) — por eso sirve como validación
# cruzada independiente de IG/SHAP/Grad-CAM, que sí usan gradientes.

@dataclass(frozen=True)
class ResultadoLimeEEG:
    clase: int
    importancia_por_canal: np.ndarray    # (n_channels,) 0-1
    importancia_por_tiempo: np.ndarray   # (n_samples,) 0-1


def _prob_de_clase(x_batch: torch.Tensor, model_cargado: EEGNet, clase: int) -> np.ndarray:
    """Probabilidad de `clase` para cada muestra de un batch."""
    with torch.no_grad():
        probs = torch.softmax(model_cargado(x_batch) / config.CONF_TEMPERATURE, dim=1)
    return probs[:, clase].cpu().numpy()


def lime_eeg_por_canal(tensor_eeg: torch.Tensor, model_cargado: EEGNet,
                       clase: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Ocluye un electrodo a la vez (lo pone en 0) y mide cuánto cae la
    probabilidad de la clase predicha. Mayor caída = electrodo más
    importante. Todas las oclusiones se evalúan en UN solo forward
    batcheado (rápido: no necesita backward, a diferencia de Grad-CAM/SHAP)."""
    x = _prep_input(tensor_eeg).to(DEVICE)
    n_channels = x.shape[2]
    if clase is None:
        with torch.no_grad():
            clase = int(model_cargado(x).argmax(dim=1).item())

    prob_base = float(_prob_de_clase(x, model_cargado, clase)[0])

    ocluidos = x.repeat(n_channels, 1, 1, 1)
    for i in range(n_channels):
        ocluidos[i, 0, i, :] = 0.0

    prob_ocluida = _prob_de_clase(ocluidos, model_cargado, clase)
    caida = np.clip(prob_base - prob_ocluida, a_min=0, a_max=None)

    pico = float(caida.max())
    importancia = caida / pico if pico > 0 else np.zeros_like(caida)
    return importancia, clase


def lime_eeg_por_ventana(tensor_eeg: torch.Tensor, model_cargado: EEGNet,
                         clase: Optional[int] = None, n_ventanas: int = 15) -> tuple[np.ndarray, int]:
    """Igual que `lime_eeg_por_canal`, pero ocluyendo bloques de TIEMPO
    (todos los electrodos a la vez) — da la contraparte temporal."""
    x = _prep_input(tensor_eeg).to(DEVICE)
    n_samples = x.shape[-1]
    if clase is None:
        with torch.no_grad():
            clase = int(model_cargado(x).argmax(dim=1).item())

    prob_base = float(_prob_de_clase(x, model_cargado, clase)[0])

    limites = np.linspace(0, n_samples, n_ventanas + 1).astype(int)
    ocluidos = x.repeat(n_ventanas, 1, 1, 1)
    for i in range(n_ventanas):
        ocluidos[i, 0, :, limites[i]:limites[i + 1]] = 0.0

    prob_ocluida = _prob_de_clase(ocluidos, model_cargado, clase)
    caida = np.clip(prob_base - prob_ocluida, a_min=0, a_max=None)
    pico = float(caida.max())
    importancia_ventanas = caida / pico if pico > 0 else np.zeros_like(caida)

    # Expandir de n_ventanas a n_samples para graficar alineado con Grad-CAM/SHAP.
    serie = np.repeat(importancia_ventanas, np.diff(limites))
    if len(serie) < n_samples:
        serie = np.pad(serie, (0, n_samples - len(serie)), mode="edge")
    elif len(serie) > n_samples:
        serie = serie[:n_samples]
    return serie, clase


def calcular_lime_eeg(tensor_eeg: torch.Tensor, model_cargado: EEGNet,
                      n_ventanas: int = 15) -> ResultadoLimeEEG:
    """Calcula ambas oclusiones (canal + tiempo) sobre la MISMA clase predicha."""
    x = _prep_input(tensor_eeg).to(DEVICE)
    with torch.no_grad():
        clase = int(model_cargado(x).argmax(dim=1).item())

    canal, _ = lime_eeg_por_canal(tensor_eeg, model_cargado, clase=clase)
    tiempo, _ = lime_eeg_por_ventana(tensor_eeg, model_cargado, clase=clase, n_ventanas=n_ventanas)

    return ResultadoLimeEEG(clase=clase, importancia_por_canal=canal, importancia_por_tiempo=tiempo)
