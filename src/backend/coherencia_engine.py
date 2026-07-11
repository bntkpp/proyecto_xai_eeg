"""
backend/coherencia_engine.py
====================================================================
Fase 1.4 — Análisis de coherencia neurofisiológica.

Contrasta las explicaciones de los 4 métodos XAI ya calculados
(Integrated Gradients, SHAP-GradientShap, Grad-CAM, LIME-oclusión)
contra literatura neurofisiológica del dolor:

  - Componentes evocados relacionados con dolor: N2 (~200-350ms) y
    P300/P3 (~250-500ms), típicamente máximos en electrodos
    centro-parietales (Cz, CPz, Pz, C1-C6...).
  - Sincronización gamma (30-80Hz) asociada a procesamiento
    nociceptivo, también con topografía central.

IMPORTANTE — esto NO valida que el modelo esté "bien" en sentido
absoluto, ni reemplaza la validación de un neurofisiólogo. Valida que
sus explicaciones sean CONSISTENTES con patrones fisiológicos
conocidos. Un modelo puede acertar la clase por razones espurias
(artefactos musculares u oculares) y aun así fallar esta coherencia —
esa es justamente la señal de alerta que este módulo busca exponer,
no ocultar.
====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from src import config

logger = config.get_logger(__name__)


@dataclass(frozen=True)
class ResultadoCoherencia:
    clase: int
    canal_pico_por_metodo: dict[str, str]
    region_consistente_por_metodo: dict[str, bool]
    ventana_pico_ms_por_metodo: dict[str, float]
    ventana_consistente_por_metodo: dict[str, bool]
    correlacion_gamma_por_metodo: dict[str, float]
    potencia_gamma_por_canal: np.ndarray
    resumen_texto: str


def _a_2d(tensor_eeg) -> np.ndarray:
    """Reduce cualquier tensor/array con dims extra de tamaño 1 a (canales, tiempo)."""
    arr = tensor_eeg.detach().cpu().numpy() if torch.is_tensor(tensor_eeg) else np.asarray(tensor_eeg)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Se esperaba un tensor reducible a (canales, tiempo), quedó {arr.shape}")
    return arr


def potencia_banda(tensor_eeg, sfreq: int, banda: tuple[float, float] = None) -> np.ndarray:
    """Potencia espectral (Welch) en una banda de frecuencia, por canal,
    normalizada 0-1. Por defecto usa la banda gamma (config.GAMMA_BAND_HZ)."""
    banda = banda or config.GAMMA_BAND_HZ
    señal = _a_2d(tensor_eeg)
    nperseg = min(128, señal.shape[-1])
    freqs, psd = scipy_signal.welch(señal, fs=sfreq, nperseg=nperseg, axis=-1)
    mascara = (freqs >= banda[0]) & (freqs <= banda[1])
    potencia = psd[:, mascara].mean(axis=1)
    pico = float(potencia.max())
    return potencia / pico if pico > 0 else np.zeros_like(potencia)


def _canal_pico(importancia: np.ndarray, ch_names) -> str:
    return ch_names[int(np.argmax(importancia))]


def _ventana_pico_ms(importancia_tiempo: np.ndarray, sfreq: int) -> float:
    idx = int(np.argmax(importancia_tiempo))
    return round(idx / sfreq * 1000, 1)


def _en_ventana(ms: float, ventana: tuple[float, float]) -> bool:
    return ventana[0] <= ms <= ventana[1]


def evaluar_coherencia(*, tensor_eeg, ch_names, resultado_eeg, xai_ext, lime_eeg,
                       sfreq: int = config.SFREQ) -> ResultadoCoherencia:
    """Arma el reporte de coherencia. Requiere que ya se hayan calculado
    los tres resultados XAI (parámetros `resultado_eeg`, `xai_ext`,
    `lime_eeg`) para la MISMA época — el frontend es responsable de no
    llamar esto hasta que los tres estén disponibles.
    """
    # ---- 1. Región: ¿el canal pico de cada método es somatosensorial central? ----
    canales_por_metodo = {
        "Integrated Gradients": resultado_eeg.pesos_electrodo,
        "SHAP (GradientShap)": xai_ext.shap_por_canal,
        "Grad-CAM (canal×tiempo, promedio)": xai_ext.gradcam_canal_tiempo.mean(axis=1),
        "LIME-EEG (oclusión)": lime_eeg.importancia_por_canal,
    }
    canal_pico_por_metodo = {m: _canal_pico(v, ch_names) for m, v in canales_por_metodo.items()}
    region_consistente_por_metodo = {
        m: (canal in config.CENTRAL_SOMATOSENSORY) for m, canal in canal_pico_por_metodo.items()
    }

    # ---- 2. Ventana temporal: ¿el pico cae en N2 o P300? ----
    tiempo_por_metodo = {
        "Grad-CAM (ventana temporal)": xai_ext.gradcam_ventana_temporal,
        "LIME-EEG (ventana temporal)": lime_eeg.importancia_por_tiempo,
    }
    ventana_pico_ms_por_metodo = {m: _ventana_pico_ms(v, sfreq) for m, v in tiempo_por_metodo.items()}
    ventana_consistente_por_metodo = {
        m: (_en_ventana(ms, config.N2_WINDOW_MS) or _en_ventana(ms, config.P300_WINDOW_MS))
        for m, ms in ventana_pico_ms_por_metodo.items()
    }

    # ---- 3. Gamma: ¿la topografía de importancia correlaciona con potencia gamma real? ----
    potencia_gamma = potencia_banda(tensor_eeg, sfreq, config.GAMMA_BAND_HZ)
    correlacion_gamma_por_metodo = {}
    for metodo, importancia in canales_por_metodo.items():
        rho, _ = scipy_stats.spearmanr(importancia, potencia_gamma)
        correlacion_gamma_por_metodo[metodo] = float(rho) if not np.isnan(rho) else 0.0

    # ---- 4. Resumen narrativo ----
    n_region_ok = sum(region_consistente_por_metodo.values())
    n_ventana_ok = sum(ventana_consistente_por_metodo.values())
    n_gamma_ok = sum(1 for r in correlacion_gamma_por_metodo.values() if r > 0.2)
    resumen = (
        f"Región central-somatosensorial: {n_region_ok}/4 métodos consistentes. "
        f"Ventana N2/P300: {n_ventana_ok}/2 métodos consistentes. "
        f"Correlación con gamma real: {n_gamma_ok}/4 métodos con ρ>0.2."
    )

    return ResultadoCoherencia(
        clase=resultado_eeg.clase,
        canal_pico_por_metodo=canal_pico_por_metodo,
        region_consistente_por_metodo=region_consistente_por_metodo,
        ventana_pico_ms_por_metodo=ventana_pico_ms_por_metodo,
        ventana_consistente_por_metodo=ventana_consistente_por_metodo,
        correlacion_gamma_por_metodo=correlacion_gamma_por_metodo,
        potencia_gamma_por_canal=potencia_gamma,
        resumen_texto=resumen,
    )