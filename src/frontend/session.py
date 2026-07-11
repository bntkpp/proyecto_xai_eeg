"""
frontend/session.py
====================================================================
Manejo del estado de sesión (`st.session_state`):
  - Historial del Pain Index a través de las épocas analizadas en
    esta sesión (manual o automáticamente).
  - Control del modo de reproducción automática (play/pause).

Streamlit re-ejecuta el script COMPLETO en cada interacción; sin
st.session_state, cualquier historial se perdería en cada rerun. Este
módulo es el único que toca session_state directamente — el resto del
frontend pasa por estas funciones, nunca por claves sueltas.
====================================================================
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

_HISTORIAL_KEY = "historial_pain_index"
_PLAYING_KEY = "reproduccion_automatica"
_EPOCA_KEY = "epoca_actual"
_DETALLE_KEY = "detalle_eeg_visible"

_COLUMNAS_HISTORIAL = [
    "timestamp", "archivo", "epoca", "pain_index",
    "nivel_eeg", "confianza_eeg", "cuerpo_pred", "cerebro_pred",
]


def init_session_state() -> None:
    """Inicializa las claves de sesión si no existen todavía (idempotente:
    seguro de llamar en cada rerun del script)."""
    st.session_state.setdefault(_HISTORIAL_KEY, [])
    st.session_state.setdefault(_PLAYING_KEY, False)
    st.session_state.setdefault(_EPOCA_KEY, 0)
    st.session_state.setdefault(_DETALLE_KEY, False)


# ---- Modo de reproducción ------------------------------------------

def is_playing() -> bool:
    return bool(st.session_state.get(_PLAYING_KEY, False))


def set_playing(valor: bool) -> None:
    st.session_state[_PLAYING_KEY] = bool(valor)


# ---- Época actual (fuente de verdad, sincronizada con el slider) ----

def get_epoca_actual() -> int:
    return int(st.session_state.get(_EPOCA_KEY, 0))


def set_epoca_actual(idx: int) -> None:
    st.session_state[_EPOCA_KEY] = int(idx)


# ---- Detalle EEG activo (persiste entre clics de OTROS botones) ----
# Streamlit: st.button() solo devuelve True en el rerun inmediato tras el
# clic. Si el detalle quedara gateado por esa expresión directamente,
# cualquier botón hijo (LIME, SHAP/Grad-CAM, limpiar historial) dispararía
# un rerun donde el botón "Analizar" vuelve a leer False y st.stop() se
# ejecuta antes de llegar a esos hijos — por eso el estado se guarda acá.

def is_detalle_activo() -> bool:
    return bool(st.session_state.get(_DETALLE_KEY, False))


def activar_detalle() -> None:
    st.session_state[_DETALLE_KEY] = True


# ---- Caché genérico de resultados pesados (LIME, SHAP/Grad-CAM) ----
# Mismo problema que el de arriba: sin esto, el resultado de un botón
# desaparece en cuanto se presiona cualquier OTRO botón. Se invalida por
# `epoca` para no mostrar una explicación calculada para una época distinta
# a la que se está viendo ahora.

def get_resultado_cacheado(nombre: str, epoca: int):
    """Devuelve el valor cacheado SOLO si fue calculado para esta `epoca`."""
    entrada = st.session_state.get(f"cache_{nombre}")
    if entrada is None or entrada.get("epoca") != epoca:
        return None
    return entrada.get("valor")


def set_resultado_cacheado(nombre: str, epoca: int, valor) -> None:
    st.session_state[f"cache_{nombre}"] = {"epoca": epoca, "valor": valor}


# ---- Historial --------------------------------------------------------

def registrar_entrada(*, archivo: str, epoca: int, pain_index: Optional[float],
                      nivel_eeg: str, confianza_eeg: float,
                      cuerpo_pred: Optional[str] = None,
                      cerebro_pred: Optional[str] = None) -> None:
    """Agrega una fila al historial de la sesión actual."""
    entrada = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "archivo": archivo,
        "epoca": epoca,
        "pain_index": pain_index,
        "nivel_eeg": nivel_eeg,
        "confianza_eeg": confianza_eeg,
        "cuerpo_pred": cuerpo_pred,
        "cerebro_pred": cerebro_pred,
    }
    st.session_state[_HISTORIAL_KEY].append(entrada)


def get_historial_df() -> pd.DataFrame:
    registros = st.session_state.get(_HISTORIAL_KEY, [])
    if not registros:
        return pd.DataFrame(columns=_COLUMNAS_HISTORIAL)
    return pd.DataFrame(registros)


def limpiar_historial() -> None:
    st.session_state[_HISTORIAL_KEY] = []