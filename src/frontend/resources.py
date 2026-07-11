"""
frontend/resources.py
====================================================================
Carga de recursos pesados (modelo EEG, modelo corporal, calibrador de
fusión) con cacheo de Streamlit (`@st.cache_resource`).

Es la ÚNICA capa que conoce Streamlit dentro de la carga de recursos:
el backend (src/backend/*) no sabe que existe una interfaz gráfica,
lo que permite probarlo y reusarlo fuera de Streamlit sin cambios.
====================================================================
"""
import streamlit as st

from src.backend import body_engine, eeg_engine, fusion_engine


@st.cache_resource(show_spinner="Cargando modelo EEG...")
def get_eeg_model():
    return eeg_engine.load_model()


@st.cache_resource(show_spinner=False)
def get_eeg_info():
    return eeg_engine.build_info()


@st.cache_resource(show_spinner="Cargando modelo corporal...")
def get_body_predictor():
    return body_engine.cargar_predictor_corporal()


@st.cache_resource(show_spinner=False)
def get_fuser():
    return fusion_engine.PainIndexFuser()


@st.cache_resource(show_spinner="Leyendo archivo EEG...")
def get_epochs(path: str):
    return eeg_engine.leer_fif(path)


@st.cache_resource(show_spinner="Preparando explicador LIME...")
def get_lime_explainer():
    """Se construye UNA sola vez sobre el dataset de fondo completo —
    reconstruirlo en cada rerun sería lento e innecesario."""
    predictor = get_body_predictor()
    return body_engine.crear_explicador_lime(predictor)
