"""
frontend/resources.py
====================================================================
Carga de recursos pesados con cacheo de Streamlit (`@st.cache_resource`).
====================================================================
"""
import streamlit as st
from src import config
from src.backend import body_engine, eeg_engine, fusion_engine

@st.cache_resource(show_spinner="Cargando modelo EEG...")
def get_eeg_model():
    # Inyectamos la nueva ruta desde config
    return eeg_engine.load_model(config.EEG_MODEL_PATH)

@st.cache_resource(show_spinner=False)
def get_eeg_info():
    return eeg_engine.build_info()

@st.cache_resource(show_spinner="Cargando modelo corporal...")
def get_body_predictor():
    # Inyectamos la nueva ruta desde config
    return body_engine.cargar_predictor_corporal(config.AUTONOMIC_MODEL_PATH)

@st.cache_resource(show_spinner=False)
def get_fuser():
    return fusion_engine.PainIndexFuser()

@st.cache_resource(show_spinner="Leyendo archivo EEG...")
def get_epochs(path: str):
    return eeg_engine.leer_fif(path)

@st.cache_resource(show_spinner="Preparando explicador LIME...")
def get_lime_explainer():
    predictor = get_body_predictor()
    return body_engine.crear_explicador_lime(predictor)
