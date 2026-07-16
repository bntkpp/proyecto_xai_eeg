"""
frontend/app.py
====================================================================
MONITOR INTERACTIVO (Frontend) — Streamlit (Orquestador Principal)
====================================================================
"""
import io
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

import mne
import numpy as np
import streamlit as st
import torch

from src import config
from src.backend import body_engine, eeg_engine
from src.frontend import resources, session, ui_components
from src.frontend.views import tab_resumen, tab_cerebro, tab_cuerpo, tab_historial, tab_datasets

mne.set_log_level("ERROR")
st.set_page_config(page_title="Monitor de Dolor — XAI", layout="wide")
session.init_session_state()

if "anotaciones" not in st.session_state: st.session_state.anotaciones = {}
if "last_alert_epoch" not in st.session_state: st.session_state.last_alert_epoch = -1

st.markdown("""<style>button[kind="primary"] { background-color: #28a745 !important; color: white !important; border-color: #28a745 !important; }</style>""", unsafe_allow_html=True)

# ====================================================================
# SIDEBAR
# ====================================================================
with st.sidebar:
    if config.LOGO_PATH.exists():
        st.image(str(config.LOGO_PATH), use_container_width=True)
        st.markdown("---")
        
    st.header("Entrada de EEG")
    subido = st.file_uploader("Sube un archivo", type=["fif", "npy", "pt", "pth"])
    archivo_local = None
    if config.EEG_DATA_DIR.is_dir():
        opciones = [f.name for f in config.EEG_DATA_DIR.iterdir() if f.suffix in (".fif", ".npy", ".pt", ".pth")]
        if opciones:
            sel = st.selectbox("o elige de data/eeg/", ["(ninguno)"] + sorted(opciones))
            if sel != "(ninguno)": archivo_local = str(config.EEG_DATA_DIR / sel)

    st.markdown("---")
    st.header("🏥 Datos del Paciente")
    is_running = session.is_playing()
    if is_running: st.info("⏸️ Detenga la reproducción para editar.")

    paciente_nombre = st.text_input("Nombre", disabled=is_running)
    c1, c2 = st.columns(2)
    with c1: paciente_edad = st.number_input("Edad", min_value=0, value=0, disabled=is_running)
    with c2: paciente_peso = st.number_input("Peso", min_value=0.0, value=0.0, disabled=is_running)
    paciente_sexo = st.selectbox("Sexo", ["Seleccione", "Masculino", "Femenino", "Otro"], disabled=is_running)
    
    datos_completos = bool(paciente_nombre.strip() and paciente_edad > 0 and paciente_peso > 0.0 and paciente_sexo != "Seleccione")
    if not datos_completos and not is_running:
        st.warning("⚠️ Complete Nombre, Edad, Peso y Sexo.")

    st.markdown("---")
    umbral_alerta = st.slider("Umbral de alerta", 0.0, 10.0, session.get_umbral_alerta(), step=0.5)
    session.set_umbral_alerta(umbral_alerta)

st.title("🧠 Monitor de Dolor en Tiempo Real (XAI)")

try:
    modelo_eeg = resources.get_eeg_model()
except Exception as e:
    st.error(str(e)); st.stop()

fuente = subido if subido is not None else archivo_local
if fuente is None: st.info("👋 Sube o selecciona un archivo EEG."); st.stop()
if not datos_completos: st.error("🔒 **Datos incompletos.** Rellene la barra lateral."); st.stop()

nombre_archivo = archivo_local or subido.name

# ====================================================================
# PREPARAR TENSOR
# ====================================================================
ch_names = getattr(config, 'CH_NAMES', [f"CH{i}" for i in range(config.N_CHANNELS)])
info, tensor, epochs, n_epocas, idx = None, None, None, 0, session.get_epoca_actual()
es_fif = isinstance(fuente, str) and fuente.lower().endswith(".fif") or (subido and subido.name.lower().endswith(".fif"))

if es_fif:
    path = archivo_local or ui_components.guardar_temporal(subido)
    epochs = resources.get_epochs(path)
    n_epocas = len(epochs)
    if idx >= n_epocas: session.set_epoca_actual(0)
        
    with st.sidebar:
        st.markdown("---")
        st.header("🕹️ Controles de Reproducción")
        if session.is_playing(): st.markdown("""<div style="color:#ff6b6b; font-weight:bold; text-align:center;">● MONITOREO ACTIVO</div>""", unsafe_allow_html=True)

        cp1, cp2 = st.columns(2)
        with cp1:
            if st.button("▶ Reproducir", use_container_width=True, disabled=session.is_playing()): session.set_playing(True); st.rerun()
        with cp2:
            if st.button("⏸ Pausar", use_container_width=True, disabled=not session.is_playing()): session.set_playing(False); st.rerun()
        
        intervalo_seg = st.slider("Segundos entre épocas", 0.5, 5.0, 2.0, step=0.5)
        st.markdown(f"<div style='text-align: center;'><b>Época en análisis:</b> {idx} / {max(n_epocas - 1, 0)}</div>", unsafe_allow_html=True)

    tensor = eeg_engine.epoca_a_tensor(epochs, idx)
    ch_names = epochs.ch_names
    info = eeg_engine.info_con_montaje(epochs)
else:
    session.set_playing(False)
    data = fuente.read() if not isinstance(fuente, str) else fuente
    tensor = torch.as_tensor(np.load(io.BytesIO(data)) if str(fuente).endswith(".npy") else torch.load(io.BytesIO(data) if isinstance(data, bytes) else data, map_location="cpu"), dtype=torch.float32)
    info = resources.get_eeg_info()

# ====================================================================
# CÁLCULOS BASE
# ====================================================================
resultado_eeg = eeg_engine.procesar_onda_eeg(tensor, modelo_eeg, ch_names=ch_names)
prob_cerebro = eeg_engine.predecir_proba_cerebro(tensor, modelo_eeg)
cerebro_pred = config.MAPA_DOLOR.get(int(np.argmax(prob_cerebro)), "?")

has_fusion = False
fuser, predictor_cuerpo, clase_real, label_evento = None, None, None, None

try:
    fuser = resources.get_fuser()
    predictor_cuerpo = resources.get_body_predictor()
    has_fusion = True
    label_evento = eeg_engine.event_label_para_epoca(epochs, idx) if es_fif else None
    clase_real = body_engine.clase_biovid_desde_evento(label_evento)
except Exception: pass

# ====================================================================
# RENDERIZADO DE PESTAÑAS MEDIANTE VISTAS
# ====================================================================
t1, t2, t3, t4, t5 = st.tabs(["📊 Resumen (clínico)", "🧠 Cerebro (XAI)", "🫀 Cuerpo", "📈 Historial y Reportes", "📁 Datasets"])

with t1: pi, prob_fus, cuerpo_pred, resultado_cuerpo = tab_resumen.render(idx, has_fusion, clase_real, label_evento, predictor_cuerpo, fuser, prob_cerebro, cerebro_pred, resultado_eeg, ch_names)
with t2: tab_cerebro.render(idx, tensor, modelo_eeg, resultado_eeg, ch_names, info)
with t3: tab_cuerpo.render(idx, has_fusion, resultado_cuerpo, prob_fus, predictor_cuerpo)
with t4: tab_historial.render(idx, es_fif, pi, resultado_eeg, cuerpo_pred, cerebro_pred, paciente_nombre, paciente_sexo, paciente_edad, nombre_archivo)
with t5: tab_datasets.render(info, n_epocas, es_fif, has_fusion, predictor_cuerpo)

if es_fif and session.is_playing() and n_epocas > 0:
    time.sleep(intervalo_seg)
    session.set_epoca_actual((idx + 1) % n_epocas)
    st.rerun()
