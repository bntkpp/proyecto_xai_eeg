"""
frontend/app.py
====================================================================
MONITOR INTERACTIVO (Frontend) — Streamlit
Dashboard médico: nivel de dolor + topomap + explicación clínica +
Pain Index multimodal (cuerpo + cerebro).

Este archivo SOLO orquesta: lee inputs del usuario, llama al backend
(src/backend/*) y pinta resultados con los componentes visuales de
ui_components.py. No contiene lógica de negocio ni de modelos.

Ejecutar (desde la RAÍZ del proyecto):
    streamlit run src/frontend/app.py
====================================================================
"""
import io
import sys
from pathlib import Path

# --------------------------------------------------------------------
# Bootstrap: agrega la RAÍZ del proyecto a sys.path para poder usar
# imports absolutos `from src...` sin instalar el proyecto como paquete
# (streamlit ejecuta este archivo como script suelto, no como parte de
# un paquete, así que los imports relativos no funcionarían aquí).
# --------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mne
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src import config
from src.backend import body_engine, eeg_engine
from src.frontend import resources, ui_components

mne.set_log_level("ERROR")
st.set_page_config(page_title="Monitor de Dolor — XAI EEG", layout="wide")

BIOVID_LABELS = ["BL1 (sin dolor)", "PA1 (leve)", "PA2 (moderado)", "PA3 (fuerte)", "PA4 (extremo)"]

# ====================================================================
# ENCABEZADO
# ====================================================================
st.title("🧠 Monitor de Dolor en Tiempo Real (XAI)")
st.warning(
    "⚠️ Herramienta de **apoyo** a la decisión clínica. No reemplaza el juicio "
    "profesional ni constituye un diagnóstico. Verificar siempre con el equipo médico.",
    icon="⚠️",
)
st.caption("EEGNet + Integrated Gradients + MNE — fusionado con señal corporal (XGBoost + SHAP).")

# ====================================================================
# SIDEBAR: selección de archivo EEG
# ====================================================================
with st.sidebar:
    st.header("Entrada de EEG")
    st.write(f"El modelo espera **{config.N_CHANNELS} electrodos × {config.N_SAMPLES} muestras** "
            f"(1.5 s @ {config.SFREQ} Hz).")
    subido = st.file_uploader("Sube un archivo (.fif / .npy / .pt)", type=["fif", "npy", "pt", "pth"])

    archivo_local = None
    if config.DATA_DIR.is_dir():
        opciones = [f.name for f in config.DATA_DIR.iterdir()
                   if f.suffix in (".fif", ".npy", ".pt", ".pth")]
        if opciones:
            sel = st.selectbox("...o elige de data/", ["(ninguno)"] + sorted(opciones))
            if sel != "(ninguno)":
                archivo_local = str(config.DATA_DIR / sel)

# ====================================================================
# CARGA DEL MODELO EEG (falla duro y visible si no está el .pth)
# ====================================================================
try:
    modelo_eeg = resources.get_eeg_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

fuente = subido if subido is not None else archivo_local
if fuente is None:
    st.info("Sube o selecciona un archivo en la barra lateral.")
    st.stop()

# ====================================================================
# PREPARAR TENSOR + NOMBRES DE CANAL + INFO DE MONTAJE
# ====================================================================
ch_names = config.CH_NAMES
info = None
tensor = None

es_fif = (
    (subido is not None and subido.name.lower().endswith(".fif")) or
    (archivo_local is not None and archivo_local.lower().endswith(".fif"))
)

if es_fif:
    try:
        path = archivo_local or ui_components.guardar_temporal(subido)
        epochs = resources.get_epochs(path)
        n_epocas = len(epochs)
        st.success(f"`.fif` cargado: {n_epocas} épocas · {len(epochs.ch_names)} canales · "
                  f"{epochs.info['sfreq']:.0f} Hz")
        idx = st.slider("Época a analizar", 0, max(n_epocas - 1, 0), 0)
        label_evento = eeg_engine.event_label_para_epoca(epochs, idx) if es_fif else None
        clase_real = body_engine.clase_biovid_desde_evento(label_evento)
        tensor = eeg_engine.epoca_a_tensor(epochs, idx)
        ch_names = epochs.ch_names
        info = eeg_engine.info_con_montaje(epochs)
    except Exception as e:
        st.error(f"No pude leer el .fif: {e}")
        st.stop()
else:
    try:
        if isinstance(fuente, str):
            tensor = (torch.as_tensor(np.load(fuente), dtype=torch.float32)
                     if fuente.endswith(".npy") else torch.load(fuente, map_location="cpu"))
        else:
            data = fuente.read()
            tensor = (torch.as_tensor(np.load(io.BytesIO(data)), dtype=torch.float32)
                     if fuente.name.endswith(".npy") else torch.load(io.BytesIO(data), map_location="cpu"))
        info = resources.get_eeg_info()
        st.warning("Usando CH_NAMES del backend (revisa que el orden coincida con tu tensor).")
    except Exception as e:
        st.error(f"No pude leer el tensor: {e}")
        st.stop()

# ====================================================================
# PAIN INDEX MULTIMODAL (Hito V) — vista clínica principal
# ====================================================================
st.markdown("## 🩺 Pain Index (0–10)")
st.caption("Fusión cuerpo (XGBoost) + cerebro (EEGNet). ⚠️ Emparejamiento **simulado**: "
          "no hay datos multimodales del mismo paciente, así que la señal corporal se elige a mano.")

try:
    fuser = resources.get_fuser()
    predictor_cuerpo = resources.get_body_predictor()
except FileNotFoundError as e:
    st.warning(f"Capa de fusión no disponible: {e}")
else:
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        clase_lbl = st.selectbox("Señal corporal (simulada)", BIOVID_LABELS, index=3)
        cid = BIOVID_LABELS.index(clase_lbl)
        fila_cuerpo = body_engine.fila_representativa(predictor_cuerpo, cid)
        prob_cuerpo = body_engine.predecir_proba_cuerpo(predictor_cuerpo, fila_cuerpo)
    with ctrl2:
        peso_cerebro = st.slider("Peso del cerebro (%)  ·  el resto es cuerpo", 0, 100, 50, step=5,
                                 help="Sube el cerebro si el cuerpo tiene artefactos de movimiento.")
    w_b = peso_cerebro / 100.0
    w_c = 1.0 - w_b

    prob_cerebro = eeg_engine.predecir_proba_cerebro(tensor, modelo_eeg)
    pi, prob_fus = fuser.calcular_pain_index(prob_cuerpo, prob_cerebro, w_cuerpo=w_c, w_cerebro=w_b)

    cuerpo_pred = BIOVID_LABELS[int(np.argmax(prob_cuerpo))]
    cerebro_pred = config.MAPA_DOLOR.get(int(np.argmax(prob_cerebro)), "?")

    gc1, gc2, gc3 = st.columns([1, 2, 1])
    with gc2:
        st.markdown(ui_components.tarjeta_gauge(pi, w_c, w_b, cuerpo_pred, cerebro_pred),
                    unsafe_allow_html=True)

    with st.expander("Ver distribución fusionada y score (detalle)"):
        df_fus = pd.DataFrame({"Probabilidad fusionada": prob_fus},
                              index=["BL1", "PA1", "PA2", "PA3", "PA4"])
        st.bar_chart(df_fus, height=220)
        st.caption(f"Score de fusión (esperanza): "
                  f"{float(np.sum(prob_fus * config.CLASES_BIOVID)):.2f} / 4.0")
        
    if clase_real is not None:
        st.success(f"🔗 Pareo real por evento: época {idx} = **{label_evento}** → {BIOVID_LABELS[clase_real]}")
        cid = clase_real
        fila_cuerpo = body_engine.fila_por_clase(predictor_cuerpo, cid)
    else:
        st.info("Esta época no trae etiqueta de evento — selecciona manualmente (modo simulado).")
        clase_lbl = st.selectbox("Señal corporal (simulada)", BIOVID_LABELS, index=3)
        cid = BIOVID_LABELS.index(clase_lbl)
        fila_cuerpo = body_engine.fila_representativa(predictor_cuerpo, cid)

prob_cuerpo = body_engine.predecir_proba_cuerpo(predictor_cuerpo, fila_cuerpo)

# ====================================================================
# DETALLE CEREBRAL: predicción + XAI + topomap
# ====================================================================
st.markdown("---")
st.markdown("### 🧠 Detalle del cerebro (EEG + XAI)")
if not st.button("Analizar EEG en detalle", type="primary"):
    st.stop()

try:
    resultado = eeg_engine.procesar_onda_eeg(tensor, modelo_eeg, ch_names=ch_names)
except Exception as e:
    st.error(f"Error procesando la onda: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    color = ui_components.COLOR_NIVEL.get(resultado.nivel_dolor, "#455a64")
    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{color};color:white'>"
        f"<div style='font-size:14px;opacity:.85'>Predicción</div>"
        f"<div style='font-size:30px;font-weight:700'>{resultado.nivel_dolor}</div>"
        f"<div style='font-size:14px;opacity:.85'>Confianza del modelo: {resultado.confianza:.0%}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(resultado.confianza, 0.0), 1.0))
    if resultado.explicacion.startswith("⚠️"):
        st.warning(resultado.explicacion)
    else:
        st.success(resultado.explicacion)
    st.metric("Electrodo dominante", ch_names[int(np.argmax(resultado.pesos_electrodo))])

with col2:
    st.pyplot(ui_components.dibujar_topomap(resultado.pesos_electrodo, info), use_container_width=True)

with st.expander("Ver los 63 pesos por electrodo"):
    orden = np.argsort(resultado.pesos_electrodo)[::-1]
    st.dataframe(
        {"Electrodo": [ch_names[i] for i in orden],
         "Peso (0-1)": [round(float(resultado.pesos_electrodo[i]), 3) for i in orden]},
        use_container_width=True, height=300,
    )
