"""
app_streamlit.py
====================================================================
MONITOR INTERACTIVO (Frontend) — Hito 4 / Actividad 3
Dashboard médico: nivel de dolor + topomap + explicación clínica.

Ejecutar (desde la raíz del proyecto):
    streamlit run src/app_streamlit.py
====================================================================
"""
import os
import io
import tempfile

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import mne

from inference_backend import (
    load_model, procesar_onda_eeg, build_info,
    leer_fif, epoca_a_tensor, info_con_montaje,
    CH_NAMES, N_CHANNELS, N_SAMPLES, WEIGHTS_PATH, DATA_DIR,
)

mne.set_log_level("ERROR")
st.set_page_config(page_title="Monitor de Dolor — XAI EEG", layout="wide")

COLOR_NIVEL = {
    "Sin Dolor (NRS 0-2)":    "#2e7d32",
    "Dolor Leve (NRS 4)":     "#f9a825",
    "Dolor Moderado (NRS 6)": "#ef6c00",
    "Dolor Severo (NRS 8)":   "#c62828",
}


# --- El modelo y el info se cargan UNA SOLA VEZ (nota de arquitectura) ---
@st.cache_resource
def get_model():
    return load_model(WEIGHTS_PATH)


@st.cache_resource
def get_info_tensor():
    return build_info()


@st.cache_resource(show_spinner=False)
def cargar_epochs(path):
    return leer_fif(path)


def ensure_local_path(fuente):
    """Devuelve una ruta en disco. Si `fuente` es un archivo subido, lo guarda en temp."""
    if isinstance(fuente, str):
        return fuente
    destino = os.path.join(tempfile.gettempdir(), fuente.name)
    with open(destino, "wb") as f:
        f.write(fuente.read())
    return destino


def cargar_tensor_suelto(fuente):
    """Carga .npy / .pt (ruta o archivo subido)."""
    if isinstance(fuente, str):
        if fuente.endswith(".npy"):
            return torch.as_tensor(np.load(fuente), dtype=torch.float32)
        return torch.load(fuente, map_location="cpu")
    data = fuente.read()
    if fuente.name.endswith(".npy"):
        return torch.as_tensor(np.load(io.BytesIO(data)), dtype=torch.float32)
    return torch.load(io.BytesIO(data), map_location="cpu")


# Canales que NO van en el cuero cabelludo (EOG/EMG/ECG): el modelo los usa,
# pero no tienen posición 10-20 y rompen el topomap. Se excluyen SOLO al graficar.
NO_CUERO_CABELLUDO = {"VEO", "HEO", "HEOR", "HEOL", "VEOU", "VEOL", "EOG",
                      "EKG", "ECG", "EMG", "M1", "M2"}


def _solo_eeg_para_topomap(pesos_63, info):
    """Devuelve (pesos, info) quedándose solo con electrodos del cuero cabelludo."""
    tipos = info.get_channel_types()
    keep = [i for i, (t, n) in enumerate(zip(tipos, info["ch_names"]))
            if t == "eeg" and n not in NO_CUERO_CABELLUDO]
    if len(keep) < len(info["ch_names"]):
        info = mne.pick_info(info, keep)
        pesos_63 = np.asarray(pesos_63)[keep]
    return pesos_63, info


def dibujar_topomap(pesos_63, info):
    pesos_63, info = _solo_eeg_para_topomap(pesos_63, info)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im, _ = mne.viz.plot_topomap(pesos_63, info, axes=ax, show=False,
                                 cmap="hot", contours=4, sensors=True)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Importancia (0-1)")
    ax.set_title("¿Dónde procesa el dolor?", fontsize=11)
    return fig


def nombre_extension(fuente):
    nombre = fuente if isinstance(fuente, str) else fuente.name
    return nombre.lower()


# ====================================================================
# INTERFAZ
# ====================================================================
st.title("🧠 Monitor de Dolor en Tiempo Real (XAI)")
st.caption("EEGNet + Integrated Gradients + MNE — apoyo a la decisión clínica, no la reemplaza.")

with st.sidebar:
    st.header("Entrada de EEG")
    st.write(f"El modelo espera **{N_CHANNELS} electrodos × {N_SAMPLES} muestras** (1.5 s @ 250 Hz).")
    subido = st.file_uploader("Sube un archivo (.fif / .npy / .pt)", type=["fif", "npy", "pt", "pth"])

    archivo_local = None
    if os.path.isdir(DATA_DIR):
        opciones = [f for f in os.listdir(DATA_DIR) if f.endswith((".fif", ".npy", ".pt", ".pth"))]
        if opciones:
            sel = st.selectbox("...o elige de data/", ["(ninguno)"] + sorted(opciones))
            if sel != "(ninguno)":
                archivo_local = os.path.join(DATA_DIR, sel)

# Modelo (cacheado)
try:
    modelo = get_model()
except FileNotFoundError:
    st.error(f"No encontré el modelo en `{WEIGHTS_PATH}`. Coloca `eegnet_fold1.pth` en la raíz del proyecto.")
    st.stop()

fuente = subido if subido is not None else archivo_local
if fuente is None:
    st.info("Sube o selecciona un archivo en la barra lateral.")
    st.stop()

ext = nombre_extension(fuente)

# --- Preparar tensor + nombres + info según el tipo de archivo ---
ch_names = CH_NAMES
info = None
tensor = None

if ext.endswith(".fif"):
    try:
        path = ensure_local_path(fuente)
        epochs = cargar_epochs(path)
        n_epocas = len(epochs)
        st.success(f"`.fif` cargado: {n_epocas} épocas · {len(epochs.ch_names)} canales · {epochs.info['sfreq']:.0f} Hz")
        idx = st.slider("Época a analizar", 0, max(n_epocas - 1, 0), 0)
        tensor = epoca_a_tensor(epochs, idx)
        ch_names = epochs.ch_names
        info = info_con_montaje(epochs)
    except Exception as e:
        st.error(f"No pude leer el .fif: {e}")
        st.stop()
else:
    try:
        tensor = cargar_tensor_suelto(fuente)
        info = get_info_tensor()
        ch_names = CH_NAMES
        st.warning("Usando CH_NAMES del backend (revisa que el orden coincida con tu tensor).")
    except Exception as e:
        st.error(f"No pude leer el tensor: {e}")
        st.stop()

if not st.button("Analizar", type="primary"):
    st.stop()

# --- Inferencia + XAI ---
try:
    nivel, pesos, texto, conf = procesar_onda_eeg(tensor, modelo, ch_names=ch_names)
except Exception as e:
    st.error(f"Error procesando la onda: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    color = COLOR_NIVEL.get(nivel, "#455a64")
    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{color};color:white'>"
        f"<div style='font-size:14px;opacity:.85'>Predicción</div>"
        f"<div style='font-size:30px;font-weight:700'>{nivel}</div>"
        f"<div style='font-size:14px;opacity:.85'>Confianza del modelo: {conf:.0%}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(conf, 0.0), 1.0))
    if texto.startswith("⚠️"):
        st.warning(texto)
    else:
        st.success(texto)
    st.metric("Electrodo dominante", ch_names[int(np.argmax(pesos))])

with col2:
    st.pyplot(dibujar_topomap(pesos, info), use_container_width=True)

with st.expander("Ver los 63 pesos por electrodo"):
    orden = np.argsort(pesos)[::-1]
    st.dataframe(
        {"Electrodo": [ch_names[i] for i in orden],
         "Peso (0-1)": [round(float(pesos[i]), 3) for i in orden]},
        use_container_width=True, height=300,
    )
