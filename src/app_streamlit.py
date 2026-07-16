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
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import mne

from inference_backend import (
    load_model, procesar_onda_eeg, predecir_proba_cerebro, build_info,
    leer_fif, epoca_a_tensor, info_con_montaje,
    CH_NAMES, N_CHANNELS, N_SAMPLES, WEIGHTS_PATH, DATA_DIR, MAPA_DOLOR,
)
from fusion_layer import PainIndexFuser, CLASES_BIOVID

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


# --- Capa de fusión multimodal (Hito V) ---
BIOVID_LABELS = ["BL1 (sin dolor)", "PA1 (leve)", "PA2 (moderado)",
                 "PA3 (fuerte)", "PA4 (extremo)"]
_BODY_META = ["subject_id", "subject_name", "class_id", "class_name",
              "sample_id", "sample_name"]


@st.cache_resource
def get_fuser():
    return PainIndexFuser()


@st.cache_resource
def get_body_model():
    """Carga el XGBoost corporal + dataset (sin SHAP) para la fusión.
    Devuelve (modelo, df, features, {clase: indice_representativo})."""
    import xgboost as xgb
    modelo = xgb.XGBClassifier()
    modelo.load_model(os.path.join(DATA_DIR, "xgboost_cuerpo.json"))
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset_elite_biovid.csv"))
    feats = [c for c in df.columns if c not in _BODY_META]
    idx_por_clase = {int(c): int(df.index[df["class_id"] == c][0]) for c in range(5)}
    return modelo, df, feats, idx_por_clase


def color_pain_index(pi):
    if pi < 3:
        return "#2e7d32"
    if pi < 5:
        return "#f9a825"
    if pi < 7:
        return "#ef6c00"
    return "#c62828"


def veredicto_pain_index(pi):
    """Palabra clínica + color por severidad (lectura de un vistazo)."""
    if pi < 3:
        return "SIN DOLOR", "#37b24d"
    if pi < 5:
        return "DOLOR LEVE", "#f1a208"
    if pi < 7:
        return "DOLOR MODERADO", "#f76707"
    return "DOLOR SEVERO", "#e03131"


# Zonas del medidor (geometría fija: centro 150,150 · radio 118 · semicírculo).
_GAUGE_ZONAS = [
    (0, 3, "#37b24d", "M 32.0 150.0 A 118 118 0 0 1 80.6 54.5"),
    (3, 5, "#f1a208", "M 80.6 54.5 A 118 118 0 0 1 150.0 32.0"),
    (5, 7, "#f76707", "M 150.0 32.0 A 118 118 0 0 1 219.4 54.5"),
    (7, 10, "#e03131", "M 219.4 54.5 A 118 118 0 0 1 268.0 150.0"),
]


def tarjeta_gauge(pi, w_c, w_b, cuerpo_pred, cerebro_pred):
    """Medidor tipo monitor clínico: aguja + zona encendida + veredicto."""
    palabra, col = veredicto_pain_index(pi)
    pi_c = min(max(pi, 0.0), 10.0)
    th = np.radians(180 - 18 * pi_c)                 # ángulo de la aguja
    nx, ny = 150 + 106 * np.cos(th), 150 - 106 * np.sin(th)
    arcs = "".join(
        f'<path d="{d}" stroke="{c}" stroke-width="22" fill="none" '
        f'opacity="{1 if (a <= pi_c < b or (b == 10 and pi_c >= 10)) else 0.28}"/>'
        for a, b, c, d in _GAUGE_ZONAS)
    return f"""<div style="border:2px solid {col};border-radius:16px;padding:14px 12px 12px;
        text-align:center;background:rgba(128,128,128,0.06)">
      <div style="font-size:12px;color:#8a8a8a;letter-spacing:.5px">
        FUSIÓN {round(w_c*100)}% CUERPO · {round(w_b*100)}% CEREBRO</div>
      <svg viewBox="0 0 300 162" width="100%" style="display:block;margin:2px auto 0;max-width:340px">
        {arcs}
        <line x1="150" y1="150" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{col}"
              stroke-width="6" stroke-linecap="round"/>
        <circle cx="150" cy="150" r="9" fill="{col}"/>
        <text x="28" y="160" font-size="12" fill="#8a8a8a">0</text>
        <text x="272" y="160" font-size="12" fill="#8a8a8a" text-anchor="end">10</text>
      </svg>
      <div style="font-size:46px;font-weight:800;color:{col};line-height:1.05;margin-top:4px">
        {pi}<span style="font-size:18px;color:#8a8a8a;font-weight:600"> / 10</span></div>
      <div style="font-size:30px;font-weight:800;color:{col};letter-spacing:1px;margin-top:2px">{palabra}</div>
      <div style="font-size:12px;color:#8a8a8a;margin-top:6px">
        🫀 Cuerpo: {cuerpo_pred} · 🧠 Cerebro: {cerebro_pred}</div>
    </div>"""


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

# ====================================================================
# PAIN INDEX MULTIMODAL (Hito V) — la vista clínica principal
# ====================================================================
st.markdown("## 🩺 Pain Index (0–10)")
st.caption("Fusión cuerpo (XGBoost) + cerebro (EEGNet). ⚠️ Emparejamiento **simulado**: "
           "no hay datos multimodales del mismo paciente, así que la señal corporal se elige a mano.")

try:
    fuser = get_fuser()
    body_model, body_df, body_feats, body_idx_por_clase = get_body_model()
except Exception as e:
    st.warning(f"Capa de fusión no disponible: {e}. "
               f"¿Corriste `python src/fusion_layer.py` para crear el calibrador?")
else:
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        clase_lbl = st.selectbox("Señal corporal (simulada)", BIOVID_LABELS, index=3)
        cid = BIOVID_LABELS.index(clase_lbl)
        fila_cuerpo = body_df[body_feats].iloc[[body_idx_por_clase[cid]]]
        prob_cuerpo = body_model.predict_proba(fila_cuerpo)[0]
    with ctrl2:
        peso_cerebro = st.slider("Peso del cerebro (%)  ·  el resto es cuerpo", 0, 100, 50, step=5,
                                 help="Modificabilidad Hito V. Estado inicial 50/50. "
                                      "Ej.: sube el cerebro si el cuerpo tiene artefactos de movimiento.")
    w_b = peso_cerebro / 100.0
    w_c = 1.0 - w_b

    prob_cerebro = predecir_proba_cerebro(tensor, modelo)          # (4,) softmax NRS
    pi, prob_fus = fuser.calcular_pain_index(prob_cuerpo, prob_cerebro,
                                             w_cuerpo=w_c, w_cerebro=w_b)

    cuerpo_pred = BIOVID_LABELS[int(np.argmax(prob_cuerpo))]
    cerebro_pred = MAPA_DOLOR.get(int(np.argmax(prob_cerebro)), "?")

    # Medidor clínico como vista principal (lectura de un vistazo).
    gc1, gc2, gc3 = st.columns([1, 2, 1])
    with gc2:
        st.markdown(tarjeta_gauge(pi, w_c, w_b, cuerpo_pred, cerebro_pred),
                    unsafe_allow_html=True)

    # El detalle numérico queda a un clic, para quien lo quiera.
    with st.expander("Ver distribución fusionada y score (detalle)"):
        df_fus = pd.DataFrame(
            {"Probabilidad fusionada": prob_fus},
            index=["BL1", "PA1", "PA2", "PA3", "PA4"])
        st.bar_chart(df_fus, height=220)
        st.caption(f"Score de fusión (esperanza): {float(np.sum(prob_fus*CLASES_BIOVID)):.2f} / 4.0")

st.markdown("---")
st.markdown("### 🧠 Detalle del cerebro (EEG + XAI)")
if not st.button("Analizar EEG en detalle", type="primary"):
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
