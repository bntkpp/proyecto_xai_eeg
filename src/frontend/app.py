"""
frontend/app.py
====================================================================
MONITOR INTERACTIVO (Frontend) — Streamlit
Dashboard médico: nivel de dolor + topomap + explicación clínica +
Pain Index multimodal (cuerpo + cerebro) + historial de sesión con
reproducción automática de épocas.

Este archivo SOLO orquesta: lee inputs del usuario, llama al backend
(src/backend/*) y pinta resultados con los componentes visuales de
ui_components.py. No contiene lógica de negocio ni de modelos.

Ejecutar (desde la RAÍZ del proyecto):
    streamlit run src/frontend/app.py
====================================================================
"""
import io
import sys
import time
from datetime import datetime
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
from src.backend import body_engine, coherencia_engine, eeg_engine, narrativa_engine, reportes_engine
from src.frontend import resources, session, ui_components

mne.set_log_level("ERROR")
st.set_page_config(page_title="Monitor de Dolor — XAI EEG", layout="wide")
session.init_session_state()

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

    st.markdown("---")
    st.header("🚨 Alertas")
    umbral_alerta = st.slider(
        "Umbral de Pain Index para alerta", 0.0, 10.0,
        session.get_umbral_alerta(), step=0.5,
        help="Cuando el Pain Index calculado alcance o supere este valor, "
             "el dashboard mostrará una alerta de intervención analgésica.",
    )
    session.set_umbral_alerta(umbral_alerta)

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

nombre_archivo = archivo_local or subido.name

# ====================================================================
# PREPARAR TENSOR + NOMBRES DE CANAL + INFO DE MONTAJE
# ====================================================================
ch_names = config.CH_NAMES
info = None
tensor = None
epochs = None
n_epocas = 0
idx = 0

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

        # ------------------------------------------------------------
        # Controles de época: reproducción automática vs. manual.
        # `session.epoca_actual` es la ÚNICA fuente de verdad. El widget
        # del slider se sincroniza con ella SIEMPRE ANTES de instanciarse
        # (Streamlit prohíbe escribir session_state[key] de un widget
        # DESPUÉS de haberlo creado en el mismo run — por eso el tick
        # automático, al final del script, solo toca `session`, nunca
        # la clave del widget directamente).
        # ------------------------------------------------------------
        if session.get_epoca_actual() >= n_epocas:
            session.set_epoca_actual(0)
        if st.session_state.get("slider_epoca_widget") != session.get_epoca_actual():
            st.session_state["slider_epoca_widget"] = session.get_epoca_actual()

        c_play, c_pause, c_speed = st.columns([1, 1, 2])
        with c_play:
            if st.button("▶ Reproducir", use_container_width=True, disabled=session.is_playing()):
                session.set_playing(True)
                st.rerun()
        with c_pause:
            if st.button("⏸ Pausar", use_container_width=True, disabled=not session.is_playing()):
                session.set_playing(False)
                st.rerun()
        with c_speed:
            intervalo_seg = st.slider("Segundos entre épocas", 0.5, 5.0, 2.0, step=0.5)

        if session.is_playing():
            st.info(f"▶ Reproduciendo automáticamente — avanza cada {intervalo_seg:.1f}s. "
                    f"Mueve el slider para volver a modo manual.")

        idx = st.slider("Época a analizar", 0, max(n_epocas - 1, 0), key="slider_epoca_widget")

        if idx != session.get_epoca_actual():
            # El usuario arrastró el slider a mano -> pausa el modo automático.
            # (Un avance por autoplay nunca llega aquí: la sincronización de
            # arriba ya deja `idx == session.get_epoca_actual()` antes de crear
            # el widget, así que esta rama solo se dispara por interacción real.)
            session.set_epoca_actual(idx)
            session.set_playing(False)

        tensor = eeg_engine.epoca_a_tensor(epochs, idx)
        ch_names = epochs.ch_names
        info = eeg_engine.info_con_montaje(epochs)
    except Exception as e:
        st.error(f"No pude leer el .fif: {e}")
        st.stop()
else:
    session.set_playing(False)  # el modo automático solo aplica a .fif con varias épocas
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

pi = None
cuerpo_pred = None
cerebro_pred = None
resultado_cuerpo = None

try:
    fuser = resources.get_fuser()
    predictor_cuerpo = resources.get_body_predictor()
except FileNotFoundError as e:
    st.warning(f"Capa de fusión no disponible: {e}")
else:
    label_evento = eeg_engine.event_label_para_epoca(epochs, idx) if es_fif else None
    clase_real = body_engine.clase_biovid_desde_evento(label_evento)

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        if clase_real is not None:
            cid = clase_real
            st.success(f"🔗 Pareo real por evento: época {idx} = **{label_evento}** "
                      f"→ {BIOVID_LABELS[cid]}")
            # `fila_por_clase` elige una fila AL AZAR dentro de la clase — sin
            # cachearla por época, cada rerun (mover cualquier otro control,
            # incluso uno que no toca la época) sortearía una fila distinta y
            # el Pain Index "saltaría" para la misma época (bug reportado).
            fila_cuerpo = session.get_resultado_cacheado("fila_cuerpo_real", idx)
            if fila_cuerpo is None:
                fila_cuerpo = body_engine.fila_por_clase(predictor_cuerpo, cid)
                session.set_resultado_cacheado("fila_cuerpo_real", idx, fila_cuerpo)
        else:
            if es_fif:
                st.info("Esta época no trae etiqueta de evento reconocible — "
                        "selección manual (modo simulado).")
            clase_lbl = st.selectbox("Señal corporal (simulada)", BIOVID_LABELS, index=3)
            cid = BIOVID_LABELS.index(clase_lbl)
            fila_cuerpo = body_engine.fila_representativa(predictor_cuerpo, cid)
        resultado_cuerpo = body_engine.procesar_datos_cuerpo(predictor_cuerpo, fila_cuerpo)
        prob_cuerpo = resultado_cuerpo.probabilidades
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

    if pi >= session.get_umbral_alerta():
        st.error(f"🚨 **Alerta de intervención analgésica** — Pain Index actual **{pi:.1f}** "
                f"alcanza o supera el umbral configurado (**{session.get_umbral_alerta():.1f}**). "
                f"Ajustable en la barra lateral.")

    with st.expander("Ver distribución fusionada y score (detalle)"):
        df_fus = pd.DataFrame({"Probabilidad fusionada": prob_fus},
                              index=["BL1", "PA1", "PA2", "PA3", "PA4"])
        st.bar_chart(df_fus, height=220)
        st.caption(f"Score de fusión (esperanza): "
                  f"{float(np.sum(prob_fus * config.CLASES_BIOVID)):.2f} / 4.0")

    # ----------------------------------------------------------------
    # Explicabilidad de la señal CORPORAL: SHAP (automático, rápido)
    # + LIME (bajo demanda, más lento — perturba ~500 muestras).
    # ----------------------------------------------------------------
    st.markdown("#### 🔍 ¿Por qué esta predicción corporal?")
    exp_shap, exp_lime = st.columns(2)
    with exp_shap:
        st.caption("**SHAP** (TreeExplainer — exacto, automático)")
        df_shap = pd.DataFrame(resultado_cuerpo.top_drivers).set_index("biomarcador")
        st.bar_chart(df_shap, height=200)
    with exp_lime:
        st.caption("**LIME** (aproximación local — bajo demanda)")
        if st.button("Calcular explicación LIME", disabled=session.is_playing(),
                    help=("Deshabilitado en modo reproducción automática por su costo."
                         if session.is_playing() else None)):
            with st.spinner("Perturbando muestras alrededor del paciente..."):
                lime_explainer = resources.get_lime_explainer()
                top_lime = body_engine.explicar_con_lime(
                    predictor_cuerpo, lime_explainer, fila_cuerpo, resultado_cuerpo.clase)
            session.set_resultado_cacheado("lime_cuerpo", idx, top_lime)

        top_lime_cache = session.get_resultado_cacheado("lime_cuerpo", idx)
        if top_lime_cache is not None:
            df_lime = pd.DataFrame(top_lime_cache).set_index("biomarcador")
            st.bar_chart(df_lime, height=200)
        else:
            st.caption("Presiona el botón para calcularla (no se ejecuta automáticamente).")

# ====================================================================
# DETALLE CEREBRAL: predicción + XAI + topomap
# ====================================================================
st.markdown("---")
st.markdown("### 🧠 Detalle del cerebro (EEG + XAI)")

modo_automatico = session.is_playing()
if modo_automatico:
    session.activar_detalle()
    st.caption("Modo automático activo: se analiza cada época sin necesidad de presionar el botón.")
else:
    if st.button("Analizar EEG en detalle", type="primary"):
        session.activar_detalle()

if not session.is_detalle_activo():
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

# ----------------------------------------------------------------
# SHAP-EEG (GradientShap) + Grad-CAM — bajo demanda, como LIME: son
# 3 pasadas backward adicionales, no conviene correrlas en cada tick
# del modo reproducción automática.
# ----------------------------------------------------------------
st.markdown("#### 🧬 SHAP + Grad-CAM (EEG)")
st.caption("Complementa a Integrated Gradients con dos métodos adicionales: "
          "SHAP (GradientShap) por electrodo, y Grad-CAM canal × tiempo "
          "(capa `temporal_conv`, antes de que la capa espacial colapse los electrodos).")

if st.button("Calcular SHAP + Grad-CAM (EEG)", disabled=session.is_playing(),
            help=("Deshabilitado en modo reproducción automática por su costo."
                 if session.is_playing() else None)):
    with st.spinner("Calculando GradientShap y Grad-CAM..."):
        xai_ext = eeg_engine.calcular_xai_extendido(tensor, modelo_eeg)
    session.set_resultado_cacheado("xai_ext_eeg", idx, xai_ext)

xai_ext = session.get_resultado_cacheado("xai_ext_eeg", idx)
if xai_ext is not None:
    if xai_ext.clase != resultado.clase:
        st.warning("⚠️ La clase usada para SHAP/Grad-CAM no coincide con la de Integrated "
                  "Gradients de arriba — vuelve a analizar la época antes de comparar.")

    comp1, comp2 = st.columns(2)
    with comp1:
        st.caption("**Integrated Gradients vs SHAP** por electrodo (top de cada uno)")
        df_comp = pd.DataFrame({
            "Integrated Gradients": resultado.pesos_electrodo,
            "SHAP (GradientShap)": xai_ext.shap_por_canal,
        }, index=ch_names)
        top_ig = set(df_comp["Integrated Gradients"].nlargest(10).index)
        top_shap = set(df_comp["SHAP (GradientShap)"].nlargest(10).index)
        electrodos_top = [ch for ch in ch_names if ch in (top_ig | top_shap)]
        st.bar_chart(df_comp.loc[electrodos_top], height=280)
        coinciden = len(top_ig & top_shap)
        st.caption(f"Coinciden {coinciden}/10 electrodos entre el top-10 de ambos métodos "
                  f"— más coincidencia = explicación más robusta, no artefacto de un solo método.")
    with comp2:
        st.caption("**Ventana temporal crítica** (Grad-CAM sobre `separable_conv`)")
        eje_ms = (np.arange(len(xai_ext.gradcam_ventana_temporal)) / config.SFREQ * 1000).round().astype(int)
        df_tiempo = pd.DataFrame({"Importancia": xai_ext.gradcam_ventana_temporal}, index=eje_ms)
        st.line_chart(df_tiempo, height=280)
        st.caption("Eje X en milisegundos desde el inicio de la época.")

    st.pyplot(ui_components.dibujar_gradcam_heatmap(xai_ext.gradcam_canal_tiempo, ch_names),
              use_container_width=True)
else:
    st.caption("Presiona el botón para calcularlo (no se ejecuta automáticamente).")

# ----------------------------------------------------------------
# LIME-EEG (oclusión) — más barato que SHAP/Grad-CAM (solo forward,
# sin backward), pero se deja igual bajo botón para mantener el
# patrón consistente con las otras explicaciones "bajo demanda".
# ----------------------------------------------------------------
st.markdown("#### 🧩 LIME-EEG (aproximación por oclusión)")
st.caption("LIME clásico (tabular) no aplica a una señal continua. Esta es la adaptación "
          "estándar: ocluye un electrodo o ventana de tiempo a la vez y mide cuánto cae la "
          "probabilidad de la clase predicha. Es perturbación real sobre el modelo (sin "
          "gradientes) — tercera validación cruzada, independiente de IG/SHAP/Grad-CAM.")

if st.button("Calcular LIME-EEG (oclusión)"):
    with st.spinner("Ocluyendo electrodos y ventanas temporales..."):
        lime_eeg = eeg_engine.calcular_lime_eeg(tensor, modelo_eeg)
    session.set_resultado_cacheado("lime_eeg", idx, lime_eeg)

lime_eeg = session.get_resultado_cacheado("lime_eeg", idx)
if lime_eeg is not None:
    if lime_eeg.clase != resultado.clase:
        st.warning("⚠️ La clase usada para LIME-EEG no coincide con la de Integrated "
                  "Gradients de arriba — vuelve a analizar la época antes de comparar.")

    l1, l2 = st.columns(2)
    with l1:
        st.caption("**Importancia por electrodo** (caída de probabilidad al ocluir)")
        df_comp2 = pd.DataFrame({
            "Integrated Gradients": resultado.pesos_electrodo,
            "LIME-EEG (oclusión)": lime_eeg.importancia_por_canal,
        }, index=ch_names)
        top_lime_eeg = set(df_comp2["LIME-EEG (oclusión)"].nlargest(10).index)
        top_ig_ahora = set(df_comp2["Integrated Gradients"].nlargest(10).index)
        electrodos_top2 = [ch for ch in ch_names if ch in (top_lime_eeg | top_ig_ahora)]
        st.bar_chart(df_comp2.loc[electrodos_top2], height=280)
        coinciden2 = len(top_lime_eeg & top_ig_ahora)
        st.caption(f"Coinciden {coinciden2}/10 electrodos con el top-10 de Integrated Gradients.")
    with l2:
        st.caption("**Importancia por ventana temporal** (caída de probabilidad al ocluir)")
        eje_ms2 = (np.arange(len(lime_eeg.importancia_por_tiempo)) / config.SFREQ * 1000).round().astype(int)
        df_tiempo2 = pd.DataFrame({"Importancia": lime_eeg.importancia_por_tiempo}, index=eje_ms2)
        st.line_chart(df_tiempo2, height=280)
        st.caption("Eje X en milisegundos desde el inicio de la época.")
else:
    st.caption("Presiona el botón para calcularlo.")

# ====================================================================
# COHERENCIA NEUROFISIOLÓGICA (Fase 1.4)
# ====================================================================
st.markdown("---")
st.markdown("### 🔬 Coherencia neurofisiológica")
st.caption("Contrasta las 4 explicaciones XAI contra la literatura del dolor: componentes "
          f"N2 ({config.N2_WINDOW_MS[0]}-{config.N2_WINDOW_MS[1]}ms) / "
          f"P300 ({config.P300_WINDOW_MS[0]}-{config.P300_WINDOW_MS[1]}ms) y sincronización "
          f"gamma ({config.GAMMA_BAND_HZ[0]:.0f}-{config.GAMMA_BAND_HZ[1]:.0f}Hz), típicamente "
          "con topografía centro-parietal. Esto NO valida que el modelo 'acierte' — valida que "
          "sus explicaciones sean fisiológicamente plausibles y no artefactos.")

xai_listo = session.get_resultado_cacheado("xai_ext_eeg", idx)
lime_listo = session.get_resultado_cacheado("lime_eeg", idx)

if xai_listo is None or lime_listo is None:
    st.info("Calcula primero **SHAP + Grad-CAM** y **LIME-EEG** arriba (en esta misma época) "
            "para poder comparar los 4 métodos entre sí.")
else:
    if st.button("Evaluar coherencia neurofisiológica"):
        with st.spinner("Calculando potencia gamma real y contrastando los 4 métodos..."):
            coherencia = coherencia_engine.evaluar_coherencia(
                tensor_eeg=tensor, ch_names=ch_names,
                resultado_eeg=resultado, xai_ext=xai_listo, lime_eeg=lime_listo,
            )
        session.set_resultado_cacheado("coherencia", idx, coherencia)

    coherencia = session.get_resultado_cacheado("coherencia", idx)
    if coherencia is not None:
        st.info(coherencia.resumen_texto)

        coh1, coh2 = st.columns(2)
        with coh1:
            st.caption("**Canal pico por método** — ¿cae en región somatosensorial central?")
            df_region = pd.DataFrame({
                "Electrodo pico": coherencia.canal_pico_por_metodo,
                "¿Consistente?": {m: ("✅" if ok else "❌")
                                 for m, ok in coherencia.region_consistente_por_metodo.items()},
            })
            st.dataframe(df_region, use_container_width=True)
        with coh2:
            st.caption("**Ventana temporal pico** — ¿cae en N2/P300?")
            df_ventana = pd.DataFrame({
                "Pico (ms)": coherencia.ventana_pico_ms_por_metodo,
                "¿Consistente?": {m: ("✅" if ok else "❌")
                                 for m, ok in coherencia.ventana_consistente_por_metodo.items()},
            })
            st.dataframe(df_ventana, use_container_width=True)

        st.caption("**Correlación con potencia gamma real** (Spearman ρ, por método — "
                  "positivo y alto = la explicación coincide con dónde hay más gamma de verdad)")
        df_gamma = pd.DataFrame({"ρ (Spearman)": coherencia.correlacion_gamma_por_metodo})
        st.bar_chart(df_gamma, height=220)

        st.caption("**Topografía de potencia gamma real** (30-80Hz, Welch) — compárala "
                  "visualmente con el topomap de Integrated Gradients de más arriba")
        st.pyplot(ui_components.dibujar_topomap(coherencia.potencia_gamma_por_canal, info),
                  use_container_width=True)
    else:
        st.caption("Presiona el botón para evaluar la coherencia.")

# ====================================================================
# PANEL DE EXPLICACIÓN NARRATIVA (Fase 3.5)
# ====================================================================
st.markdown("---")
st.markdown("### 💬 Explicación en lenguaje natural")

xai_ext_actual = session.get_resultado_cacheado("xai_ext_eeg", idx)
coherencia_actual = session.get_resultado_cacheado("coherencia", idx)

explicacion = narrativa_engine.generar_explicacion(
    pi=pi, resultado_eeg=resultado, ch_names=ch_names,
    resultado_cuerpo=resultado_cuerpo, xai_ext=xai_ext_actual, coherencia=coherencia_actual,
)
st.markdown(explicacion.texto)
if explicacion.nivel_detalle != "coherencia":
    st.caption("💡 Esta explicación mejora automáticamente a medida que calculas más secciones "
              "arriba (SHAP+Grad-CAM, LIME-EEG, coherencia) para esta misma época.")





# ====================================================================
# HISTORIAL DE LA SESIÓN
# ====================================================================
session.registrar_entrada(
    archivo=nombre_archivo,
    epoca=idx if es_fif else -1,
    pain_index=pi,
    nivel_eeg=resultado.nivel_dolor,
    confianza_eeg=resultado.confianza,
    cuerpo_pred=cuerpo_pred,
    cerebro_pred=cerebro_pred,
)

st.markdown("---")
st.markdown("### 📈 Historial de la sesión")
df_hist = session.get_historial_df()
if df_hist.empty or df_hist["pain_index"].isna().all():
    st.caption("Aún no hay registros con Pain Index en esta sesión.")
else:
    racha = session.racha_sobre_umbral(session.get_umbral_alerta())
    if racha >= 2:
        st.warning(f"⏱️ Pain Index sostenido **≥ {session.get_umbral_alerta():.1f}** durante "
                  f"**{racha} épocas consecutivas** — a diferencia de un pico puntual, esto "
                  f"sugiere dolor persistente, no un artefacto momentáneo.")
    st.line_chart(df_hist.dropna(subset=["pain_index"]).set_index("epoca")["pain_index"], height=220)
    with st.expander("Ver tabla completa del historial"):
        st.dataframe(df_hist, use_container_width=True, height=250)

    st.markdown("#### 📄 Exportar reporte del turno")
    rep_c1, rep_c2, rep_c3 = st.columns([1, 1, 2])
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    with rep_c1:
        st.download_button(
            "⬇️ Descargar CSV", data=reportes_engine.generar_csv_bytes(df_hist),
            file_name=f"historial_pain_index_{marca_tiempo}.csv", mime="text/csv",
        )
    with rep_c2:
        st.download_button(
            "⬇️ Descargar PDF",
            data=reportes_engine.generar_pdf_bytes(
                df_hist, nombre_turno=nombre_archivo, umbral_alerta=session.get_umbral_alerta()),
            file_name=f"reporte_turno_{marca_tiempo}.pdf", mime="application/pdf",
        )

    if st.button("🗑️ Limpiar historial"):
        session.limpiar_historial()
        st.rerun()

# ====================================================================
# TICK DEL MODO AUTOMÁTICO — debe ir al FINAL, después de renderizar
# y registrar todo. Duerme, avanza la época y fuerza un rerun.
#
# IMPORTANTE: acá solo se actualiza `session.epoca_actual` (una clave
# normal), NUNCA `st.session_state["slider_epoca_widget"]` — ese widget
# ya fue instanciado en este mismo run, y Streamlit prohíbe escribir la
# session_state de un widget después de haberlo creado. La sincronización
# real ocurre al inicio del PRÓXIMO run, antes de crear el slider de nuevo
# (ver el bloque "Controles de época" más arriba).
# ====================================================================
if es_fif and session.is_playing() and n_epocas > 0:
    time.sleep(intervalo_seg)
    siguiente = (idx + 1) % n_epocas
    session.set_epoca_actual(siguiente)
    st.rerun()