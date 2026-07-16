import streamlit as st
import numpy as np
import pandas as pd
import dataclasses
from src import config
from src.frontend import session, ui_components
from src.backend import eeg_engine

def render(idx, tensor, modelo_eeg, resultado_eeg, ch_names, info):
    st.markdown("### 🧠 Métricas y Modelado Avanzado de EEG")
    col1, col2 = st.columns([1, 1])
    with col1:
        color = ui_components.COLOR_NIVEL.get(resultado_eeg.nivel_dolor, "#455a64")
        st.markdown(f"<div style='padding:18px;border-radius:12px;background:{color};color:white'>"
                    f"<div style='font-size:14px;opacity:.85'>Predicción de Red (EEGNet)</div>"
                    f"<div style='font-size:30px;font-weight:700'>{resultado_eeg.nivel_dolor}</div>"
                    f"<div style='font-size:14px;opacity:.85'>Confianza: {resultado_eeg.confianza:.0%}</div>"
                    f"</div>", unsafe_allow_html=True)
        st.progress(min(max(resultado_eeg.confianza, 0.0), 1.0))
        
        explicacion_texto = getattr(resultado_eeg, 'explicacion', "Análisis completado.")
        st.success(explicacion_texto)
        
        electrodo_dom = ch_names[int(np.argmax(resultado_eeg.pesos_electrodo))]
        st.metric("Electrodo dominante", electrodo_dom)
    with col2:
        st.pyplot(ui_components.dibujar_topomap(resultado_eeg.pesos_electrodo, info), use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 📈 Visor de Señal Cruda (Top 3 Electrodos Relevantes)")
        top_3_idx = np.argsort(resultado_eeg.pesos_electrodo)[-3:][::-1]
        top_3_names = [ch_names[i] for i in top_3_idx]
        tensor_sq = tensor.cpu().squeeze().numpy()
        eje_ms = (np.arange(tensor_sq.shape[-1]) / config.SFREQ * 1000).round().astype(int)
        df_raw = pd.DataFrame(tensor_sq[top_3_idx].T, columns=top_3_names, index=eje_ms)
        st.line_chart(df_raw, height=250)

    with st.container(border=True):
        st.markdown("#### 🧬 SHAP + Grad-CAM (EEG)")
        if st.button("Calcular SHAP + Grad-CAM", disabled=session.is_playing(), key="btn_eeg_shap"):
            with st.spinner("Calculando GradientShap y Grad-CAM..."):
                session.set_resultado_cacheado("xai_ext_eeg", idx, eeg_engine.calcular_xai_extendido(tensor, modelo_eeg))

        xai_ext = session.get_resultado_cacheado("xai_ext_eeg", idx)
        if xai_ext is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("**Integrated Gradients vs SHAP**")
                df_comp = pd.DataFrame({"IG": resultado_eeg.pesos_electrodo, "SHAP": xai_ext.shap_por_canal}, index=ch_names)
                top_ig, top_shap = set(df_comp["IG"].nlargest(10).index), set(df_comp["SHAP"].nlargest(10).index)
                st.bar_chart(df_comp.loc[list(top_ig | top_shap)], height=250)
            with c2:
                st.caption("**Ventana temporal crítica**")
                df_tiempo = pd.DataFrame({"Importancia": xai_ext.gradcam_ventana_temporal}, index=eje_ms)
                st.line_chart(df_tiempo, height=250)
            st.pyplot(ui_components.dibujar_gradcam_heatmap(xai_ext.gradcam_canal_tiempo, ch_names), use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 🧩 LIME-EEG (Aproximación local)")
        if st.button("Calcular LIME-EEG", key="btn_eeg_lime", disabled=session.is_playing()):
            with st.spinner("Ocluyendo electrodos..."):
                session.set_resultado_cacheado("lime_eeg", idx, eeg_engine.calcular_lime_eeg(tensor, modelo_eeg))

        lime_listo = session.get_resultado_cacheado("lime_eeg", idx)
        if lime_listo is not None:
            st.caption("**Impacto de la oclusión temporal sobre la predicción**")
            try:
                valores_lime = None
                if dataclasses.is_dataclass(lime_listo):
                    diccionario = dataclasses.asdict(lime_listo)
                    for key, value in diccionario.items():
                        if isinstance(value, (np.ndarray, list)) and len(value) == len(eje_ms):
                            valores_lime = value
                            break
                    if valores_lime is None:
                        arrays = [v for v in diccionario.values() if isinstance(v, (np.ndarray, list))]
                        valores_lime = arrays[0] if arrays else None
                elif isinstance(lime_listo, tuple):
                    valores_lime = lime_listo[0]
                else:
                    valores_lime = lime_listo

                if valores_lime is not None:
                    df_lime = pd.DataFrame({"Caída de Confianza (Importancia)": valores_lime}, index=eje_ms)
                    st.line_chart(df_lime, height=250)
                else:
                    st.error("LIME calculó correctamente, pero no se encontró ninguna serie temporal en el objeto para graficar.")
            except Exception as e:
                st.error(f"Error al procesar el gráfico. Variables dentro de LIME: {dir(lime_listo)}. Detalle: {e}")
