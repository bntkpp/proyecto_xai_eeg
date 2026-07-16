import streamlit as st
import pandas as pd
from src import config
from src.frontend import session, resources
from src.backend import body_engine

def render(idx, has_fusion, resultado_cuerpo, prob_fus, predictor_cuerpo):
    st.markdown("### 🫀 Análisis Detallado del Modelo Corporal (Biomarcadores)")
    if has_fusion and resultado_cuerpo is not None:
        if prob_fus is not None:
            with st.expander("Ver distribución detallada de la probabilidad fusionada", expanded=True):
                df_fus = pd.DataFrame({"Probabilidad": prob_fus}, index=config.BIOVID_LABELS)
                st.bar_chart(df_fus, height=200)
        
        st.markdown("#### 🔍 Explicabilidad de Señal Autonómica")
        exp_shap, exp_lime = st.columns(2)
        with exp_shap:
            st.caption("**SHAP (Atribución Global)**")
            df_shap = pd.DataFrame(resultado_cuerpo.top_drivers).set_index("biomarcador")
            st.bar_chart(df_shap, height=220)
        with exp_lime:
            st.caption("**LIME Corporal (Perturbación Local)**")
            if st.button("Calcular LIME (Cuerpo)", disabled=session.is_playing(), key="btn_cuerpo_lime"):
                with st.spinner("Calculando vecindarios locales..."):
                    lime_explainer = resources.get_lime_explainer()
                    fila_cuerpo = body_engine.fila_por_clase(predictor_cuerpo, resultado_cuerpo.clase)
                    top_lime = body_engine.explicar_con_lime(predictor_cuerpo, lime_explainer, fila_cuerpo, resultado_cuerpo.clase)
                session.set_resultado_cacheado("lime_cuerpo", idx, top_lime)

            top_lime_cache = session.get_resultado_cacheado("lime_cuerpo", idx)
            if top_lime_cache is not None:
                df_lime = pd.DataFrame(top_lime_cache).set_index("biomarcador")
                st.bar_chart(df_lime, height=220)
    else:
        st.info("No hay señales corporales cargadas.")
