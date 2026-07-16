import streamlit as st
import pandas as pd
from src import config

def render(info, n_epocas, es_fif, has_fusion, predictor_cuerpo):
    st.markdown("### 📁 Gestión y Diagnóstico de Datos")
    st.caption("Exploración de los datos subyacentes.")
    
    st.header("🧠 Datos del Cerebro (EEG)")
    with st.container(border=True):
        if info is not None:
            e1, e2, e3, e4 = st.columns(4)
            with e1: st.metric("Frecuencia Muestreo", f"{int(info['sfreq'])} Hz")
            with e2: st.metric("Canales", f"{len(info['ch_names'])}")
            with e3: st.metric("Épocas", f"{n_epocas if es_fif else 1}")
            with e4: st.metric("Duración Época", f"{(config.N_SAMPLES / config.SFREQ):.2f} s")
            with st.expander("Ver distribución de electrodos"):
                st.write(", ".join(info['ch_names']))
        else:
            st.info("Sube un archivo EEG para ver las métricas.")

    st.header("🫀 Datos de Señales Autonómicas (Cuerpo)")
    with st.container(border=True):
        if has_fusion and predictor_cuerpo and hasattr(predictor_cuerpo, 'dataset') and not predictor_cuerpo.dataset.empty:
            df_cuerpo = predictor_cuerpo.dataset
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Muestras", f"{len(df_cuerpo):,}")
            with c2: st.metric("Pacientes", f"{df_cuerpo['subject_id'].nunique() if 'subject_id' in df_cuerpo.columns else 'N/A'}")
            with c3: st.metric("Biomarcadores", f"{len(predictor_cuerpo.features)}")
            with c4: st.metric("Clases Dolor", "5")
                
            st.markdown("---")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**⚖️ Balance de Clases**")
                if 'class_id' in df_cuerpo.columns:
                    df_balance = df_cuerpo['class_id'].value_counts().sort_index().reset_index()
                    df_balance["Etiqueta"] = [config.BIOVID_LABELS[int(c)] for c in df_balance["class_id"]]
                    st.bar_chart(df_balance.set_index("Etiqueta")["count"], color="#4dabf7")
            with col_b2:
                st.markdown("**📈 Tendencia Fisiológica**")
                if 'class_id' in df_cuerpo.columns:
                    feat = st.selectbox("Selecciona biomarcador:", predictor_cuerpo.features)
                    df_trend = df_cuerpo.groupby('class_id')[feat].mean().reset_index()
                    df_trend["Etiqueta"] = [config.BIOVID_LABELS[int(c)] for c in df_trend["class_id"]]
                    st.line_chart(df_trend.set_index("Etiqueta")[feat], color="#ff6b6b")

            with st.expander("🔗 Matriz de Correlación"):
                vars_disp = [f for f in predictor_cuerpo.features if f in df_cuerpo.columns]
                sel = st.multiselect("Cruzar:", options=vars_disp, default=vars_disp[:min(6, len(vars_disp))])
                if len(sel) >= 2:
                    st.dataframe(df_cuerpo[sel].corr().style.background_gradient(cmap="coolwarm", axis=None).format(precision=3), use_container_width=True)
        else:
            st.info("Capa de Datos Local Desconectada.")
