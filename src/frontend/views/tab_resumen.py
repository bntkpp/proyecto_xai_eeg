import streamlit as st
import numpy as np
from src import config
from src.frontend import session, ui_components
from src.backend import body_engine, narrativa_engine

def render(idx, has_fusion, clase_real, label_evento, predictor_cuerpo, fuser, prob_cerebro, cerebro_pred, resultado_eeg, ch_names):
    pi, prob_fus, cuerpo_pred, resultado_cuerpo = None, None, "N/A", None

    if has_fusion:
        st.markdown("### 🩺 Pain Index Multimodal")
        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            if clase_real is not None:
                st.success(f"🔗 Pareo automático: Época {idx} = **{label_evento}** → {config.BIOVID_LABELS[clase_real]}")
                cid_sel = clase_real
            else:
                st.info("Selección manual de contexto (Señal corporal simulada):")
                clase_lbl = st.selectbox("Ajustar señal corporal base", config.BIOVID_LABELS, index=3, key="sb_resumen_cuerpo")
                cid_sel = config.BIOVID_LABELS.index(clase_lbl)
            
            fila_cuerpo = body_engine.fila_representativa(predictor_cuerpo, cid_sel)
            resultado_cuerpo = body_engine.procesar_datos_cuerpo(predictor_cuerpo, fila_cuerpo)
            prob_cuerpo = resultado_cuerpo.probabilidades
                
        with ctrl2:
            peso_cerebro = st.slider("Peso del cerebro (%) vs Cuerpo", 0, 100, 50, step=5, key="slider_resumen_peso")
            
        w_b, w_c = peso_cerebro / 100.0, 1.0 - (peso_cerebro / 100.0)
        pi, prob_fus = fuser.calcular_pain_index(prob_cuerpo, prob_cerebro, w_cuerpo=w_c, w_cerebro=w_b)
        cuerpo_pred = config.BIOVID_LABELS[int(np.argmax(prob_cuerpo))]
        
        gc1, gc2, gc3 = st.columns([1, 2, 1])
        with gc2:
            st.markdown(ui_components.tarjeta_gauge(pi, w_c, w_b, cuerpo_pred, cerebro_pred), unsafe_allow_html=True)
            
        umbral = session.get_umbral_alerta()
        if pi >= umbral:
            st.markdown("<br>", unsafe_allow_html=True)
            st.error(f"**Alerta analgésica** — El Pain Index (**{pi:.1f}**) alcanza umbral (**{umbral:.1f}**).", icon="🚨")
            if st.session_state.get("last_alert_epoch", -1) != idx:
                st.toast(f"Atención: Umbral de dolor superado ({pi:.1f}/10)", icon="🚨")
                st.session_state.last_alert_epoch = idx
    else:
        st.warning("Fusión no disponible.")

    st.markdown("---")
    st.markdown("### 💬 Explicación en lenguaje natural")
    xai_ext_actual = session.get_resultado_cacheado("xai_ext_eeg", idx)
    coherencia_actual = session.get_resultado_cacheado("coherencia", idx)

    explicacion = narrativa_engine.generar_explicacion(
        pi=pi, resultado_eeg=resultado_eeg, ch_names=ch_names,
        resultado_cuerpo=resultado_cuerpo, xai_ext=xai_ext_actual, coherencia=coherencia_actual,
    )
    st.markdown(explicacion.texto)

    return pi, prob_fus, cuerpo_pred, resultado_cuerpo
