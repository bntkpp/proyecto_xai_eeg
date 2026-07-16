import streamlit as st
from datetime import datetime
from src.frontend import session
from src.backend import reportes_engine

def render(idx, es_fif, pi, resultado_eeg, cuerpo_pred, cerebro_pred, paciente_nombre, paciente_sexo, paciente_edad, nombre_archivo):
    st.markdown("### 📈 Registro Histórico de la Sesión Actual")
    session.registrar_entrada(
        archivo=nombre_archivo, epoca=idx if es_fif else -1, pain_index=pi,
        nivel_eeg=resultado_eeg.nivel_dolor, confianza_eeg=resultado_eeg.confianza,
        cuerpo_pred=cuerpo_pred, cerebro_pred=cerebro_pred,
    )

    st.markdown("#### 📝 Añadir Registros en Época Actual")
    st.markdown("**1. Registrar evento médico**")
    col_nota1, col_nota2 = st.columns([3, 1])
    with col_nota1:
        anotacion = st.text_input("Descripción", placeholder="Ej. El paciente reporta mareos", label_visibility="collapsed")
    with col_nota2:
        if st.button("Anotar Evento", use_container_width=True) and anotacion:
            st.session_state.anotaciones[idx] = st.session_state.anotaciones.get(idx, "") + f" | {anotacion}"
            st.toast(f"✅ Anotación guardada: {anotacion}", icon="✅")

    st.markdown("**2. Registrar medicación**")
    col_m1, col_m2, col_m3, col_m4 = st.columns([2, 2, 1, 1])
    with col_m1:
        farmaco = st.selectbox("Fármaco", ["Fentanilo", "Propofol", "Remifentanilo", "Ketamina", "Morfina", "Paracetamol", "Otro"], label_visibility="collapsed")
        if farmaco == "Otro": farmaco = st.text_input("Especificar", placeholder="Nombre del fármaco")
    with col_m2:
        dosis = st.number_input("Dosis", min_value=0.0, step=0.1, format="%.1f", label_visibility="collapsed")
    with col_m3:
        unidad = st.selectbox("Unidad", ["mg", "mcg", "g", "ml"], label_visibility="collapsed")
    with col_m4:
        if st.button("Aplicar Dosis", type="primary", use_container_width=True):
            if farmaco and dosis > 0:
                texto_registro = f"💊 Medicación: {farmaco} - {dosis} {unidad}"
                st.session_state.anotaciones[idx] = st.session_state.anotaciones.get(idx, "") + f" | {texto_registro}"
                st.toast(f"✅ Registrado: {farmaco}", icon="✅")
            else:
                st.error("Ingrese fármaco y dosis válida.")
    
    st.markdown("---")
    df_hist = session.get_historial_df().copy()
    
    if not df_hist.empty and not df_hist["pain_index"].isna().all():
        if "anotaciones" in st.session_state:
            df_hist["Anotación"] = df_hist["epoca"].map(st.session_state.anotaciones).fillna("")

        racha = session.racha_sobre_umbral(session.get_umbral_alerta())
        if racha >= 2:
            st.warning(f"⏱️ **Aviso Clínico**: Dolor persistente ≥ {session.get_umbral_alerta():.1f} por {racha} épocas.")
                      
        st.line_chart(df_hist.dropna(subset=["pain_index"]).set_index("epoca")["pain_index"], height=240)
        
        with st.expander("Revisar tabla de auditoría clínica", expanded=True):
            st.dataframe(df_hist, use_container_width=True, height=220, column_config={"Anotación": st.column_config.TextColumn("Anotación", width="large")})

        st.markdown("#### 📄 Exportar Documentación Médica")
        rep_c1, rep_c2, _ = st.columns([1, 1, 2])
        marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M")
        sexo_letra = paciente_sexo[0].upper() if paciente_sexo != "Seleccione" else "U"
        nombre_pdf_dinamico = f"{paciente_nombre.replace(' ', '_')}_{paciente_edad}_{sexo_letra}_{marca_tiempo}"

        with rep_c1:
            st.download_button("⬇️ Descargar .CSV", data=reportes_engine.generar_csv_bytes(df_hist), file_name=f"{nombre_pdf_dinamico}.csv", mime="text/csv")
        with rep_c2:
            st.download_button("⬇️ Descargar Firmado .PDF", data=reportes_engine.generar_pdf_bytes(df_hist, nombre_turno=nombre_pdf_dinamico, umbral_alerta=session.get_umbral_alerta()), file_name=f"{nombre_pdf_dinamico}.pdf", mime="application/pdf")

        st.markdown("---")
        if st.button("🗑️ Limpiar Historial", key="btn_clear_hist"):
            session.limpiar_historial()
            st.session_state.anotaciones = {}
            st.rerun()
