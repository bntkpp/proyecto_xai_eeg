"""
backend/reportes_engine.py
====================================================================
Fase 3.6 — Reportes clínicos: exportación CSV y PDF del historial de
Pain Index del turno (sesión actual del dashboard).

Genera los archivos EN MEMORIA (bytes) — el frontend los ofrece con
st.download_button, sin necesidad de escribir nada a disco en el
servidor donde corre Streamlit.
====================================================================
"""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src import config

logger = config.get_logger(__name__)


def generar_csv_bytes(df_historial: pd.DataFrame) -> bytes:
    """CSV del historial completo, listo para st.download_button."""
    return df_historial.to_csv(index=False).encode("utf-8")


def _grafico_pain_index_png(df_historial: pd.DataFrame) -> io.BytesIO:
    """Gráfico de línea del Pain Index a través de las épocas, como PNG en memoria."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    df_validas = df_historial.dropna(subset=["pain_index"])
    if not df_validas.empty:
        ax.plot(df_validas["epoca"], df_validas["pain_index"], marker="o", color="#e03131")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pain Index")
    ax.set_ylim(0, 10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def generar_pdf_bytes(df_historial: pd.DataFrame, *, nombre_turno: str,
                      umbral_alerta: float) -> bytes:
    """Reporte clínico en PDF: resumen del turno + gráfico + tabla completa."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Reporte de Turno &mdash; Monitor de Dolor XAI-EEG", styles["Title"]))
    story.append(Paragraph(
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &middot; "
        f"Archivo: {nombre_turno}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Herramienta de apoyo a la decision clinica. No reemplaza el juicio profesional "
        "ni constituye un diagnostico.", styles["Italic"]))
    story.append(Spacer(1, 14))

    # ---- Resumen del turno ----
    df_validas = df_historial.dropna(subset=["pain_index"])
    n_registros = len(df_historial)
    if not df_validas.empty:
        pi_prom = df_validas["pain_index"].mean()
        pi_max = df_validas["pain_index"].max()
        pi_min = df_validas["pain_index"].min()
        n_sobre_umbral = int((df_validas["pain_index"] >= umbral_alerta).sum())
    else:
        pi_prom = pi_max = pi_min = None
        n_sobre_umbral = 0

    resumen_datos = [
        ["Registros totales", str(n_registros)],
        ["Pain Index promedio", f"{pi_prom:.1f}" if pi_prom is not None else "-"],
        ["Pain Index maximo", f"{pi_max:.1f}" if pi_max is not None else "-"],
        ["Pain Index minimo", f"{pi_min:.1f}" if pi_min is not None else "-"],
        ["Umbral de alerta configurado", f"{umbral_alerta:.1f}"],
        ["Epocas que cruzaron el umbral", str(n_sobre_umbral)],
    ]
    tabla_resumen = Table(resumen_datos, colWidths=[9 * cm, 6 * cm])
    tabla_resumen.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    story.append(Paragraph("Resumen del turno", styles["Heading2"]))
    story.append(tabla_resumen)
    story.append(Spacer(1, 14))

    # ---- Gráfico ----
    if not df_validas.empty:
        story.append(Paragraph("Evolucion del Pain Index", styles["Heading2"]))
        story.append(Image(_grafico_pain_index_png(df_historial), width=16 * cm, height=6.5 * cm))
        story.append(Spacer(1, 14))

    # ---- Tabla completa ----
    story.append(Paragraph("Historial completo", styles["Heading2"]))
    encabezados = ["Hora", "Epoca", "PI", "Nivel EEG", "Conf.", "Cuerpo", "Cerebro"]
    filas = [encabezados]
    for _, fila in df_historial.iterrows():
        filas.append([
            str(fila.get("timestamp", "")),
            str(fila.get("epoca", "")),
            f"{fila['pain_index']:.1f}" if pd.notna(fila.get("pain_index")) else "-",
            str(fila.get("nivel_eeg", "") or "-"),
            f"{fila['confianza_eeg']:.0%}" if pd.notna(fila.get("confianza_eeg")) else "-",
            str(fila.get("cuerpo_pred", "") or "-"),
            str(fila.get("cerebro_pred", "") or "-"),
        ])

    tabla_hist = Table(filas, repeatRows=1)
    tabla_hist.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
    ]))
    story.append(tabla_hist)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()