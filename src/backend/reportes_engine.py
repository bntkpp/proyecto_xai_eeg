"""
backend/reportes_engine.py
====================================================================
Motor de generación de reportes clínicos en PDF y CSV.
"""
import io
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from src import config

def generar_csv_bytes(df: pd.DataFrame) -> bytes:
    """Genera el reporte en formato CSV."""
    return df.to_csv(index=False).encode('utf-8')

def generar_pdf_bytes(df: pd.DataFrame, nombre_turno: str, umbral_alerta: float) -> bytes:
    """
    Genera un PDF de auditoría clínica de múltiples páginas.
    - Página 1: Resumen del turno (Estadísticas y metadatos) con Logo.
    - Página 2: Gráfico con ejes definidos.
    - Página 3+: Tabla de registro de eventos (incluyendo Anotaciones).
    """
    buf = io.BytesIO()
    df_plot = df.dropna(subset=["pain_index"]).copy()
    
    # Usar el contexto 'default' para ignorar el modo oscuro de Streamlit
    with plt.style.context('default'):
        with PdfPages(buf) as pdf:
            
            # ---------------------------------------------------------
            # PÁGINA 1: RESUMEN DEL TURNO
            # ---------------------------------------------------------
            fig_sum, ax_sum = plt.subplots(figsize=(10, 6))
            fig_sum.patch.set_facecolor('white')
            ax_sum.axis("off")
            
            # Cargar e insertar Logo en el PDF (.jpeg)
            logo_path = config.LOGO_PATH
            text_x_offset = 0.1 
            
            if logo_path.exists():
                try:
                    img_logo = Image.open(logo_path)
                    # [izquierda, abajo, ancho, alto]
                    ax_logo = fig_sum.add_axes([0.05, 0.75, 0.18, 0.18]) 
                    ax_logo.imshow(img_logo)
                    ax_logo.axis('off')
                    text_x_offset = 0.25 # Desplaza el texto a la derecha
                except Exception:
                    pass 
            
            # Encabezados y Metadatos
            fig_sum.text(text_x_offset, 0.90, "Reporte de Turno — Monitor de Dolor XAI-EEG", fontsize=18, fontweight='bold', color='black')
            fecha_generacion = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig_sum.text(text_x_offset, 0.83, f"Generado: {fecha_generacion} · Archivo: {nombre_turno}", fontsize=11, color='black')
            fig_sum.text(text_x_offset, 0.77, "Herramienta de apoyo a la decision clinica. No reemplaza el juicio profesional ni constituye un diagnostico.", fontsize=11, style='italic', color='black')
            
            # Título de la tabla de resumen
            fig_sum.text(0.1, 0.65, "Resumen del turno", fontsize=16, fontweight='bold', color='black')
            
            # Cálculos estadísticos
            if not df_plot.empty:
                registros = len(df_plot)
                promedio = f"{df_plot['pain_index'].mean():.1f}"
                maximo = f"{df_plot['pain_index'].max():.1f}"
                minimo = f"{df_plot['pain_index'].min():.1f}"
                epocas_cruzaron = len(df_plot[df_plot['pain_index'] >= umbral_alerta])
            else:
                registros, promedio, maximo, minimo, epocas_cruzaron = 0, "N/A", "N/A", "N/A", 0

            datos_resumen = [
                ["Registros totales", str(registros)],
                ["Pain Index promedio", promedio],
                ["Pain Index maximo", maximo],
                ["Pain Index minimo", minimo],
                ["Umbral de alerta configurado", f"{umbral_alerta:.1f}"],
                ["Epocas que cruzaron el umbral", str(epocas_cruzaron)]
            ]
            
            # Crear tabla de resumen
            tabla_res = ax_sum.table(
                cellText=datos_resumen, 
                loc='center', 
                cellLoc='left', 
                bbox=[0.1, 0.15, 0.8, 0.45] 
            )
            
            tabla_res.auto_set_font_size(False)
            tabla_res.set_fontsize(11)
            
            for key, cell in tabla_res.get_celld().items():
                cell.set_text_props(color='black')
                cell.set_edgecolor('black')
                cell.PAD = 0.05
            
            pdf.savefig(fig_sum)
            plt.close(fig_sum)

            # ---------------------------------------------------------
            # PÁGINA 2: GRÁFICO DEL PAIN INDEX
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('white') 
            
            if not df_plot.empty:
                ax.plot(df_plot["epoca"], df_plot["pain_index"], marker='o', color="#e63946", linewidth=2, label="Pain Index")
                ax.axhline(y=umbral_alerta, color="black", linestyle="--", linewidth=1.5, label="Umbral de Alerta")
            
            ax.set_title(f"Evolución del Pain Index - Sesión: {nombre_turno}", fontsize=14, fontweight="bold", color="black")
            ax.set_xlabel("Época (Intervalo de tiempo)", fontsize=12, color="black")
            ax.set_ylabel("Nivel de Dolor (Pain Index 0-10)", fontsize=12, color="black")
            ax.set_ylim(-0.5, 10.5)
            
            if not df_plot.empty:
                ax.set_xticks(df_plot["epoca"].astype(int))
                
            ax.tick_params(axis='both', which='major', labelsize=10, colors="black")
            ax.grid(True, linestyle=":", alpha=0.7, color="gray")
            
            legend = ax.legend(loc="upper right", facecolor="white", edgecolor="black")
            for text in legend.get_texts():
                text.set_color("black")
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
            # ---------------------------------------------------------
            # PÁGINA 3: TABLA CON ANOTACIONES MÉDICAS
            # ---------------------------------------------------------
            columnas_deseadas = ["epoca", "pain_index", "cuerpo_pred", "cerebro_pred", "Anotación"]
            columnas_finales = [c for c in columnas_deseadas if c in df.columns]
            
            df_table = df[columnas_finales].copy()
            df_table.fillna("", inplace=True)
            
            if "pain_index" in df_table.columns:
                df_table["pain_index"] = df_table["pain_index"].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
            if "Anotación" in df_table.columns:
                df_table["Anotación"] = df_table["Anotación"].apply(lambda x: str(x)[:65] + "..." if len(str(x)) > 65 else str(x))
            
            fig_tab, ax_tab = plt.subplots(figsize=(10, max(3, len(df_table)*0.5))) 
            fig_tab.patch.set_facecolor('white')
            ax_tab.axis("off")
            ax_tab.set_title("Auditoría Clínica y Registro de Eventos Médicos", fontsize=14, fontweight="bold", pad=15, color="black")
            
            tabla = ax_tab.table(
                cellText=df_table.values,
                colLabels=[c.capitalize().replace("_", " ") for c in df_table.columns],
                loc="center",
                cellLoc="center"
            )
            
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(9)
            tabla.scale(1, 1.5)
            
            if "Anotación" in df_table.columns:
                idx_anotacion = df_table.columns.get_loc("Anotación")
            
            for key, cell in tabla.get_celld().items():
                cell.set_text_props(color='black')
                cell.set_edgecolor('black')
                
                if key[0] == 0:  
                    cell.set_text_props(weight='bold', color='black')
                    cell.set_facecolor('#e0e0e0') 
                
                if "Anotación" in df_table.columns:
                    row, col = key
                    if col == idx_anotacion:
                        cell.set_width(0.35) 
                    else:
                        cell.set_width(0.15)
            
            fig_tab.tight_layout()
            pdf.savefig(fig_tab)
            plt.close(fig_tab)

    buf.seek(0)
    return buf.getvalue()
