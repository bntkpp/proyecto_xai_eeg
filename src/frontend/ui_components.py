"""
frontend/ui_components.py
====================================================================
Componentes visuales puros: reciben datos YA CALCULADOS por el
backend y devuelven HTML/figuras. No cargan modelos, no hacen
inferencia, no conocen rutas de archivos de datos.
====================================================================
"""
from __future__ import annotations

import os
import re
import tempfile
import uuid

import matplotlib.pyplot as plt
import mne
import numpy as np

from src import config


# =========================================================
# FORZAR MODO OSCURO PARA MATPLOTLIB (Letras blancas)
# =========================================================
plt.rcParams.update({
    "text.color": "white",          # Color del texto general (títulos)
    "axes.labelcolor": "white",     # Color de las etiquetas (X, Y)
    "xtick.color": "white",         # Color de los números del eje X
    "ytick.color": "white",         # Color de los números del eje Y
    "axes.edgecolor": "white",      # Color de los bordes del gráfico
    "figure.facecolor": "none",     # Fondo transparente para la figura
    "axes.facecolor": "none"        # Fondo transparente para los ejes
})




COLOR_NIVEL = {
    "Sin Dolor (NRS 0-2)":    "#2e7d32",
    "Dolor Leve (NRS 4)":     "#f9a825",
    "Dolor Moderado (NRS 6)": "#ef6c00",
    "Dolor Severo (NRS 8)":   "#c62828",
}

# Zonas del medidor (geometría fija: centro 150,150 · radio 118 · semicírculo).
_GAUGE_ZONAS = [
    (0, 3, "#37b24d", "M 32.0 150.0 A 118 118 0 0 1 80.6 54.5"),
    (3, 5, "#f1a208", "M 80.6 54.5 A 118 118 0 0 1 150.0 32.0"),
    (5, 7, "#f76707", "M 150.0 32.0 A 118 118 0 0 1 219.4 54.5"),
    (7, 10, "#e03131", "M 219.4 54.5 A 118 118 0 0 1 268.0 150.0"),
]


def veredicto_pain_index(pi: float) -> tuple[str, str]:
    """Palabra clínica + color por severidad (lectura de un vistazo)."""
    if pi < 3:
        return "SIN DOLOR", "#37b24d"
    if pi < 5:
        return "DOLOR LEVE", "#f1a208"
    if pi < 7:
        return "DOLOR MODERADO", "#f76707"
    return "DOLOR SEVERO", "#e03131"


def tarjeta_gauge(pi: float, w_c: float, w_b: float, cuerpo_pred: str, cerebro_pred: str) -> str:
    """Medidor tipo monitor clínico: aguja + zona encendida + veredicto."""
    palabra, col = veredicto_pain_index(pi)
    pi_c = min(max(pi, 0.0), 10.0)
    th = np.radians(180 - 18 * pi_c)
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


def _solo_eeg_para_topomap(pesos_63, info):
    """Devuelve (pesos, info) quedándose solo con electrodos del cuero cabelludo."""
    tipos = info.get_channel_types()
    keep = [i for i, (t, n) in enumerate(zip(tipos, info["ch_names"]))
            if t == "eeg" and n not in config.NO_SCALP_CHANNELS]
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
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig


def dibujar_gradcam_heatmap(mapa_canal_tiempo: np.ndarray, ch_names, sfreq: int = config.SFREQ):
    """Heatmap canal × tiempo de Grad-CAM (capa temporal_conv, ANTES de que
    spatial_conv colapse los electrodos). Filas = electrodos en su orden
    real (no alfabético), columnas = tiempo en milisegundos."""
    n_channels, n_samples = mapa_canal_tiempo.shape
    tiempos_ms = (np.arange(n_samples) / sfreq) * 1000

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(mapa_canal_tiempo, aspect="auto", cmap="hot",
                   extent=[tiempos_ms[0], tiempos_ms[-1], n_channels, 0])
    ax.set_xlabel("Tiempo (ms)")
    ax.set_ylabel("Electrodo")
    # Con ~60 electrodos no caben todas las etiquetas legibles: se muestra 1 de cada `paso`.
    paso = max(1, n_channels // 25)
    ax.set_yticks(np.arange(0, n_channels, paso) + 0.5)
    ax.set_yticklabels([ch_names[i] for i in range(0, n_channels, paso)], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Importancia Grad-CAM (0-1)")
    ax.set_title("Grad-CAM: canal × tiempo (capa temporal_conv)", fontsize=11)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig


def guardar_temporal(archivo_subido) -> str:
    """Guarda un archivo subido de Streamlit en un temporal y devuelve su ruta.

    Sanitiza el nombre (evita path traversal) y le agrega un prefijo único
    (evita colisiones entre pacientes/usuarios concurrentes).
    """
    nombre_seguro = re.sub(r"[^A-Za-z0-9_.-]", "_", archivo_subido.name)
    destino = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{nombre_seguro}")
    with open(destino, "wb") as f:
        f.write(archivo_subido.read())
    return destino
