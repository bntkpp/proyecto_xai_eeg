"""
api_cuerpo.py
====================================================================
Endpoint FastAPI para exponer el motor de inferencia corporal
(XGBoost + TreeSHAP) al Dashboard de Frontend.

Ejecutar desde la raíz del proyecto:
    python -m src.cuerpo_backend.api_cuerpo

O con uvicorn directamente:
    uvicorn src.cuerpo_backend.api_cuerpo:app --reload --host 0.0.0.0 --port 8000

Documentación interactiva automática:
    http://localhost:8000/docs
====================================================================
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

try:
    # Cuando se ejecuta como módulo: python -m src.cuerpo_backend.api_cuerpo
    from .cuerpo_backend import procesar_datos_cuerpo
except ImportError:
    # Cuando se ejecuta directamente: python src/cuerpo_backend/api_cuerpo.py
    from cuerpo_backend import procesar_datos_cuerpo

app = FastAPI(
    title="API Cuerpo - Detección de Dolor",
    description="Recibe biomarcadores corporales en tiempo real y retorna predicción de dolor + explicabilidad SHAP.",
    version="1.0.0",
)


class PacienteCuerpo(BaseModel):
    """Payload con las 10 variables corporales del Equipo de Élite."""

    gsr_slope_late: float = Field(..., description="Pendiente tardía de la GSR")
    gsr_min_max_diff: float = Field(..., description="Diferencia min-max de GSR")
    emg_corrugator_max: float = Field(..., description="Máximo del EMG corrugador")
    ecg_apen: float = Field(..., description="Entropía aproximada del ECG")
    emg_zygomaticus_auc: float = Field(..., description="Área bajo la curva del EMG cigomático")
    emg_corrugator_mean: float = Field(..., description="Media del EMG corrugador")
    emg_trapezius_max: float = Field(..., description="Máximo del EMG trapecio")
    emg_trapezius_auc: float = Field(..., description="Área bajo la curva del EMG trapecio")
    ecg_bpm: float = Field(..., description="Ritmo cardíaco en BPM")
    gsr_apen: float = Field(..., description="Entropía aproximada de la GSR")


@app.get("/")
def root():
    """Endpoint de salud/verificación."""
    return {
        "mensaje": "API Cuerpo activa",
        "endpoint": "POST /predecir-dolor-cuerpo",
        "docs": "/docs",
    }


@app.post("/predecir-dolor-cuerpo")
def predecir_dolor_cuerpo(datos: PacienteCuerpo):
    """
    Recibe los biomarcadores corporales de un paciente y retorna:
    - Nivel de dolor predicho (NRS)
    - Certeza de la predicción
    - Top 3 variables que más influyeron (SHAP)
    """
    # Convertir payload Pydantic a DataFrame de 1 fila
    fila_paciente = pd.DataFrame([datos.model_dump()])

    # Ejecutar inferencia + XAI
    resultado = procesar_datos_cuerpo(fila_paciente)

    return resultado


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
