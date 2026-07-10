"""
backend/body_engine.py
====================================================================
Motor de inferencia corporal: XGBoost + SHAP sobre biomarcadores
(GSR, EMG, ECG). Reemplaza a `cuerpo_backend.py`, SIN la capa FastAPI
(por decisión del proyecto, no se usa API por ahora).

Este módulo tampoco sabe que existe Streamlit — devuelve datos
estructurados (dataclasses), no HTML ni nada de presentación.
====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src import config

logger = config.get_logger(__name__)

MAPA_DOLOR_CUERPO = {
    0: "Sin Dolor", 1: "Dolor Leve", 2: "Dolor Moderado",
    3: "Dolor Severo", 4: "Dolor Extremo",
}


@dataclass(frozen=True)
class PredictorCorporal:
    """Agrupa modelo + dataset + explainer, cargados UNA sola vez."""
    modelo: xgb.XGBClassifier
    dataset: pd.DataFrame
    features: list[str]
    explainer: shap.TreeExplainer


@dataclass(frozen=True)
class ResultadoCuerpo:
    clase: int
    estado: str
    certeza_pct: float
    probabilidades: np.ndarray            # (5,)
    top_drivers: list[dict]


def cargar_predictor_corporal(
    modelo_path=config.BODY_MODEL_PATH,
    dataset_path=config.BODY_DATASET_PATH,
) -> PredictorCorporal:
    """Carga el XGBoost corporal + dataset de fondo para SHAP.

    Lanza FileNotFoundError explícito si falta alguno de los dos
    archivos, en vez de fallar más adelante con un error críptico.
    """
    if not modelo_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo corporal en {modelo_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset corporal en {dataset_path}")

    modelo = xgb.XGBClassifier()
    modelo.load_model(str(modelo_path))
    df = pd.read_csv(dataset_path)
    features = [c for c in df.columns if c not in config.BODY_META_COLS]
    explainer = shap.TreeExplainer(modelo, data=df[features])
    logger.info("Predictor corporal cargado (%d filas, %d features)", len(df), len(features))
    return PredictorCorporal(modelo, df, features, explainer)


def fila_representativa(predictor: PredictorCorporal, clase: int) -> pd.DataFrame:
    """Fila de ejemplo del dataset para una clase BioVid (0-4).

    Se usa para SIMULAR una señal corporal en el dashboard: hoy no hay
    datos multimodales reales del mismo paciente (cuerpo + cerebro a
    la vez), así que el clínico elige la clase a mano.
    """
    idx = int(predictor.dataset.index[predictor.dataset["class_id"] == clase][0])
    return predictor.dataset[predictor.features].iloc[[idx]]


def predecir_proba_cuerpo(predictor: PredictorCorporal, fila: pd.DataFrame) -> np.ndarray:
    """Vector de 5 probabilidades (BioVid) para una fila de biomarcadores."""
    return predictor.modelo.predict_proba(fila[predictor.features])[0]


def procesar_datos_cuerpo(predictor: PredictorCorporal, fila_paciente: pd.DataFrame) -> ResultadoCuerpo:
    """Predicción + explicabilidad SHAP local para una fila de biomarcadores."""
    fila = fila_paciente[predictor.features]
    clase = int(predictor.modelo.predict(fila)[0])
    probs = predictor.modelo.predict_proba(fila)[0]
    certeza = float(probs[clase]) * 100

    shap_values = predictor.explainer.shap_values(fila)
    if isinstance(shap_values, list):
        shap_clase = shap_values[clase][0]
    elif len(shap_values.shape) == 3:
        shap_clase = shap_values[0, :, clase]
    else:
        shap_clase = shap_values[0]

    importancia = dict(zip(predictor.features, shap_clase))
    top3 = sorted(importancia.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_drivers = [{"biomarcador": k, "peso_shap": round(float(v), 4)} for k, v in top3]

    return ResultadoCuerpo(
        clase=clase,
        estado=MAPA_DOLOR_CUERPO.get(clase, "Desconocido"),
        certeza_pct=round(certeza, 2),
        probabilidades=probs,
        top_drivers=top_drivers,
    )



def clase_biovid_desde_evento(label: str | None) -> int | None:
    """Traduce 'NRS_6' -> 3 (PA3). None si no hay match."""
    if label is None:
        return None
    return config.NRS_LABEL_TO_BIOVID_CLASS.get(label)



def fila_por_clase(predictor: PredictorCorporal, clase: int, rng=None) -> pd.DataFrame:
    filas_clase = predictor.dataset.index[predictor.dataset["class_id"] == clase]
    idx = (rng or np.random.default_rng()).choice(filas_clase)
    return predictor.dataset[predictor.features].iloc[[idx]]