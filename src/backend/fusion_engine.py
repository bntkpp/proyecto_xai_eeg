"""
backend/fusion_engine.py
====================================================================
HITO V — Fusión multimodal cuerpo + cerebro -> Pain Index (0-10)
calibrado con regresión isotónica.

Pipeline:
    prob_cuerpo(5)  ┐
                    ├─► fusión ponderada (W1,W2) ─► score continuo (0-4)
    prob_cerebro(4) ┘        (el cerebro se mapea a la grilla de 5 de BioVid)
                                                        │
                                          Regresión Isotónica (monótona)
                                                        ▼
                                              Pain Index  0.0 – 10.0
====================================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src import config
from src.backend import body_engine

logger = config.get_logger(__name__)

_POS_BIOVID = np.linspace(0.0, 1.0, 5)     # [0, .25, .50, .75, 1]
_POS_CEREBRO = np.linspace(0.0, 1.0, 4)    # [0, .333, .667, 1]


def mapear_cerebro_a_biovid(prob_cerebro: np.ndarray) -> np.ndarray:
    """Convierte 4 probabilidades (NRS) en 5 (grilla BioVid) por
    redistribución ordinal, conservando suma=1 y centro de masa."""
    prob_cerebro = np.asarray(prob_cerebro, dtype=float).reshape(-1)
    if prob_cerebro.size != 4:
        raise ValueError(f"prob_cerebro debe tener 4 clases, tiene {prob_cerebro.size}")
    salida = np.zeros(5)
    for i, p in enumerate(prob_cerebro):
        pos = _POS_CEREBRO[i]
        if pos <= 0:
            salida[0] += p
        elif pos >= 1:
            salida[-1] += p
        else:
            r = int(np.searchsorted(_POS_BIOVID, pos))
            l = r - 1
            frac = (pos - _POS_BIOVID[l]) / (_POS_BIOVID[r] - _POS_BIOVID[l])
            salida[l] += p * (1 - frac)
            salida[r] += p * frac
    return salida


def _resolver_pesos(w_cuerpo: Optional[float], w_cerebro: Optional[float]) -> tuple[float, float]:
    """Prioridad: argumento > .env > default 0.5/0.5. Normaliza a suma 1."""
    w_c = config.W_CUERPO_DEFAULT if w_cuerpo is None else float(w_cuerpo)
    w_b = config.W_CEREBRO_DEFAULT if w_cerebro is None else float(w_cerebro)
    total = w_c + w_b
    if total <= 0:
        raise ValueError("La suma de pesos debe ser > 0.")
    return w_c / total, w_b / total


def fusionar_probabilidades(prob_cuerpo, prob_cerebro,
                            w_cuerpo: Optional[float] = None,
                            w_cerebro: Optional[float] = None) -> np.ndarray:
    """prob_fusionada = W1 · P_cuerpo + W2 · P_cerebro (5 clases)."""
    prob_cuerpo = np.asarray(prob_cuerpo, dtype=float).reshape(-1)
    if prob_cuerpo.size != 5:
        raise ValueError(f"prob_cuerpo debe tener 5 clases, tiene {prob_cuerpo.size}")
    prob_cerebro = np.asarray(prob_cerebro, dtype=float).reshape(-1)
    if prob_cerebro.size == 4:
        prob_cerebro = mapear_cerebro_a_biovid(prob_cerebro)
    elif prob_cerebro.size != 5:
        raise ValueError(f"prob_cerebro debe tener 4 o 5 clases, tiene {prob_cerebro.size}")

    w_c, w_b = _resolver_pesos(w_cuerpo, w_cerebro)
    return w_c * prob_cuerpo + w_b * prob_cerebro


def score_continuo(prob_fusionada: np.ndarray) -> float:
    """Esperanza matemática de la distribución fusionada -> score en [0,4]."""
    return float(np.sum(np.asarray(prob_fusionada, dtype=float) * config.CLASES_BIOVID))


def entrenar_calibrador_isotonico(predictor: body_engine.PredictorCorporal,
                                  ruta_guardado: Path = config.CALIBRATOR_PATH,
                                  verbose: bool = True) -> IsotonicRegression:
    """Entrena score_fusión(0-4) -> Pain Index(0-10) con scores REALES del
    modelo corporal, con split por sujeto para evitar fuga de datos."""
    if verbose:
        logger.info("Entrenando calibrador isotónico con scores reales del cuerpo...")

    df = predictor.dataset
    subs_calib = df["subject_id"] % 2 == 0
    df_cal = df[subs_calib]

    probs = predictor.modelo.predict_proba(df_cal[predictor.features])
    scores = probs @ config.CLASES_BIOVID
    y_nrs = config.NRS_POR_CLASE_BIOVID[df_cal["class_id"].to_numpy()]

    calibrador = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=10)
    calibrador.fit(scores, y_nrs)
    joblib.dump(calibrador, ruta_guardado)

    if verbose:
        logger.info("Calibrador guardado en %s (n_calib=%d, %d sujetos)",
                    ruta_guardado, len(df_cal), df_cal["subject_id"].nunique())
    return calibrador


class PainIndexFuser:
    """Combina las probabilidades de ambos modelos y devuelve el Pain Index."""

    def __init__(self, ruta_calibrador: Path = config.CALIBRATOR_PATH):
        if not Path(ruta_calibrador).exists():
            raise FileNotFoundError(
                f"No existe el calibrador en {ruta_calibrador}. "
                f"Corre primero entrenar_calibrador_isotonico()."
            )
        self.calibrador = joblib.load(ruta_calibrador)

    def calcular_pain_index(self, prob_cuerpo, prob_cerebro,
                            w_cuerpo: Optional[float] = None,
                            w_cerebro: Optional[float] = None) -> tuple[float, np.ndarray]:
        """Devuelve (pain_index[0-10], prob_fusionada[5])."""
        prob_fusionada = fusionar_probabilidades(prob_cuerpo, prob_cerebro, w_cuerpo, w_cerebro)
        score = score_continuo(prob_fusionada)
        pain_index = float(self.calibrador.predict([score])[0])
        return round(pain_index, 1), prob_fusionada
