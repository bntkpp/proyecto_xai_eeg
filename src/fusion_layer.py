"""
fusion_layer.py
====================================================================
HITO V · Actividad 1 — Fusión Multimodal y Calibración del Pain Index (PI)

Une las probabilidades del modelo CORPORAL (XGBoost, 5 clases BioVid) y del
modelo CEREBRAL (EEGNet, 4 clases NRS) en un único Pain Index de 0 a 10.

Pipeline:
    prob_cuerpo(5)  ┐
                    ├─► fusión ponderada (W1,W2) ─► score continuo (esperanza 0-4)
    prob_cerebro(4) ┘        (el cerebro se mapea a la grilla de 5 de BioVid)
                                                        │
                                          Regresión Isotónica (monótona)
                                                        ▼
                                              Pain Index  0.0 – 10.0

Decisión de diseño (desajuste 4 vs 5 clases):
    El cuerpo da 5 clases (BL1,PA1..PA4) y el cerebro 4 (NRS 0-2,4,6,8). Para
    poder SUMAR los vectores (como exige el entregable #1) se mapea el vector
    cerebral de 4→5 por redistribución ORDINAL de probabilidad sobre un eje
    normalizado [0,1] (conserva la suma=1 y el centro de masa). No se
    "hardcodea" qué NRS = qué PA.

Modificabilidad de pesos:
    Los pesos W1 (cuerpo) y W2 (cerebro) se leen, por orden de prioridad:
      1) argumentos de la función  (uso normal desde el frontend)
      2) variables de entorno / archivo .env  (PI_W_CUERPO, PI_W_CEREBRO)
      3) valor por defecto equilibrado 0.5 / 0.5
====================================================================
"""
import os
from pathlib import Path

import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

# --------------------------------------------------------------------
# RUTAS Y CONFIGURACIÓN
# --------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent            # raíz del proyecto
CALIB_PATH = Path(os.getenv("PI_CALIBRADOR",
                            str(_ROOT / "data" / "calibrador_isotonico.pkl")))
_MODELO_CUERPO = _ROOT / "data" / "xgboost_cuerpo.json"
_DATASET_CUERPO = _ROOT / "data" / "dataset_elite_biovid.csv"
_ENV_FILE = _ROOT / ".env"


def _cargar_env(path=_ENV_FILE):
    """Lector .env mínimo (sin dependencias). Solo setea claves aún no definidas."""
    if not path.exists():
        return
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        k, v = linea.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_cargar_env()

# Pesos por defecto (modificables por .env o por argumento). Estado inicial 0.5/0.5.
W_CUERPO_DEFAULT = float(os.getenv("PI_W_CUERPO", "0.5"))
W_CEREBRO_DEFAULT = float(os.getenv("PI_W_CEREBRO", "0.5"))

# Grilla ordinal de clases BioVid (0..4) y sus posiciones normalizadas.
CLASES_BIOVID = np.array([0, 1, 2, 3, 4], dtype=float)
_POS_BIOVID = np.linspace(0.0, 1.0, 5)     # [0, .25, .50, .75, 1]
_POS_CEREBRO = np.linspace(0.0, 1.0, 4)    # [0, .333, .667, 1]

# Mapeo clase BioVid -> NRS clínico (0,2,4,6,8) con la cima EXTENDIDA a 10 para
# completar la escala visual del dashboard (ver enunciado Hito V).
NRS_POR_CLASE_BIOVID = np.array([0.0, 2.5, 5.0, 7.5, 10.0])


# ====================================================================
# 1. LÓGICA DE FUSIÓN
# ====================================================================
def mapear_cerebro_a_biovid(prob_cerebro):
    """Convierte un vector de 4 probabilidades (NRS) en uno de 5 (grilla BioVid)
    redistribuyendo la masa de cada clase a sus dos vecinos por interpolación
    lineal en el eje ordinal [0,1]. Conserva la suma (=1)."""
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
            r = int(np.searchsorted(_POS_BIOVID, pos))    # vecino derecho
            l = r - 1
            frac = (pos - _POS_BIOVID[l]) / (_POS_BIOVID[r] - _POS_BIOVID[l])
            salida[l] += p * (1 - frac)
            salida[r] += p * frac
    return salida


def _resolver_pesos(w_cuerpo, w_cerebro):
    """Aplica prioridad argumento > env > default, valida y normaliza a suma 1."""
    w_c = W_CUERPO_DEFAULT if w_cuerpo is None else float(w_cuerpo)
    w_b = W_CEREBRO_DEFAULT if w_cerebro is None else float(w_cerebro)
    total = w_c + w_b
    if total <= 0:
        raise ValueError("La suma de pesos debe ser > 0.")
    return w_c / total, w_b / total


def fusionar_probabilidades(prob_cuerpo, prob_cerebro,
                            w_cuerpo=None, w_cerebro=None):
    """ENTREGABLE #1 — Fusión ponderada de las dos distribuciones.

    Recibe el vector de 5 probs del cuerpo y el de 4 del cerebro (o ya de 5),
    aplica los pesos configurables y devuelve la probabilidad fusionada (5,).

        prob_fusionada = W1 · P_cuerpo + W2 · P_cerebro
    """
    prob_cuerpo = np.asarray(prob_cuerpo, dtype=float).reshape(-1)
    if prob_cuerpo.size != 5:
        raise ValueError(f"prob_cuerpo debe tener 5 clases, tiene {prob_cuerpo.size}")
    prob_cerebro = np.asarray(prob_cerebro, dtype=float).reshape(-1)
    if prob_cerebro.size == 4:
        prob_cerebro = mapear_cerebro_a_biovid(prob_cerebro)
    elif prob_cerebro.size != 5:
        raise ValueError(f"prob_cerebro debe tener 4 o 5 clases, tiene {prob_cerebro.size}")

    w_c, w_b = _resolver_pesos(w_cuerpo, w_cerebro)
    prob_fusionada = w_c * prob_cuerpo + w_b * prob_cerebro
    return prob_fusionada


def score_continuo(prob_fusionada):
    """Esperanza matemática de la distribución fusionada -> score en [0,4]."""
    prob_fusionada = np.asarray(prob_fusionada, dtype=float).reshape(-1)
    return float(np.sum(prob_fusionada * CLASES_BIOVID))


# --------------------------------------------------------------------
# Helpers ligeros para el dashboard (modelo de cuerpo SIN SHAP)
# --------------------------------------------------------------------
_META_CUERPO = ["subject_id", "subject_name", "class_id", "class_name",
                "sample_id", "sample_name"]


def cargar_recursos_cuerpo():
    """Carga (modelo XGBoost, dataframe, features). Cacheable en el frontend."""
    import pandas as pd
    import xgboost as xgb
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(_MODELO_CUERPO))
    df = pd.read_csv(_DATASET_CUERPO)
    feats = [c for c in df.columns if c not in _META_CUERPO]
    return modelo, df, feats


def probas_cuerpo(recursos, indice_fila=150):
    """Devuelve (prob5, clase_real_nombre) de una fila del dataset BioVid."""
    modelo, df, feats = recursos
    idx = int(indice_fila) % len(df)
    prob = modelo.predict_proba(df[feats].iloc[[idx]])[0]
    return np.asarray(prob, dtype=float), str(df["class_name"].iloc[idx])


# ====================================================================
# 2. CALIBRACIÓN MÉDICA (Regresión Isotónica)
# ====================================================================
def entrenar_calibrador_isotonico(ruta_guardado=CALIB_PATH, verbose=True):
    """Entrena la Regresión Isotónica score_fusión(0-4) -> Pain Index(0-10).

    A diferencia del ejemplo del enunciado (datos simulados), se entrena con
    scores REALES del modelo corporal sobre su validación: el cuerpo vive
    nativamente en la escala BioVid, así que su esperanza es directamente el
    eje que queremos calibrar. Se usa un split por SUJETO para no calibrar
    sobre las mismas muestras (evitar fuga de datos).

    NOTA: idealmente se re-entrena con scores FUSIONADOS reales cuando exista
    dataset multimodal pareado (mismo paciente con cuerpo+cerebro a la vez).
    """
    import pandas as pd
    import xgboost as xgb

    if verbose:
        print("[*] Entrenando calibrador isotónico con scores REALES del cuerpo...")

    modelo = xgb.XGBClassifier()
    modelo.load_model(str(_MODELO_CUERPO))
    df = pd.read_csv(_DATASET_CUERPO)
    meta = ["subject_id", "subject_name", "class_id", "class_name",
            "sample_id", "sample_name"]
    feats = [c for c in df.columns if c not in meta]

    # Split por sujeto: sujetos con id par -> calibración
    subs_calib = df["subject_id"] % 2 == 0
    df_cal = df[subs_calib]

    probs = modelo.predict_proba(df_cal[feats])           # (n,5)
    scores = probs @ CLASES_BIOVID                        # esperanza (n,) en [0,4]
    y_nrs = NRS_POR_CLASE_BIOVID[df_cal["class_id"].to_numpy()]   # objetivo 0-10

    calibrador = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=10)
    calibrador.fit(scores, y_nrs)
    joblib.dump(calibrador, ruta_guardado)

    if verbose:
        print(f"[+] Calibrador guardado en '{ruta_guardado}'  "
              f"(n_calib={len(df_cal)}, {df_cal['subject_id'].nunique()} sujetos)")
    return calibrador


# ====================================================================
# 3. MOTOR DE FUSIÓN (Inferencia en tiempo real)
# ====================================================================
class PainIndexFuser:
    """Combina las probabilidades de ambos modelos y devuelve el Pain Index."""

    def __init__(self, ruta_calibrador=CALIB_PATH):
        if not Path(ruta_calibrador).exists():
            raise FileNotFoundError(
                f"No existe el calibrador en {ruta_calibrador}. "
                f"Corre primero entrenar_calibrador_isotonico().")
        self.calibrador = joblib.load(ruta_calibrador)

    def calcular_pain_index(self, prob_cuerpo, prob_cerebro,
                            w_cuerpo=None, w_cerebro=None):
        """Devuelve (pain_index[0-10], prob_fusionada[5])."""
        prob_fusionada = fusionar_probabilidades(
            prob_cuerpo, prob_cerebro, w_cuerpo, w_cerebro)
        score = score_continuo(prob_fusionada)
        pain_index = float(self.calibrador.predict([score])[0])
        return round(pain_index, 1), prob_fusionada


# ====================================================================
# 4. DEMOSTRACIÓN (pesos variables + estabilidad)
# ====================================================================
if __name__ == "__main__":
    entrenar_calibrador_isotonico()
    motor = PainIndexFuser()

    # Cuerpo (5 clases BioVid) y Cerebro (4 clases NRS)
    probs_xgboost = [0.05, 0.10, 0.45, 0.30, 0.10]     # dolor clase 2-3
    probs_eegnet  = [0.02, 0.10, 0.28, 0.60]           # cerebro: NRS alto

    print("\n--- PRUEBA DE PESOS VARIABLES (sin recompilar modelos) ---")
    for wc, wb in [(0.5, 0.5), (0.3, 0.7), (0.8, 0.2)]:
        pi, fus = motor.calcular_pain_index(probs_xgboost, probs_eegnet, wc, wb)
        print(f"  W_cuerpo={wc:.1f} | W_cerebro={wb:.1f}  ->  PI = {pi:>4} / 10   "
              f"(fus={np.round(fus,3)})")
