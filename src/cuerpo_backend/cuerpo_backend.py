import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# ==========================================
# 1. INICIALIZACIÓN GLOBAL (Ejecutado 1 sola vez)
# ==========================================
print("[*] Inicializando Motor Backend XGBoost y SHAP...")

# Rutas robustas respecto a la ubicación de este archivo
_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = _ROOT / "data" / "xgboost_cuerpo.json"
DATASET_PATH = _ROOT / "data" / "dataset_elite_biovid.csv"

# Cargar Modelo XGBoost globalmente
modelo_cuerpo = xgb.XGBClassifier()
modelo_cuerpo.load_model(str(MODEL_PATH))

# Cargar dataset de élite y separar features de metadatos
df_elite = pd.read_csv(DATASET_PATH)
COLS_METADATOS = [
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
]
FEATURES = [c for c in df_elite.columns if c not in COLS_METADATOS]
X_BACKGROUND = df_elite[FEATURES]

# Inicializar TreeSHAP Explainer globalmente (con fondo del dataset de élite)
explainer_cuerpo = shap.TreeExplainer(modelo_cuerpo, data=X_BACKGROUND)

MAPA_DOLOR = {
    0: "Sin Dolor",
    1: "Dolor Leve",
    2: "Dolor Moderado",
    3: "Dolor Severo",
    4: "Dolor Extremo",
}


# ==========================================
# 2. FUNCIÓN LECTORA INDEPENDIENTE (Data Fetcher)
# ==========================================
def obtener_fila_paciente(indice_fila=150):
    """
    Lee el dataset, separa los metadatos y devuelve una ÚNICA fila
    para simular la llegada de datos de un paciente en tiempo real.
    También devuelve la lista de características (features).
    """
    # Extraer solo las features para el modelo
    X_data = X_BACKGROUND.copy()

    # Seleccionar y retornar estrictamente 1 fila como DataFrame
    fila_paciente_df = X_data.iloc[[indice_fila]]

    return fila_paciente_df, FEATURES


# ==========================================
# 3. MOTOR DE INFERENCIA EN TIEMPO REAL
# ==========================================
def procesar_datos_cuerpo(fila_paciente_df=None, indice_fila=150):
    """
    Función orquestadora:
    1. Obtiene 1 fila de paciente (en vivo o del dataset de demo).
    2. Ejecuta la inferencia de XGBoost.
    3. Calcula XAI (SHAP) localmente para ese paciente.
    4. Retorna un diccionario/JSON empaquetado para el frontend.

    Parámetros
    ----------
    fila_paciente_df : pd.DataFrame, optional
        DataFrame de UNA fila con las columnas de FEATURES.
        Si es None, se lee la fila `indice_fila` del dataset de élite.
    indice_fila : int, optional
        Índice de demo cuando no se pasa `fila_paciente_df`.

    Retorna
    -------
    dict
        Estructura JSON con predicción y explicabilidad SHAP.
    """
    # 1. Invocación a la función lectora (o uso de datos en vivo)
    if fila_paciente_df is None:
        fila_paciente_df, features = obtener_fila_paciente(indice_fila)
    else:
        features = FEATURES
        # Asegurar que tenga exactamente las columnas del modelo
        fila_paciente_df = fila_paciente_df[features]

    # 2. Predicción y Probabilidad
    prediccion_num = int(modelo_cuerpo.predict(fila_paciente_df)[0])
    probabilidades = modelo_cuerpo.predict_proba(fila_paciente_df)[0]
    prob_max = probabilidades[prediccion_num]

    estado_dolor = MAPA_DOLOR.get(prediccion_num, "Desconocido")

    # 3. IA Explicable Local (SHAP para este paciente exacto)
    shap_values_local = explainer_cuerpo.shap_values(fila_paciente_df)

    # Manejo de dimensionalidad según versión de SHAP (Multiclase)
    if isinstance(shap_values_local, list):
        shap_clase = shap_values_local[prediccion_num][0]
    elif len(shap_values_local.shape) == 3:
        shap_clase = shap_values_local[0, :, prediccion_num]
    else:
        shap_clase = shap_values_local[0]

    # Crear diccionario de {Variable: Peso SHAP}
    importancia_variables = dict(zip(features, shap_clase))

    # Ordenar variables por su impacto positivo (las que empujan la predicción hacia arriba)
    variables_ordenadas = sorted(
        importancia_variables.items(), key=lambda item: item[1], reverse=True
    )
    top_3_variables = [
        {"biomarcador": k, "peso_shap": round(float(v), 4)}
        for k, v in variables_ordenadas[:3]
    ]

    # 4. Empaquetado Estructurado (JSON-like)
    respuesta_frontend = {
        "prediccion": {
            "nrs_clase": prediccion_num,
            "estado": estado_dolor,
            "certeza_probabilidad": round(float(prob_max) * 100, 2),
        },
        "explicabilidad_shap": {"top_3_drivers": top_3_variables},
    }

    return respuesta_frontend


# ==========================================
# 4. BLOQUE DE PRUEBA (Main)
# ==========================================
if __name__ == "__main__":
    resultado = procesar_datos_cuerpo()

    print("\n[+] OUTPUT GENERADO PARA EL FRONTEND:")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))
