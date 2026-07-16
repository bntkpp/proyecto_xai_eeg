# 🧠 Monitor de Dolor en Tiempo Real (XAI)

Sistema de apoyo a la decisión clínica que estima un **Pain Index (0-10)** en tiempo real a partir de la fusión multimodal de señales **EEG** (cerebro) y **biomarcadores autonómicos** (cuerpo: GSR, EMG, ECG), con explicabilidad (XAI) integrada en cada predicción.

> ⚠️ **Aviso clínico:** esta herramienta es un apoyo a la decisión clínica. No reemplaza el juicio profesional ni constituye un diagnóstico.

---

## ✨ Características principales

- **Monitoreo en tiempo real (simulado):** reproducción automática época a época de registros EEG (`.fif`), con panel de control (play/pausa, velocidad).
- **Fusión multimodal (Pain Index):** combina la predicción de un modelo EEGNet (cerebro) y un modelo XGBoost (cuerpo) mediante ponderación ajustable y calibración isotónica → score clínico de 0 a 10.
- **Explicabilidad (XAI) multi-método:**
  - EEG: Integrated Gradients, SHAP (GradientShap), Grad-CAM (canal × tiempo) y LIME por oclusión temporal.
  - Cuerpo: SHAP (TreeExplainer) y LIME tabular sobre biomarcadores.
- **Análisis de coherencia neurofisiológica:** contrasta los 4 métodos XAI de EEG contra literatura del dolor (componentes N2/P300, banda gamma, topografía centro-parietal) para detectar predicciones "correctas por razones espurias".
- **Explicación en lenguaje natural:** panel que traduce automáticamente los resultados técnicos (XAI + coherencia + SHAP corporal) a un texto clínico legible.
- **Historial clínico de sesión:** registro de anotaciones médicas, medicación administrada, alertas por umbral configurable y rachas de dolor sostenido.
- **Reportes exportables:** generación de reportes en **PDF** (multi-página, con logo, resumen, gráfico y tabla de auditoría) y **CSV**.
- **Exploración de datasets:** métricas y visualizaciones de los datos EEG y del dataset autonómico (balance de clases, tendencias fisiológicas, correlaciones).

---

## 🏗️ Arquitectura

Aplicación **Streamlit** con separación estricta entre backend (motores de inferencia/explicabilidad, sin conocimiento de UI) y frontend (orquestación, estado de sesión y componentes visuales).

```
PainPredictNeuro-Dashboard/
├── assets/
│   └── logo.jpeg
├── data/
│   ├── autonomic/
│   │   └── dataset_elite_biovid.csv
│   └── eeg/
│       └── sub-XXX_ses-X_task-95ByBP_eeg_clean-epo.fif
├── models/
│   ├── eegnet_model.pth
│   ├── isotonic_calibrator.pkl
│   └── xgboost_autonomic.json
├── notebooks_model_training/
├── main.py                        # Entrypoint (python main.py)
├── requirements.txt
└── src/
    ├── config.py                  # Rutas, constantes, canales, logging
    ├── backend/
    │   ├── eeg_engine.py          # Inferencia EEGNet + XAI (IG, SHAP, Grad-CAM, LIME)
    │   ├── body_engine.py         # Inferencia XGBoost + SHAP/LIME corporal
    │   ├── fusion_engine.py       # Fusión ponderada + calibración isotónica (Pain Index)
    │   ├── coherencia_engine.py   # Validación neurofisiológica cruzada de XAI
    │   ├── narrativa_engine.py    # Explicación en lenguaje natural
    │   └── reportes_engine.py     # Generación de reportes PDF/CSV
    └── frontend/
        ├── app.py                 # Orquestador principal (Streamlit)
        ├── resources.py           # Carga cacheada de modelos/recursos pesados
        ├── session.py             # Manejo de st.session_state (historial, playback, cache)
        ├── ui_components.py       # Componentes visuales puros (gauge, topomap, heatmaps)
        └── views/
            ├── tab_resumen.py     # Pain Index + explicación en lenguaje natural
            ├── tab_cerebro.py     # XAI detallado de EEG
            ├── tab_cuerpo.py      # XAI detallado corporal
            ├── tab_historial.py   # Historial, anotaciones, alertas, reportes
            └── tab_datasets.py    # Exploración de datasets subyacentes
```

---

## 🔬 Pipeline de fusión (Pain Index)

```
prob_cuerpo (5 clases BioVid) ┐
                               ├─► Fusión ponderada (W_cuerpo, W_cerebro) ─► score continuo [0-4]
prob_cerebro (4 clases NRS)   ┘         (el cerebro se remapea a la grilla BioVid de 5)
                                                          │
                                            Regresión Isotónica (monótona)
                                                          ▼
                                                Pain Index  0.0 – 10.0
```

Los pesos de fusión son ajustables desde la interfaz (slider "Peso del cerebro vs Cuerpo") y también configurables por variables de entorno (`PI_W_CUERPO`, `PI_W_CEREBRO`) vía `.env`.

---

## 📑 Pestañas de la interfaz

| Pestaña | Contenido |
|---|---|
| 📊 **Resumen (clínico)** | Gauge del Pain Index, fusión cuerpo/cerebro, alertas por umbral, explicación en lenguaje natural |
| 🧠 **Cerebro (XAI)** | Predicción EEGNet, topomapa, señal cruda, SHAP + Grad-CAM, LIME-EEG |
| 🫀 **Cuerpo** | Probabilidades fusionadas, SHAP y LIME sobre biomarcadores autonómicos |
| 📈 **Historial y Reportes** | Registro de eventos/medicación, gráfico de evolución, exportación PDF/CSV |
| 📁 **Datasets** | Diagnóstico de los datos EEG y autonómicos cargados (balance de clases, correlaciones) |

---

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd PainPredictNeuro-Dashboard

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Requisitos de datos y modelos

Antes de ejecutar la app, deben existir:

- `models/eegnet_model.pth` — modelo EEGNet entrenado.
- `models/xgboost_autonomic.json` — modelo XGBoost corporal.
- `models/isotonic_calibrator.pkl` — calibrador de fusión (entrenable con `fusion_engine.entrenar_calibrador_isotonico`).
- `data/autonomic/dataset_elite_biovid.csv` — dataset de biomarcadores (BioVid).
- `data/eeg/*.fif` — registros EEG de entrada (opcional: también se admite carga manual `.fif`, `.npy`, `.pt`, `.pth` desde la barra lateral).

---

## ▶️ Uso

```bash
python main.py
```

Esto detecta automáticamente la raíz del proyecto y lanza `src/frontend/app.py` con Streamlit. Alternativamente:

```bash
streamlit run src/frontend/app.py
```

1. Completar los datos del paciente en la barra lateral (Nombre, Edad, Peso, Sexo).
2. Subir un archivo EEG o seleccionar uno de `data/eeg/`.
3. Si es un archivo `.fif`, usar los controles de reproducción (▶/⏸) para recorrer las épocas automáticamente.
4. Ajustar el umbral de alerta y el peso cerebro/cuerpo según necesidad.
5. Explorar las pestañas de XAI, exportar reportes y registrar eventos clínicos.

---

## 🛠️ Stack tecnológico

- **Frontend:** Streamlit
- **EEG:** MNE-Python, PyTorch, Captum (XAI)
- **Cuerpo:** XGBoost, SHAP, LIME, scikit-learn
- **Fusión:** Regresión isotónica (scikit-learn)
- **Reportes:** Matplotlib, ReportLab, Pillow
- **Otros:** NumPy, SciPy, Pandas, Joblib

---


