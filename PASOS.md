# PASOS — Cómo armar y correr el proyecto

> Guía secuencial. Haz los pasos en orden; cada uno tarda pocos minutos. Al final tendrás el monitor médico funcionando con tus archivos `.fif`. Para el detalle conceptual de *por qué* hacemos cada cosa, mira `plan_hitos_xai_eeg.md`.

## Estructura de la carpeta

```
proyecto_xai_eeg/
├── PASOS.md                 ← esta guía
├── plan_hitos_xai_eeg.md    ← el plan de hitos (el "por qué")
├── requirements.txt         ← dependencias
├── eegnet_fold1.pth         ← (PONLO TÚ) el modelo
├── data/                    ← (PONLOS TÚ) tus archivos .fif
│   └── LEEME.txt
└── src/
    ├── inference_backend.py ← motor de IA (predicción + XAI + texto)
    ├── app_streamlit.py     ← el monitor (interfaz web)
    └── leer_canales.py      ← utilidad para inspeccionar un .fif
```

---

## Paso 1 — Instalar dependencias

Abre una terminal **dentro de la carpeta `proyecto_xai_eeg`** y corre:

```bash
pip install -r requirements.txt
```

(Si usas entorno virtual: `python -m venv .venv` y actívalo antes de instalar. Recomendado.)

---

## Paso 2 — Colocar el modelo

1. Toma tu archivo de modelo (lo subiste como `eegnet_fold1 (1).pth`).
2. **Renómbralo a `eegnet_fold1.pth`** y déjalo en la **raíz** de `proyecto_xai_eeg/`.
   - (Si prefieres no renombrar, edita `WEIGHTS_PATH` en `src/inference_backend.py`.)

---

## Paso 3 — Descargar 2-3 archivos `.fif` del Drive

1. De la carpeta `Carpeta_Salida_PIA`, descarga 2 o 3 archivos, p. ej. `sub-069_ses-5_task-95ByBP_eeg_clean-epo.fif`.
2. Cópialos dentro de `data/`.

> Recuerda: cada `.fif` es un paciente/sesión con datos **epocados**. Adentro ya vienen los nombres de los 63 electrodos, el montaje y la frecuencia de muestreo.

---

## Paso 4 — Inspeccionar un `.fif` (verificación clave) ⭐

Esto responde dos preguntas del plan de una vez. Corre:

```bash
python src/leer_canales.py "data/sub-001_ses-1_task-95ByBP_eeg_clean-epo.fif"
```

Fíjate en la salida:

- **`N.º de canales: 63`** → debe ser 63. Si no, hay canales extra (EOG, etc.) que habrá que descartar.
- **`Forma de los datos ... n_muestras`** → este número debe ser **375** (lo que espera el modelo).
  - **Si NO es 375** (p. ej. 1501 o 1000): tus datos están a otra frecuencia. Hay que remuestrear a 250 Hz **antes** de pasarlos al modelo:
    ```python
    epochs.resample(250)     # y/o recortar a 1.5 s con epochs.crop(...)
    ```
    o, si el modelo fue entrenado con ese largo, ajusta `N_SAMPLES` en `src/inference_backend.py`.
- **`¿Trae montaje embebido?`** → si dice `True`, el mapa saldrá perfecto sin tocar nada. Si dice `False`, el código le pone el 10-20 estándar automáticamente.

> Con esto queda resuelto el **Hito 4.3.1** (forma real) y el **Hito 4.3.5** (nombres de canales). Como leemos los nombres del propio `.fif`, **no tienes que escribir `CH_NAMES` a mano**.

---

## Paso 5 — Probar solo el backend (sin interfaz)

Verifica que el modelo carga y que el flujo IA+XAI corre:

```bash
python src/inference_backend.py
```

Debe imprimir `Modelo cargado OK (state_dict calza).` y una predicción de prueba. Si dice que el `state_dict` no calza, avísame (significaría que el `.pth` no es exactamente esta arquitectura).

---

## Paso 6 — Levantar el monitor médico

```bash
streamlit run src/app_streamlit.py
```

Se abre en el navegador. Ahí:

1. En la barra lateral, elige un `.fif` de `data/` (o súbelo).
2. Mueve el slider **"Época a analizar"** para elegir qué segmento de 1.5 s evaluar.
3. Pulsa **Analizar**.
4. Verás: nivel de dolor (con color), confianza %, el **mapa de calor en la cabeza** y el texto clínico.

---

## Paso 7 — Validar que los canales caen en el lugar correcto ⭐

Antes de fiarte clínicamente del mapa, haz esta prueba de cordura (Hito 4.3.5). En una consola de Python dentro del proyecto:

```python
import numpy as np
from src.inference_backend import build_info, CH_NAMES
import mne, matplotlib.pyplot as plt

info = build_info()
pesos = np.zeros(63)
pesos[CH_NAMES.index("Cz")] = 1.0      # enciende SOLO Cz
mne.viz.plot_topomap(pesos, info); plt.show()
```

El punto caliente debe caer en el **centro** de la cabeza. Repite con `"Fp1"` → debe iluminar la **frente izquierda**. Si sale en otro lado, el orden de canales no corresponde (con `.fif` esto no debería pasar, porque usamos los nombres reales del archivo).

---

## Mapa de pasos ↔ hitos del plan

| Paso | Hito del plan |
|---|---|
| 1, 2 | 4.3.1 / 4.3.2 (entorno y carga del modelo) |
| 3, 4 | 4.3.1 / 4.3.5 (datos y canales) |
| 5 | 4.3.3 / 4.3.4 (predicción y XAI) |
| 6 | 4.3.6 / 4.3.7 (texto y frontend) |
| 7 | 4.3.5 / 4.3.9 (verificación y validación) |

---

## Solución de problemas

- **`load_state_dict ... Missing/Unexpected keys`** → el `.pth` no es esta arquitectura exacta, o guardaste un checkpoint completo. Descomenta la línea `state = state.get("model_state_dict", state)` en `inference_backend.py`.
- **`El modelo espera 375 muestras, recibí XXX`** → remuestrea a 250 Hz (`epochs.resample(250)`) o ajusta `N_SAMPLES`. Ver Paso 4.
- **`Esperaba 63 electrodos, recibí 64/65...`** → tu `.fif` trae canales no-EEG. Filtra con `epochs.pick("eeg")` o `epochs.pick_channels([...])`.
- **El topomap sale vacío o plano** → revisa que el montaje exista (Paso 4). El código pone 10-20 si falta.

---

## Para Claude Code

Puedes pegarle este `PASOS.md` y el `plan_hitos_xai_eeg.md` como contexto, y pedirle tareas hito por hito (ej. *"implementa el Hito 4.3.7: mejora el frontend Streamlit según el plan"*). El código base ya está en `src/` para que itere sobre él, no desde cero.
