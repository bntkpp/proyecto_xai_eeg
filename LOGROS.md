# LOGROS — Puesta en marcha del Monitor de Dolor XAI (EEG)

> Bitácora de lo que quedó **funcionando** en el proyecto `proyecto_xai_eeg`, con los problemas reales que se encontraron en los datos y cómo se resolvieron. Mapeado a los hitos del `plan_hitos_xai_eeg.md`.
>
> Fecha: 2026-06-30

---

## Resumen en una línea

Se dejó operativo el pipeline completo **EEG → predicción de dolor (EEGNet) → explicabilidad (Integrated Gradients) → mapa cerebral (MNE) → texto clínico**, adaptándolo a las características reales de los archivos `.fif` del paciente (1000 Hz y canales EOG incluidos).

---

## 1. Entorno y dependencias ✅ (Hitos 4.3.1 / 4.3.2)

- Entorno virtual `.venv` creado y activado en Windows/PowerShell.
- Dependencias instaladas desde `requirements.txt`.
- Ejecución confirmada en **CPU** (`Device: cpu`).

---

## 2. Carga del modelo verificada ✅ (Hito 4.3.2 / 4.3.3)

Al correr el backend:

```powershell
python src/inference_backend.py
```

salida:

```
Modelo cargado OK (state_dict calza).
```

Esto confirma que la arquitectura **EEGNet reconstruida** en `src/inference_backend.py` coincide **exactamente** con el checkpoint `eegnet_fold1.pth` (`load_state_dict(..., strict=True)` sin errores). Características verificadas del modelo:

| Parámetro | Valor |
|---|---|
| Electrodos de entrada | 63 |
| Clases de salida | 4 (Sin dolor / Leve / Moderado / Severo) |
| Frecuencia de muestreo de entrenamiento | 250 Hz |
| Muestras de tiempo por época | 375 (1.5 s @ 250 Hz) |

> Nota: el `UserWarning` de `padding='same' with even kernel lengths` es inofensivo (PyTorch hace una copia interna del input). No afecta el resultado.

---

## 3. Inspección de los datos reales `.fif` ✅ (Hitos 4.3.1 / 4.3.5)

Con `src/leer_canales.py` se inspeccionó un archivo real
(`data/sub-001_ses-1_task-95ByBP_eeg_clean-epo.fif`) y se descubrieron **dos diferencias clave** entre los datos del paciente y lo que el modelo espera:

| Característica | Archivo del paciente | Esperado por el modelo | Acción |
|---|---|---|---|
| N.º de canales | 63 | 63 | OK (con matiz, ver §4) |
| Muestras por época | **1501** | **375** | Remuestreo (ver §4) |
| Frecuencia de muestreo | **1000 Hz** | **250 Hz** | Remuestreo (ver §4) |
| Montaje embebido | No | — | El código aplica 10-20 estándar |
| N.º de épocas | 26 (índices 0–25) | — | Seleccionables por slider |

---

## 4. Problemas encontrados y resueltos

### 4.1 Frecuencia de muestreo incorrecta (1000 Hz → 250 Hz) ✅

**Problema:** los datos venían a 1000 Hz con 1501 muestras, pero el modelo fue entrenado a 250 Hz / 375 muestras. La cuenta calza: `1501 ÷ 4 ≈ 375`.

**Solución:** se modificó `leer_fif()` en `src/inference_backend.py` para **remuestrear automáticamente** a 250 Hz cuando la frecuencia no coincide, y recortar a exactamente 375 muestras:

```python
if round(epochs.info["sfreq"]) != SFREQ:
    epochs.resample(SFREQ)
# ... y se asegura que queden EXACTAMENTE N_SAMPLES (375) muestras
```

Resultado: el monitor ahora carga el `.fif` como **63 canales · 250 Hz** y la inferencia ya no falla por tamaño de entrada. No fue necesario tocar `N_SAMPLES`.

### 4.2 Canales no cerebrales (EOG) rompían el mapa de calor ✅

**Problema:** de los 63 canales, dos **no son electrodos del cuero cabelludo**:

- `VEO` → EOG vertical (parpadeo)
- `HEOR` → EOG horizontal derecho

Es decir: **61 EEG + 2 EOG = 63**. Como el total da 63, el modelo corre sin problema, pero al dibujar el topomap MNE les asignaba posición (0,0) y fallaba con:

```
ValueError: The following electrodes have overlapping positions ...: VEO, HEOR
```

**Solución:** en `src/app_streamlit.py` se agregó `_solo_eeg_para_topomap()`, que **excluye los canales no cerebrales únicamente al graficar** (el modelo sigue usando los 63 para predecir). Filtra EOG/EMG/ECG/mastoides por tipo y por nombre.

Resultado: el mapa de calor se dibuja correctamente sobre los ~61 electrodos válidos.

---

## 5. Estado del pipeline end-to-end ✅

Flujo confirmado funcionando de punta a punta:

```
.fif (1000 Hz)
   └─ leer_fif()  → remuestreo a 250 Hz / 375 muestras
        └─ EEGNet  → predicción de nivel de dolor + confianza
             └─ Integrated Gradients (Captum)  → 63 pesos por electrodo
                  └─ MNE topomap  → mapa de calor en la cabeza (solo EEG)
                       └─ texto clínico por región (somatosensorial / ocular / muscular)
```

Interfaz Streamlit operativa:

```powershell
streamlit run src/app_streamlit.py
```

con selección de archivo, slider de época (0–25), botón **Analizar**, tarjeta de nivel de dolor con color, confianza, topomap y explicación clínica.

---

## 6. Interpretación clínica útil (efecto secundario positivo)

Gracias a que los canales EOG se conservan en la predicción, si el **"Electrodo dominante"** resulta ser `VEO` o `HEOR`, eso indica que un **artefacto ocular** está dominando la decisión del modelo → esa época debe descartarse o reemplazarse con el slider. Es una señal de control de calidad, no un error.

---

## 7. Pendientes / próximos pasos

- [ ] **Confirmar el entrenamiento:** verificar que `eegnet_fold1.pth` fue entrenado con estos mismos 63 canales (incluyendo `VEO`/`HEOR`) y en este mismo orden. El código corre igual, pero la validez clínica depende de que el orden de canales coincida con el del entrenamiento.
- [ ] (Opcional) Excluir también los canales no cerebrales del cálculo de **"Electrodo dominante"** y del texto clínico, para que solo reporten electrodos reales del cuero cabelludo.
- [ ] **Validación de cordura del montaje (Paso 7 / Hito 4.3.5):** encender un solo electrodo (ej. `Cz`, `Fp1`) y confirmar que el punto caliente cae donde corresponde.

---

## Archivos tocados en esta sesión

| Archivo | Cambio |
|---|---|
| `src/inference_backend.py` | `leer_fif()` ahora remuestrea a 250 Hz y recorta a 375 muestras |
| `src/app_streamlit.py` | Nuevo `_solo_eeg_para_topomap()`: el topomap excluye canales no cerebrales |
