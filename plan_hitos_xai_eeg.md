# Plan de Hitos — Hito 4 · Actividad 3

## Implementación de XAI y Mapeo Cerebral en Tiempo Real (EEGNet + Captum + MNE + Streamlit)

> **Para qué sirve este documento.** Es tu mapa de ruta. Cada hito te dice *qué hacemos*, *por qué lo hacemos* (en lenguaje simple, para que entiendas la lógica clínica y técnica) y *cómo sabrás que quedó bien*. Se lo puedes entregar a Claude Code hito por hito. El código base ya adaptado a tu modelo real está en `inference_backend.py` y `app_streamlit.py`.

---

## 0. Resumen ejecutivo: qué vamos a construir

Vamos a convertir un script de IA en un **producto médico de dos piezas**:

1. **Backend (Motor de Inferencia).** Una función que recibe 1.5 s de EEG crudo (un tensor), lo pasa por el modelo `.pth`, y devuelve tres cosas: el **nivel de dolor** predicho, un **vector de 63 pesos** (cuánto pesó cada electrodo en la decisión) y un **texto clínico** automático.
2. **Frontend (Monitor en Streamlit).** Una interfaz web que toma esos 63 números y dibuja el **mapa de calor en la cabeza** (topomap) más el texto explicativo, para que el médico *vea* dónde el cerebro está procesando el dolor y confíe (o desconfíe) de la alerta.

La idea central: **un número no basta**. El médico necesita el mapa para confiar.

---

## 1. Lo que YA verificamos del modelo (leyendo el `.pth` real)

Antes de planear, abrí tu archivo `eegnet_fold1 (1).pth` y leí los pesos. Esto es lo que **realmente** contiene (no son supuestos):

| Característica | Valor real | De dónde sale |
|---|---|---|
| N.º de electrodos (canales) | **63** ✓ | `spatial_conv.0.weight` = (16, 1, **63**, 1) |
| N.º de clases | **4** ✓ | `classifier.1.weight` = (**4**, 176) |
| Tipo de arquitectura | EEGNet (F1=8, D=2, F2=16) | Forma de todas las capas |
| Nombres de los módulos | `temporal_conv`, `spatial_conv`, `separable_conv`, `classifier` | Llaves del `state_dict` |
| Kernel temporal | 125 muestras | `temporal_conv.0.weight` = (8,1,1,**125**) |
| Entrada de muestras (tiempo) | **≈ 375** (no 1501) | El clasificador es `Linear(176→4)` y 176 = 16 × 11 ⇒ la red solo acepta una longitud fija |
| Frecuencia de muestreo implícita | **≈ 250 Hz** | Kernel 125 = fs/2; 1.5 s × 250 Hz = 375 muestras |

### ⚠️ Tres correcciones importantes frente al template que te pasaron

1. **La forma de entrada NO es `(1,1,63,1501)`.** El clasificador está cableado a 176 features (16 mapas × 11 instantes de tiempo). Eso solo se produce con **~375 muestras** de entrada (1.5 s a 250 Hz), no 1501. Si le metes 1501, el modelo va a **fallar**. → En el Hito 4.3.1 confirmamos la forma real leyendo un tensor del Drive.

2. **Grad-CAM / torchcam NO te puede dar 63 pesos por electrodo.** El `spatial_conv` tiene kernel `(63,1)`: *colapsa* los 63 electrodos en uno solo. Después de esa capa, la dimensión "electrodo" **ya no existe**. Por eso, para obtener un peso por electrodo, lo correcto es **atribución sobre la entrada** (Integrated Gradients de Captum, justo como en tu template). Tu template eligió bien el método; lo dejamos así y descartamos torchcam para esta parte.

3. **No confíes en los índices fijos de canal (14, 15, 16…).** El template asume "índice 15 = Cz". Eso solo es cierto si tus tensores guardaron los electrodos en ese orden exacto. El orden real lo define tu preprocesamiento. → Mapearemos **por nombre** (`'Cz'`), no por número. Esta es la parte que pediste corroborar y está en el Hito 4.3.5.

---

## 2. Arquitectura del producto (cómo se separan backend y frontend)

```
                 ┌─────────────────────────────────────────────┐
   Tensor EEG    │  BACKEND  (inference_backend.py)             │
   (1,1,63,375)  │                                             │
   ───────────►  │  modelo .pth  ──►  predicción (clase)        │
                 │       │                                       │
                 │       └──►  Integrated Gradients ──► 63 pesos │
                 │                    │                          │
                 │                    └──►  lógica de texto       │
                 └───────────────┬─────────────────────────────┘
                                 │  (nivel_dolor, pesos_63, texto)
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │  FRONTEND  (app_streamlit.py)                │
                 │  mne.viz.plot_topomap(63 pesos, info 10-20)  │
                 │  + nivel de dolor + texto clínico            │
                 └─────────────────────────────────────────────┘
```

**Regla de oro (la "Nota de Arquitectura Crítica" de tu enunciado):** el modelo `.pth` se carga **una sola vez** al levantar la app (con `@st.cache_resource` en Streamlit), nunca dentro de la función que procesa cada onda. Si lo recargas por cada segmento de EEG, el sistema se congela.

---

## 3. Plan de hitos

Cada sub-hito está pensado para entregárselo a Claude Code como una tarea independiente. Llévalos en orden: cada uno depende del anterior.

### Hito 4.3.1 — Preparación del entorno y verificación de los datos

**Objetivo:** tener el entorno listo y *confirmar la forma real* de los tensores.

**Tareas:**

1. Crear entorno e instalar dependencias (`requirements.txt`): `torch`, `captum`, `mne`, `streamlit`, `numpy`, `matplotlib`.
2. Descargar 2-3 tensores de pacientes del Drive a una carpeta local `data/`.
3. Cargar un tensor y hacer `print(tensor.shape)`. **Anotar la forma exacta** (esperado: algo como `(63, 375)` o `(1,1,63,375)`).
4. Confirmar el rango de valores (¿está normalizado? ¿en microvoltios?).

**Qué hacemos y por qué (para ti):** antes de tocar IA, nos aseguramos de que la "materia prima" (el tensor) tiene la forma que el modelo espera. El modelo es exigente: solo traga una longitud exacta de tiempo. Si los tensores vienen como `(63, 375)` tendremos que añadir dos dimensiones para dejarlos como `(1, 1, 63, 375)` (lote y "canal de imagen"), que es como una CNN espera los datos.

**Entregable:** `requirements.txt` + una nota con la forma real del tensor.

**Criterio de aceptación:** sé con certeza cuántas muestras de tiempo tiene un segmento y en qué orden vienen las dimensiones.

---

### Hito 4.3.2 — Reconstrucción y carga del modelo (global, en modo evaluación)

**Objetivo:** cargar `eegnet_fold1.pth` sin errores y dejarlo listo para predecir.

**Tareas:**

1. Definir la clase `EEGNet` con la arquitectura **exacta** (ya reconstruida en `inference_backend.py` a partir de tus pesos).
2. Instanciar `EEGNet(n_classes=4, channels=63, samples=375)`.
3. `model.load_state_dict(torch.load('eegnet_fold1.pth', map_location=device))` con `strict=True`.
4. `model.eval()` (apaga Dropout y BatchNorm — **obligatorio** para que la predicción sea estable y reproducible).
5. Cargar todo esto **a nivel global / cacheado**, una sola vez.

**Qué hacemos y por qué (para ti):** un archivo `.pth` solo guarda los *números* (los pesos), no la *forma* de la red. Para usarlos hay que reconstruir la misma arquitectura y "verter" los pesos dentro. Yo ya leí tu archivo y reconstruí la clase para que calce exacto; si calza, `load_state_dict` no se queja. `model.eval()` es clave: en modo entrenamiento la red mete aleatoriedad (Dropout) y usaría estadísticas del lote; en modo evaluación queda determinista, que es lo que un equipo médico necesita.

**Entregable:** modelo cargado que imprime su arquitectura sin error.

**Criterio de aceptación:** `load_state_dict` retorna `<All keys matched successfully>`.

---

### Hito 4.3.3 — Motor de inferencia: la predicción

**Objetivo:** función que recibe un tensor y devuelve el nivel de dolor.

**Tareas:**

1. Dentro de `procesar_onda_eeg`, mover el tensor al `device`.
2. `with torch.no_grad():` → `out = model(tensor)` → `clase = out.argmax(1).item()`.
3. Traducir el índice de clase a texto clínico con el diccionario `MAPA_DOLOR` (0=Sin dolor … 3=Severo).
4. (Recomendado) calcular también `softmax(out)` para mostrar la **confianza** del modelo (%) — un médico querrá saber si la red está 95% segura o apenas 40%.

**Qué hacemos y por qué (para ti):** esta es la parte "fácil": pasar la onda por la red y leer cuál de las 4 salidas ganó. Usamos `no_grad` porque para *predecir* no necesitamos calcular gradientes; ahorra memoria y va más rápido. La confianza (softmax) es un extra que vuelve la alerta mucho más útil clínicamente.

**Entregable:** la función retorna correctamente "Dolor Moderado (NRS 6)", etc.

**Criterio de aceptación:** misma entrada → misma salida siempre (gracias a `eval()`).

---

### Hito 4.3.4 — Extracción de saliencia XAI (Integrated Gradients → 63 pesos)

**Objetivo:** obtener el vector de 63 números (importancia de cada electrodo).

**Tareas:**

1. `ig = IntegratedGradients(model)`.
2. Pasar el tensor de entrada con `requires_grad` activo.
3. `atribuciones = ig.attribute(tensor, target=clase_predicha)` → forma `(1,1,63,T)`.
4. Reducir a 63 valores: valor absoluto, promediar a lo largo del tiempo → `mean(dim=3)` → vector `(63,)`.
5. Normalizar 0–1 para visualizar.

**Qué hacemos y por qué (para ti):** "saliencia" = qué partes de la entrada empujaron la decisión. Integrated Gradients pregunta, matemáticamente: *"si apago gradualmente este electrodo, ¿cuánto cambia la predicción?"*. Tomamos valor absoluto (nos importa la magnitud de la influencia, no el signo) y promediamos en el tiempo porque queremos **un solo número por electrodo**, no por instante. Resultado: 63 pesos, perfectos para pintar el mapa.

**Nota técnica:** la predicción del Hito 4.3.3 va `no_grad`; **esta** parte NO puede ir en `no_grad` (IG necesita gradientes). En el código están separadas correctamente. IG corre ~50 pasadas por defecto: para "tiempo real" considera bajar `n_steps` (ej. 25) si la latencia molesta.

**Entregable:** `pesos_63_canales` con `shape == (63,)`, valores entre 0 y 1.

**Criterio de aceptación:** `assert pesos.shape == (63,)` pasa, y los pesos cambian si cambias el tensor de entrada (no son constantes).

---

### Hito 4.3.5 — Mapeo de canales por NOMBRE + montaje 10-20 (la verificación que pediste) ⭐

**Objetivo:** garantizar que cada uno de los 63 pesos se dibuja en el electrodo correcto de la cabeza, y que las reglas de "región" usan el electrodo real.

**Tareas:**

1. Conseguir la **lista oficial de los 63 nombres de electrodos en orden** (la que usó tu preprocesamiento). *Tú dijiste que la tienes / la consigues* — pégala en `CH_NAMES` dentro de `inference_backend.py`.
2. Construir `info = mne.create_info(CH_NAMES, sfreq=250, ch_types='eeg')`.
3. `info.set_montage(mne.channels.make_standard_montage('standard_1020'))` (o `'standard_1005'` si tu set de 63 es de alta densidad). Esto le da a MNE las coordenadas X/Y de cada electrodo en el cuero cabelludo.
4. Definir los **grupos por región usando nombres** (no índices): central/somatosensorial, frontal/ocular, etc. (ver tabla del punto 4).
5. Validar: dibujar un topomap de prueba donde un solo electrodo conocido (ej. `Cz`) valga 1 y el resto 0; confirmar que el punto caliente cae en el centro de la cabeza.

**Qué hacemos y por qué (para ti):** este es el corazón de tu pregunta *"corroboren bien los canales por área del cerebro"*. El error más común y peligroso aquí es asumir que "el electrodo número 15 es Cz". Eso depende 100% del orden en que tu pipeline guardó los datos. Si el orden está corrido, el mapa mostraría dolor en la frente cuando en realidad está en el centro → **alerta clínica falsa**. La solución robusta: trabajar con **nombres**. MNE coloca cada nombre en su coordenada real, sin importar el orden del arreglo, y nuestras reglas de texto buscan `CH_NAMES.index('Cz')` en vez de un número mágico.

**Entregable:** `info` con montaje válido; test del electrodo único pasa.

**Criterio de aceptación:** poner peso 1 en `'Cz'` ilumina el centro; poner peso 1 en `'Fp1'` ilumina la frente izquierda. Si esto falla, el orden de `CH_NAMES` está mal.

---

### Hito 4.3.6 — Generación de explicación textual clínica

**Objetivo:** texto automático que interprete el electrodo dominante.

**Tareas:**

1. Hallar el electrodo de mayor peso: `nombre_max = CH_NAMES[np.argmax(pesos)]`.
2. Aplicar reglas **por nombre** (no índice):
   - Si `nombre_max` ∈ grupo **central/somatosensorial** → "Activación somatosensorial: patrón fisiológico válido de procesamiento del dolor."
   - Si ∈ grupo **frontal/ocular** → "⚠️ Posible artefacto ocular/parpadeo o actividad cognitiva — revisar antes de confiar."
   - Si ∈ grupo **temporal** → "⚠️ Posible artefacto muscular (EMG)."
   - Otro → "Activación en región periférica (electrodo {nombre_max}); interpretar con cautela."

**Qué hacemos y por qué (para ti):** traducimos números a una frase que un médico entienda en 2 segundos. La distinción importante: si lo que "enciende" la predicción está en el centro de la cabeza (córtex somatosensorial), es señal *real* de dolor; si está en la frente, probablemente es un **parpadeo** disfrazado de señal, y el sistema debe advertirlo en vez de generar una falsa alarma. Esto es lo que hace que el médico confíe en la herramienta.

**Entregable:** función que devuelve el string correcto según la región.

**Criterio de aceptación:** forzando el peso máximo en `Cz` sale "somatosensorial"; en `Fp1` sale la advertencia ocular.

---

### Hito 4.3.7 — Frontend Streamlit + topomap dinámico

**Objetivo:** la interfaz médica que junta todo.

**Tareas:**

1. Cargar el modelo una vez con `@st.cache_resource` (cumple la nota de arquitectura).
2. Widget para elegir/cargar un tensor de paciente.
3. Botón "Analizar" → llama a `procesar_onda_eeg`.
4. Mostrar: nivel de dolor (grande, con color), confianza %, el **topomap** (`mne.viz.plot_topomap`) y el texto clínico.
5. Estética de UCI: colores claros, semáforo (verde/amarillo/rojo) según severidad o según si hay advertencia de artefacto.

**Qué hacemos y por qué (para ti):** aquí se vuelve "producto". Streamlit nos deja construir la web sin saber desarrollo web. El `cache_resource` es lo que evita que el modelo se recargue en cada clic. El topomap es la pieza que convierte 63 números fríos en una imagen que el médico interpreta de un vistazo.

**Entregable:** `streamlit run app_streamlit.py` abre el monitor y dibuja el mapa.

**Criterio de aceptación:** cambiar de paciente cambia el mapa y el texto, sin recargar el modelo (revisar que no haya lag de carga repetida).

---

### Hito 4.3.8 — Integración, latencia y pruebas

**Objetivo:** que funcione de punta a punta y rápido.

**Tareas:**

1. Probar los 2-3 tensores descargados; revisar que cada uno produzca mapa + texto coherentes.
2. Medir el tiempo de `procesar_onda_eeg` (objetivo: < 1 s por segmento para sensación de "tiempo real"). Si es lento, bajar `n_steps` de IG.
3. Manejo de errores: tensor con forma equivocada → mensaje claro, no un crash.
4. Caso borde: ¿qué pasa si todos los pesos son iguales? (normalización divide por cero — ya cubierto en el código).

**Qué hacemos y por qué (para ti):** "tiempo real" significa que el médico no espera. Medimos y optimizamos. También blindamos contra entradas raras para que la herramienta no se caiga en plena UCI.

**Criterio de aceptación:** los 3 pacientes corren sin error y bajo el umbral de latencia.

---

### Hito 4.3.9 — Validación final y entrega

**Objetivo:** cerrar con evidencia.

**Tareas:**

1. Capturas del monitor para cada nivel de dolor (0–3).
2. Revisión cruzada del Hito 4.3.5 (test de electrodo único) documentada.
3. README corto: cómo instalar y correr.
4. (Opcional) breve descargo: es una herramienta de *apoyo a la decisión*, no reemplaza juicio clínico.

**Criterio de aceptación:** un tercero puede clonar, instalar y correr siguiendo el README.

---

## 4. Verificación neuroanatómica de canales por área ⭐

Esto responde directo a *"corroboren bien si los canales seleccionados para cada área del cerebro"*. Tu template usaba `C3,Cz,C4` para "central" y `Fp1,Fpz,Fp2` para "frontal". Está en la dirección correcta, pero conviene ampliarlo y justificarlo. Grupos sugeridos (Sistema 10-10):

| Región funcional | Electrodos sugeridos | Qué significa si domina aquí |
|---|---|---|
| **Somatosensorial / central (S1)** — procesamiento real del dolor | `C3, C1, Cz, C2, C4, FCz, CP1, CPz, CP2` | ✅ Patrón fisiológico válido: corteza somatosensorial primaria. Es la señal que *queremos* ver en dolor. |
| **Frontal / ocular** — artefactos | `Fp1, Fpz, Fp2, AF7, AF3, AF4, AF8` | ⚠️ Parpadeo / movimiento ocular o actividad cognitiva frontal. Sospechar artefacto antes de confiar. |
| **Temporal / muscular** — artefactos EMG | `T7, T8, FT7, FT8, TP7, TP8` | ⚠️ Tensión muscular (mandíbula/cuello). Alta frecuencia, no es dolor cortical. |
| **Occipital / parietal posterior** | `O1, Oz, O2, POz, PO3, PO4` | Actividad visual/alfa. Inesperado en una tarea de dolor → revisar. |

**Notas clínicas para afinar:**

- El dolor es **contralateral**: un estímulo en la mano derecha activa S1 *izquierdo* (electrodos `C3/CP3`). Si tu protocolo lateraliza el estímulo, puedes refinar la regla por hemisferio.
- La corteza cingulada anterior (componente afectivo del dolor) proyecta a la línea media frontocentral (`FCz/Cz`); por eso `FCz` está en el grupo somatosensorial y no en "frontal-ocular".
- **Lo más importante:** estos nombres solo sirven si `CH_NAMES` está en el orden real de tus tensores. Por eso el Hito 4.3.5 (mapear por nombre + test de electrodo único) es obligatorio antes de fiarte de cualquier interpretación.

---

## 5. Riesgos y notas críticas

1. **Longitud de entrada fija (~375, no 1501).** Verifícalo en el Hito 4.3.1. Es la causa #1 de que el modelo "no cargue" o "no prediga".
2. **Orden de canales.** Riesgo #1 de un mapa engañoso. Mitigado mapeando por nombre.
3. **No uses torchcam para los 63 pesos** (el `spatial_conv` colapsa los electrodos). Usa Integrated Gradients sobre la entrada, como ya está en el código.
4. **Carga global del modelo.** Nunca dentro de `procesar_onda_eeg`. En Streamlit, `@st.cache_resource`.
5. **`model.eval()` siempre** antes de inferir.
6. **Latencia de IG.** ~50 pasadas por segmento; baja `n_steps` si necesitas más velocidad.

---

## 6. Cómo correr el proyecto

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar el modelo y los tensores
#    eegnet_fold1.pth  +  carpeta data/ con los tensores del Drive

# 3. (Opcional) probar solo el backend
python inference_backend.py

# 4. Levantar el monitor
streamlit run app_streamlit.py
```

**Archivos del proyecto:**

- `inference_backend.py` — modelo + carga + `procesar_onda_eeg` (predicción, XAI, texto). Ya adaptado a tu `.pth` real.
- `app_streamlit.py` — el monitor médico (frontend).
- `requirements.txt` — dependencias.

> Recuerda: lo único que **debes** completar a mano es la lista `CH_NAMES` con los 63 nombres en el orden real de tus tensores (Hito 4.3.5). Todo lo demás está listo para correr.
