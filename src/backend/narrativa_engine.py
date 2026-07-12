"""
backend/narrativa_engine.py
====================================================================
Fase 3.5 — Panel de explicación en lenguaje natural.

Arma un texto tipo "El Pain Index es 7.2 porque..." a partir de los
resultados que YA se calcularon en el resto del pipeline (Integrated
Gradients, SHAP, Grad-CAM, LIME, coherencia neurofisiológica, SHAP
corporal). No vuelve a calcular nada — solo sintetiza en texto lo que
otros módulos ya produjeron.

Nota honesta: el ejemplo original del objetivo ("gamma EEG... HRV
disminuyó... GSR mostró 3 SCR...") asume series de tiempo crudas de
ECG/GSR que este proyecto no tiene (ver Fase 3.3, descartada) — el
texto se arma con lo que sí existe: gamma EEG real, XAI multi-método,
y SHAP sobre los biomarcadores agregados del cuerpo.

Degrada elegantemente: si en la época actual todavía no se calcularon
SHAP/Grad-CAM o la coherencia (son bajo demanda, por costo), el texto
usa el nivel de detalle disponible y lo dice explícitamente.
====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src import config

logger = config.get_logger(__name__)


@dataclass(frozen=True)
class ExplicacionNarrativa:
    texto: str            # markdown, listo para st.markdown()
    nivel_detalle: str    # "basico" | "xai" | "coherencia" — cuánta info se usó


def generar_explicacion(*, pi: Optional[float], resultado_eeg, ch_names: Sequence[str],
                        resultado_cuerpo=None, xai_ext=None, coherencia=None) -> ExplicacionNarrativa:
    """Construye la explicación narrativa para la época actual.

    Parámetros opcionales (None si aún no se calcularon en esta época):
      resultado_cuerpo : body_engine.ResultadoCuerpo (con SHAP)
      xai_ext           : eeg_engine.ResultadoXAIExtendido (SHAP + Grad-CAM)
      coherencia         : coherencia_engine.ResultadoCoherencia
    """
    lineas: list[str] = []

    pi_txt = f"**{pi:.1f}/10**" if pi is not None else "**no disponible** (falta la capa de fusión)"
    lineas.append(
        f"### El Pain Index es {pi_txt}\n"
        f"El cerebro clasifica esta época como *{resultado_eeg.nivel_dolor}*, "
        f"con **{resultado_eeg.confianza:.0%}** de confianza del modelo."
    )

    # ---- Explicación cerebral: el nivel de detalle depende de qué se calculó ----
    canal_ig = ch_names[int(np.argmax(resultado_eeg.pesos_electrodo))]

    if coherencia is not None:
        canal_gamma = ch_names[int(np.argmax(coherencia.potencia_gamma_por_canal))]
        n_region_ok = sum(coherencia.region_consistente_por_metodo.values())
        veredicto = "consistente" if n_region_ok >= 2 else "poco consistente"
        lineas.append(
            f"🧠 **Por qué (cerebro):** la mayor potencia gamma real (30-80Hz) de la época se "
            f"registró en el electrodo **{canal_gamma}**. De los 4 métodos de explicabilidad "
            f"(Integrated Gradients, SHAP, Grad-CAM, LIME-oclusión), **{n_region_ok}/4** señalan "
            f"la región central-somatosensorial como la más determinante — {veredicto} con el "
            f"patrón fisiológico esperado de procesamiento del dolor."
        )
        nivel_detalle = "coherencia"
    elif xai_ext is not None:
        canal_shap = ch_names[int(np.argmax(xai_ext.shap_por_canal))]
        relacion = "coincide con" if canal_shap == canal_ig else "difiere de"
        lineas.append(
            f"🧠 **Por qué (cerebro):** el electrodo **{canal_ig}** fue el más relevante según "
            f"Integrated Gradients; SHAP (GradientShap) {relacion} este resultado, señalando a "
            f"**{canal_shap}**. *(Calcula la coherencia neurofisiológica más abajo para contrastar "
            f"esto contra la potencia gamma real y los 4 métodos a la vez.)*"
        )
        nivel_detalle = "xai"
    else:
        lineas.append(
            f"🧠 **Por qué (cerebro):** el electrodo **{canal_ig}** fue el más determinante según "
            f"Integrated Gradients. *(Calcula SHAP + Grad-CAM más abajo para una explicación más "
            f"completa, con validación cruzada entre métodos.)*"
        )
        nivel_detalle = "basico"

    # ---- Explicación corporal ----
    if resultado_cuerpo is not None and resultado_cuerpo.top_drivers:
        top = resultado_cuerpo.top_drivers[0]
        direccion = "aumentó" if top["peso_shap"] > 0 else "disminuyó"
        lineas.append(
            f"🫀 **Por qué (cuerpo):** el biomarcador **{top['biomarcador']}** fue el que más "
            f"{direccion} la probabilidad de la clase **{resultado_cuerpo.estado}** "
            f"(peso SHAP {top['peso_shap']:+.3f}), según el modelo corporal (XGBoost)."
        )
    else:
        lineas.append("🫀 **Por qué (cuerpo):** no hay señal corporal disponible en este momento.")

    return ExplicacionNarrativa(texto="\n\n".join(lineas), nivel_detalle=nivel_detalle)