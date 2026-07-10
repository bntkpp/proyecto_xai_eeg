"""
tools/smoke_test_eeg.py
====================================================================
Prueba de humo del motor EEG: confirma que el checkpoint carga y que
el pipeline predicción + XAI corre sin errores.

Reemplaza al viejo bloque `if __name__ == "__main__":` que vivía
dentro de inference_backend.py. Ya NO tiene fallback silencioso a
pesos aleatorios (Fase 0.1 del plan de mejoras): si falta el .pth,
por defecto el script se detiene con error.

Uso normal (requiere el .pth real):
    python src/tools/smoke_test_eeg.py

Modo de desarrollo SIN modelo entrenado (la predicción es BASURA,
solo valida que el código no truena) — hay que pedirlo explícito:
    python src/tools/smoke_test_eeg.py --allow-dummy
====================================================================
"""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from src import config
from src.backend import eeg_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dummy", action="store_true",
                        help="Permite continuar con pesos aleatorios si falta el .pth. "
                             "SOLO para desarrollo, la predicción resultante no es válida.")
    args = parser.parse_args()

    print(f"Device: {eeg_engine.DEVICE}")
    print(f"Modelo esperado en: {config.WEIGHTS_PATH}")

    try:
        modelo = eeg_engine.load_model()
    except FileNotFoundError as e:
        if not args.allow_dummy:
            print(f"\n[ERROR] {e}\n")
            sys.exit(1)
        print("\n" + "!" * 70)
        print("MODO DUMMY: pesos aleatorios. La predicción de abajo es BASURA,")
        print("solo confirma que el código corre sin errores.")
        print("!" * 70 + "\n")
        modelo = eeg_engine.EEGNet().to(eeg_engine.DEVICE).eval()

    dummy = torch.randn(1, 1, config.N_CHANNELS, config.N_SAMPLES)
    resultado = eeg_engine.procesar_onda_eeg(dummy, modelo)
    print(f"Predicción : {resultado.nivel_dolor}  (confianza {resultado.confianza:.0%})")
    print(f"Electrodo  : {config.CH_NAMES[int(np.argmax(resultado.pesos_electrodo))]}")
    print(f"Explicación: {resultado.explicacion}")


if __name__ == "__main__":
    main()
