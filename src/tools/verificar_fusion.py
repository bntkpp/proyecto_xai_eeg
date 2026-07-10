"""
tools/verificar_fusion.py
====================================================================
Batería de verificación (QA) para la capa de Fusión Multimodal (Hito V).
Comprueba INVARIANTES de correctitud, no la validez clínica de los datos.

Correr:  python src/tools/verificar_fusion.py
Cada chequeo imprime [PASS]/[FAIL]. Devuelve exit-code 1 si algo falla.
====================================================================
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np

from src import config
from src.backend.fusion_engine import (
    PainIndexFuser,
    fusionar_probabilidades,
    mapear_cerebro_a_biovid,
)

RNG = np.random.default_rng(0)
_fails: list[str] = []


def check(nombre: str, condicion: bool, detalle: str = "") -> None:
    estado = "PASS" if condicion else "FAIL"
    if not condicion:
        _fails.append(nombre)
    print(f"  [{estado}] {nombre}" + (f"  — {detalle}" if detalle else ""))


def rand_prob(n: int) -> np.ndarray:
    v = RNG.random(n)
    return v / v.sum()


def main() -> None:
    print("=" * 60)
    print("QA CAPA DE FUSIÓN — INVARIANTES DE CORRECTITUD")
    print("=" * 60)

    fuser = PainIndexFuser()
    calib = joblib.load(config.CALIBRATOR_PATH)

    print("\n1. La fusión produce una distribución válida (suma 1, no negativa)")
    ok_sum = ok_neg = True
    for _ in range(2000):
        pc, pb = rand_prob(5), rand_prob(4)
        f = fusionar_probabilidades(pc, pb, RNG.random(), RNG.random())
        ok_sum &= abs(f.sum() - 1) < 1e-9
        ok_neg &= (f >= 0).all()
    check("prob_fusionada suma 1 (2000 casos)", ok_sum)
    check("prob_fusionada no negativa", ok_neg)

    print("\n2. El mapeo cerebro 4->5 conserva masa y centro de masa ordinal")
    ok_mass = ok_pos = True
    for _ in range(2000):
        pb = rand_prob(4)
        m = mapear_cerebro_a_biovid(pb)
        ok_mass &= abs(m.sum() - 1) < 1e-9
        cm_in = np.sum(pb * np.linspace(0, 1, 4))
        cm_out = np.sum(m * np.linspace(0, 1, 5))
        ok_pos &= abs(cm_in - cm_out) < 1e-9
    check("mapeo conserva suma=1", ok_mass)
    check("mapeo conserva centro de masa ordinal", ok_pos)

    print("\n3. Semántica de los pesos")
    pc, pb = rand_prob(5), rand_prob(4)
    f_solo_cuerpo = fusionar_probabilidades(pc, pb, 1.0, 0.0)
    f_solo_cerebro = fusionar_probabilidades(pc, pb, 0.0, 1.0)
    check("peso 100% cuerpo == prob_cuerpo", np.allclose(f_solo_cuerpo, pc))
    check("peso 100% cerebro == mapeo(prob_cerebro)",
          np.allclose(f_solo_cerebro, mapear_cerebro_a_biovid(pb)))
    check("pesos se normalizan (0.7,0.3)==(7,3)",
          np.allclose(fusionar_probabilidades(pc, pb, 0.7, 0.3),
                     fusionar_probabilidades(pc, pb, 7, 3)))

    print("\n4. Monotonía clínica: más dolor => el PI nunca baja")
    xs = np.linspace(0, 4, 500)
    check("calibrador no-decreciente en [0,4]", (np.diff(calib.predict(xs)) >= -1e-9).all())

    onehots5 = np.eye(5)
    onehots4 = np.eye(4)
    ok_body = True
    for pb in onehots4:
        for wb in [0.0, 0.3, 0.5, 0.7, 1.0]:
            pis = [fuser.calcular_pain_index(e, pb, 1 - wb, wb)[0] for e in onehots5]
            ok_body &= all(b - a >= -1e-9 for a, b in zip(pis, pis[1:]))
    check("PI no-decreciente al subir señal CORPORAL (todos los pesos)", ok_body)

    ok_brain = True
    for pc_ in onehots5:
        for wb in [0.0, 0.3, 0.5, 0.7, 1.0]:
            pis = [fuser.calcular_pain_index(pc_, e, 1 - wb, wb)[0] for e in onehots4]
            ok_brain &= all(b - a >= -1e-9 for a, b in zip(pis, pis[1:]))
    check("PI no-decreciente al subir señal CEREBRAL (todos los pesos)", ok_brain)

    print("\n5. Rango [0,10] y robustez a entradas extremas")
    ok_range = True
    for _ in range(2000):
        pi, _ = fuser.calcular_pain_index(rand_prob(5), rand_prob(4), RNG.random(), RNG.random())
        ok_range &= (0 <= pi <= 10)
    check("PI siempre en [0,10] (2000 casos)", ok_range)
    pi_min, _ = fuser.calcular_pain_index(onehots5[0], onehots4[0], 0.5, 0.5)
    pi_max, _ = fuser.calcular_pain_index(onehots5[4], onehots4[3], 0.5, 0.5)
    check("PI(sin dolor) < PI(dolor máximo)", pi_min < pi_max, f"{pi_min} < {pi_max}")

    print("\n6. Reproducibilidad (mismo input => mismo output)")
    pc, pb = rand_prob(5), rand_prob(4)
    r1 = fuser.calcular_pain_index(pc, pb, 0.4, 0.6)[0]
    r2 = fuser.calcular_pain_index(pc, pb, 0.4, 0.6)[0]
    check("dos corridas idénticas dan el mismo PI", r1 == r2)

    print("\n" + "=" * 60)
    if _fails:
        print(f"RESULTADO: {len(_fails)} CHEQUEO(S) FALLARON: {_fails}")
        sys.exit(1)
    print("RESULTADO: TODOS LOS CHEQUEOS PASARON [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
