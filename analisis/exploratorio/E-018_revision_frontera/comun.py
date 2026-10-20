# E-018 (EXPLORATORIO, post hoc; 2026-10-19): revisión adversaria de E-017 (A-026), hecha por un agente independiente
# y versionada aquí sin cambios de cálculo (solo rutas). Ejecutar desde esta carpeta: python ataqueN_*.py; salida y stdout en
# resultados/2026-10-19_E-018/.
import sys, json
from pathlib import Path
R = Path(__file__).resolve().parents[3]
OUT = R/'resultados'/'2026-10-19_E-018'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R/'analisis'))
import numpy as np, pandas as pd
from bms_comun import cargar_datos
UMB = 5.93
df = cargar_datos()
A_ALL, B_ALL = 3.8845550541124774, 0.046770919640184046
def fr_MA(wt, a=A_ALL, b=B_ALL):
    den = a - b/wt
    return np.where(den > 0, np.exp(UMB/den) - wt, np.inf)
def MA(wt, wc, a=A_ALL, b=B_ALL): return (a - b/wt)*np.log(wc + wt)
def primer_cruce(wc, t, u=UMB):
    for i in range(1, len(wc)):
        if t[i-1] < u <= t[i]:
            return wc[i-1] + (u - t[i-1])*(wc[i] - wc[i-1])/(t[i] - t[i-1])
    return np.nan
def ultimo_cruce(wc, t, u=UMB):
    # último cruce ascendente (a partir de él todo >= u)
    for i in range(len(wc)-1, 0, -1):
        if t[i-1] < u <= t[i]:
            return wc[i-1] + (u - t[i-1])*(wc[i] - wc[i-1])/(t[i] - t[i-1])
    return np.nan
def pava(y, w=None):
    y = np.asarray(y, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    bl = [[y[i], w[i], 1] for i in range(len(y))]
    out = []
    for b in bl:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, n2 = out.pop(); v1, w1, n1 = out.pop()
            out.append([(v1*w1 + v2*w2)/(w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[v]*n for v, _, n in out])
def celdas(d):
    return d.groupby(['cyc', 'nivel']).agg(WT=('WT', 'first'), WC=('WC', 'mean'), T=('T', 'mean')).reset_index()
