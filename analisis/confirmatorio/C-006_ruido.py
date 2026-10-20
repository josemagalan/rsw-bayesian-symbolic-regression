"""C-006 (CONFIRMATORIO, protocolo §8): estimación del ruido sin modelo.
Estimadores (para cada nivel de EF y juntando ambos):
  - Rice: σ² = media de (y_{i+1} − y_i)²/2 entre vecinos de la rejilla.
  - Gasser, Sroka y Jennen-Steinmetz (1986): pseudo-residuos ε_i = a·y_{i−1} + b·y_{i+1} − y_i, con
    a = (x_{i+1} − x_i)/(x_{i+1} − x_{i−1}), b = (x_i − x_{i−1})/(x_{i+1} − x_{i−1}); σ² = media de ε²/(a² + b² + 1).
    Elimina la tendencia lineal local; Rice solo la constante (sesgo hacia arriba por la pendiente).
  - Dirección WT: para cada (EF, nivel nominal de WC), los 11 tiempos ordenados (x = WT).
    Dirección WC: para cada (EF, ciclos), los 11 niveles ordenados por la WC medida (x = WC).
  - Gemelas: var(d)/2 con d = y(EF 1.5) − y(EF 1.0) en cada celda (ddof = 1); cota superior si EF influye.
IC al 95 % por bootstrap de celdas (2000 réplicas, semilla 20261016): cada término (diferencia o pseudo-residuo)
se asigna a la celda de su punto inicial (Rice) o central (Gasser) y se remuestrean las 121 celdas.
Advertencia: los estimadores de diferencias tienen sesgo positivo por curvatura (más en WT bajo, donde la respuesta
cambia rápido) y la dirección WC incluye la variación de la WC medida; se informan todos.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-006_ruido.py   [--prueba-sintetica: σ = 0.28 conocido]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA

PRUEBA = '--prueba-sintetica' in sys.argv
OUT = ROOT/'resultados'/('_pruebas/C-006_sintetica' if PRUEBA else 'C-006_ruido'); OUT.mkdir(parents=True, exist_ok=True)
B, SEMILLA = 2000, 20261016
df = cargar_datos()
if PRUEBA:   # respuesta sintética con σ = 0.28 conocido (comprobación del estimador)
    rng0 = np.random.default_rng(7)
    df['T'] = 7.0*(1-np.exp(-0.6*df['WC']**2*df['WT'])) + rng0.normal(0, 0.28, len(df))
celdas = np.sort(df.celda.unique()); idx = {c: i for i, c in enumerate(celdas)}; NC = len(celdas)


def terminos(g, xcol):
    g = g.sort_values(xcol); x, y, ce = g[xcol].values, g['T'].values, g.celda.map(idx).values
    rice = [(ce[i], (y[i+1] - y[i])**2/2) for i in range(len(y) - 1)]
    gas = []
    for i in range(1, len(y) - 1):
        a = (x[i+1] - x[i])/(x[i+1] - x[i-1]); b = (x[i] - x[i-1])/(x[i+1] - x[i-1])
        gas.append((ce[i], (a*y[i-1] + b*y[i+1] - y[i])**2/(a*a + b*b + 1)))
    return rice, gas


T = {}
for ef in (1.0, 1.5):
    sub = df[df.EF == ef]
    for direc, grupo, xcol in (('WT', 'nivel', 'WT'), ('WC', 'cyc', 'WC')):
        r, g = [], []
        for _, gg in sub.groupby(grupo):
            a, b = terminos(gg, xcol); r += a; g += b
        T[('Rice', direc, ef)] = r; T[('Gasser', direc, ef)] = g
for est in ('Rice', 'Gasser'):
    for direc in ('WT', 'WC'):
        T[(est, direc, 'ambos')] = T[(est, direc, 1.0)] + T[(est, direc, 1.5)]
    T[(est, 'WT+WC', 'ambos')] = T[(est, 'WT', 'ambos')] + T[(est, 'WC', 'ambos')]

rng = np.random.default_rng(SEMILLA)
W = np.array([np.bincount(rng.integers(0, NC, NC), minlength=NC) for _ in range(B)])   # pesos de celda por réplica
filas = []
for (est, direc, ef), lst in T.items():
    ce = np.array([c for c, _ in lst]); v = np.array([t for _, t in lst])
    S = np.bincount(ce, weights=v, minlength=NC); N = np.bincount(ce, minlength=NC)
    s2 = S.sum()/N.sum(); bs = (W @ S)/(W @ N)
    filas.append(dict(estimador=est, direccion=direc, EF=ef, n_terminos=len(v), sigma2=s2, sigma=np.sqrt(s2),
                      sigma2_ic95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      sigma_ic95=[float(np.sqrt(np.percentile(bs, 2.5))), float(np.sqrt(np.percentile(bs, 97.5)))]))
# gemelas
pv = df.pivot_table(index='celda', columns='EF', values='T'); d = (pv[1.5] - pv[1.0]).reindex(celdas).values
s2 = np.var(d, ddof=1)/2
bs = np.array([np.var(d[rng.integers(0, NC, NC)], ddof=1)/2 for _ in range(B)])
filas.append(dict(estimador='gemelas var(d)/2', direccion='EF', EF='1.5 − 1.0', n_terminos=len(d), sigma2=s2, sigma=np.sqrt(s2),
                  sigma2_ic95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  sigma_ic95=[float(np.sqrt(np.percentile(bs, 2.5))), float(np.sqrt(np.percentile(bs, 97.5)))]))
t = pd.DataFrame(filas); t.to_csv(OUT/'estimaciones_ruido.csv', index=False)
ref = t[(t.estimador == 'Gasser') & (t.direccion == 'WT+WC') & (t.EF == 'ambos')].iloc[0]
inf = dict(protocolo_sha=PROTOCOLO_SHA, B=B, semilla=SEMILLA, estimaciones=t.to_dict('records'),
           sigma_gasser_conjunto=float(ref.sigma), sigma_gemelas=float(np.sqrt(s2)),
           posicion=('más cerca de σ = 0.28' if abs(ref.sigma - 0.28) < abs(ref.sigma - 0.10) else 'más cerca de σ = 0.10'))
(OUT/'C-006_informe.json').write_text(json.dumps(inf, indent=1, default=str, ensure_ascii=False), encoding='utf-8')
pd.set_option('display.width', 200)
print(t[['estimador', 'direccion', 'EF', 'n_terminos', 'sigma2', 'sigma', 'sigma_ic95']].to_string(index=False))
print('Posición respecto a los pilotos (Gasser, WT+WC, ambos EF):', inf['posicion'])
