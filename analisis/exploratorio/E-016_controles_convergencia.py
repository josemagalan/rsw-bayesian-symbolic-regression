"""E-016 (EXPLORATORIO, post hoc; 2026-10-19): controles de la revisión adversaria de A-024 (convergencia BMS–PySR).
Versiona los cálculos del revisor (scripts cv.py, dist.py y pdp.py en su espacio temporal) para que los números sean citables.
  1. CV (25 particiones agrupadas) de formas con el factor temporal (a − b/WT) y distintas dependencias en WC: logaritmo con y sin
     umbral libre, potencia, raíz, saturación exponencial y Michaelis–Menten, con y sin umbral explícito c; y factor temporal
     exponencial (1 − e^(−WT/τ)).
  2. Ajuste con todos los datos de esas formas: MSE en muestra, distancia de su superficie a F1 = (a − b/WT)·ln WC (proyección en
     la rejilla nominal 11 × 11), R² de perfiles en ln WC (¿discrimina el R² de perfiles?); IC bootstrap por celdas del umbral c de
     ln(WC/c); F1 con WC en amperios (dependencia de unidades).
  3. Dependencia parcial (PDP, de E-015): RMSE del ajuste 1D de cada forma alternativa.
Uso: python analisis/exploratorio/E-016_controles_convergencia.py     (salida en resultados/2026-10-19_E-016)"""
import json, platform, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from bms_comun import cargar_datos, ROOT
warnings.filterwarnings('ignore')

OUT = ROOT/'resultados'/'2026-10-19_E-016'; OUT.mkdir(parents=True, exist_ok=True)
df = cargar_datos(); folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
L = np.log
pos = lambda x: np.clip(x, 1e-9, None)
F1 = lambda X, a, b: (a - b/X[0])*L(X[1])
FORMAS = {
    'F1 (a−b/WT)·lnWC': (F1, [4, .05]),
    'F2 (a−b/WT)·ln(WC+WT)': (lambda X, a, b: (a - b/X[0])*L(X[1] + X[0]), [3.9, .047]),
    'P1 (a−b/WT)·ln(WC/c)': (lambda X, a, b, c: (a - b/X[0])*L(X[1]/c), [4, .05, 1]),
    'P2 (a−b/WT)·WC^c': (lambda X, a, b, c: (a - b/X[0])*X[1]**c, [2, .03, .6]),
    'P3 (a−b/WT)·√(WC−c)': (lambda X, a, b, c: (a - b/X[0])*np.sqrt(pos(X[1] - c)), [3, .04, 1]),
    'P4 (a−b/WT)·(WC−c)^d': (lambda X, a, b, c, d: (a - b/X[0])*pos(X[1] - c)**d, [3, .04, 1, .5]),
    'P5 (a−b/WT)·(1−e^(−k·WC))': (lambda X, a, b, k: (a - b/X[0])*(1 - np.exp(-k*X[1])), [8, .1, .2]),
    'P6 (a−b/WT)·(1−e^(−k(WC−c)))': (lambda X, a, b, k, c: (a - b/X[0])*(1 - np.exp(-k*(X[1] - c))), [8, .1, .3, 1]),
    'P7 (a−b/WT)·WC/(K+WC)': (lambda X, a, b, K: (a - b/X[0])*X[1]/(K + X[1]), [10, .1, 3]),
    'P8 (a−b/WT)·(WC−c)/(K+WC−c)': (lambda X, a, b, K, c: (a - b/X[0])*(X[1] - c)/(K + X[1] - c), [10, .1, 3, 1]),
    'P9 (a−b/WT)·(WC−c)': (lambda X, a, b, c: (a - b/X[0])*(X[1] - c), [1, .01, 1]),
    'Q1 a(1−e^(−WT/τ))·lnWC': (lambda X, a, t: a*(1 - np.exp(-X[0]/t))*L(X[1]), [3.6, .05]),
    'Q2 a(1−e^(−WT/τ))·(WC−c)^d': (lambda X, a, t, c, d: a*(1 - np.exp(-X[0]/t))*pos(X[1] - c)**d, [3, .05, 1, .5]),
    'C1 (a−b/WT)·√WC': (lambda X, a, b: (a - b/X[0])*np.sqrt(X[1]), [3, .03]),
    'C4 a(1−e^(−b·WC^c·WT^d)) (E-001)': (lambda X, a, b, c, d: a*(1 - np.exp(-b*X[1]**c*X[0]**d)), [7.3, .34, 1.38, .35]),
}
UMBRAL = {'P1 (a−b/WT)·ln(WC/c)': 2, 'P3 (a−b/WT)·√(WC−c)': 2, 'P4 (a−b/WT)·(WC−c)^d': 2, 'P6 (a−b/WT)·(1−e^(−k(WC−c)))': 3,
          'P8 (a−b/WT)·(WC−c)/(K+WC−c)': 3, 'P9 (a−b/WT)·(WC−c)': 2, 'Q2 a(1−e^(−WT/τ))·(WC−c)^d': 2}   # índice del parámetro c

# ---------------------------------------------------------------- 1. CV
filas = []
for p in range(25):
    ids = set(folds.query('repeticion == @p // 5 and particion == @p % 5').id); te, tr = df[df.id.isin(ids)], df[~df.id.isin(ids)]
    for n, (f, p0) in FORMAS.items():
        try:
            th, _ = curve_fit(f, np.vstack([tr.WT, tr.WC]), tr['T'].values, p0=p0, maxfev=100000)
            mse = float(np.mean((te['T'].values - f(np.vstack([te.WT, te.WC]), *th))**2))
        except Exception:
            th, mse = [np.nan]*len(p0), np.nan
        filas.append(dict(forma=n, particion=p, mse=mse, k=len(p0), umbral_c=th[UMBRAL[n]] if n in UMBRAL else np.nan))
CV = pd.DataFrame(filas); CV.to_csv(OUT/'cv_por_particion.csv', index=False)
W = CV.pivot(index='particion', columns='forma', values='mse'); base = W['F1 (a−b/WT)·lnWC']; tab = []
for n in FORMAS:
    d = (W[n] - base).dropna(); m = len(d); se = np.sqrt((1/m + .25)*d.var(ddof=1)); q = stats.t.ppf(.975, m - 1)
    tab.append(dict(forma=n, k=len(FORMAS[n][1]), mse=W[n].mean(), d_vs_F1=d.mean(), ic95_inf=d.mean() - q*se, ic95_sup=d.mean() + q*se,
                    frac_mejor_que_F1=float((d < 0).mean()), umbral_c_mediana=CV[CV.forma == n].umbral_c.median(), n=m))
CVT = pd.DataFrame(tab).sort_values('mse'); CVT.to_csv(OUT/'cv_resumen.csv', index=False)

# ---------------------------------------------------------------- 2. todos los datos: distancia a F1, R² de perfiles, umbral
X = np.vstack([df.WT, df.WC]); y = df['T'].values
GWT = np.arange(2, 13)*0.02; GWC = np.linspace(1.5, 6.5, 11); GT, GC = np.meshgrid(GWT, GWC, indexing='ij'); XG = np.vstack([GT.ravel(), GC.ravel()])


def r2prof(Z, g, eje):
    x = GWC if eje == 'WC' else GWT; gg = g(x); r = []; mx = []
    for z in (Z if eje == 'WC' else Z.T):
        A = np.column_stack([np.ones_like(gg), gg]); c, *_ = np.linalg.lstsq(A, z, rcond=None); e = z - A@c
        r.append(1 - np.sum(e**2)/np.sum((z - z.mean())**2)); mx.append(np.abs(e).max())
    return float(np.mean(r)), float(np.max(mx))


filas = []
for n, (f, p0) in FORMAS.items():
    th, _ = curve_fit(f, X, y, p0=p0, maxfev=100000); Z = f(XG, *th).reshape(GT.shape)
    tp, _ = curve_fit(F1, XG, Z.ravel(), p0=[4, .05])
    rl, ml = r2prof(Z, np.log, 'WC'); rlin, _ = r2prof(Z, lambda x: x, 'WC'); ri, _ = r2prof(Z, lambda x: 1/x, 'WT')
    filas.append(dict(forma=n, k=len(p0), mse_en_muestra=float(np.mean((y - f(X, *th))**2)),
                      dist_F1=float(np.sqrt(np.mean((F1(XG, *tp) - Z.ravel())**2))), r2_perfil_lnWC=rl, max_residuo_perfil_ln=ml,
                      r2_perfil_linealWC=rlin, r2_perfil_invWT=ri, umbral_c=float(th[UMBRAL[n]]) if n in UMBRAL else np.nan,
                      parametros=json.dumps([round(float(v), 4) for v in th])))
TD = pd.DataFrame(filas); TD.to_csv(OUT/'todos_los_datos.csv', index=False)
fP1 = FORMAS['P1 (a−b/WT)·ln(WC/c)'][0]; rng = np.random.default_rng(20261019); cel = df.celda.values; uc = np.unique(cel); cs = []
for _ in range(1000):
    idx = np.concatenate([np.flatnonzero(cel == c) for c in rng.choice(uc, len(uc))])
    try:
        cs.append(curve_fit(fP1, X[:, idx], y[idx], p0=[4, .05, 1], maxfev=20000)[0][2])
    except Exception:
        pass
fA = lambda X, a, b: (a - b/X[0])*L(X[1]*1000); thA, _ = curve_fit(fA, X, y, p0=[.6, .01])
UMB = dict(c_lnWCc_mediana=float(np.median(cs)), c_lnWCc_ic95=np.percentile(cs, [2.5, 97.5]).tolist(), n_boot=len(cs),
           umbral_c_rango_formas=[float(np.nanmin(TD.umbral_c)), float(np.nanmax(TD.umbral_c))], WC_min_ensayada=float(df.WC.min()),
           F1_en_amperios_mse=float(np.mean((y - fA(X, *thA))**2)), F1_en_kA_mse=float(TD.loc[TD.forma.str.startswith('F1'), 'mse_en_muestra'].iloc[0]))

# ---------------------------------------------------------------- 3. PDP (de E-015): RMSE del ajuste 1D
P = pd.read_csv(ROOT/'resultados'/'2026-10-19_E-015'/'pdp.csv')
FWC = {'ln': (lambda x, a, b: a + b*np.log(x), [0, 3]), 'ln(x−c)': (lambda x, a, b, c: a + b*np.log(pos(x - c)), [0, 3, .1]),
       '√(x−c)': (lambda x, a, b, c: a + b*np.sqrt(pos(x - c)), [0, 3, 1]), 'x^c': (lambda x, a, b, c: a + b*x**c, [0, 3, .5]),
       '1−e^(−kx)': (lambda x, a, b, k: a + b*(1 - np.exp(-k*x)), [0, 8, .3]), 'x/(K+x)': (lambda x, a, b, K: a + b*x/(K + x), [0, 10, 3]),
       'lineal': (lambda x, a, b: a + b*x, [0, 1])}
FWT = {'1/x': (lambda x, a, b: a + b/x, [5, -.05]), 'e^(−x/τ)': (lambda x, a, b, t: a + b*np.exp(-x/t), [5, -5, .05]),
       'x^c': (lambda x, a, b, c: a + b*x**c, [5, -1, -.5]), 'ln': (lambda x, a, b: a + b*np.log(x), [5, 1])}
filas = []
for var, FF in [('WC', FWC), ('WT', FWT)]:
    for m, g in P[P.variable == var].groupby('metodo'):
        x, z = g.x.values, g.pdp.values; r = dict(variable=var, metodo=m)
        for n, (f, p0) in FF.items():
            try:
                th, _ = curve_fit(f, x, z, p0=p0, maxfev=50000); r[n] = float(np.sqrt(np.mean((z - f(x, *th))**2)))
            except Exception:
                r[n] = np.nan
        filas.append(r)
PD = pd.DataFrame(filas); PD.to_csv(OUT/'pdp_rmse_ajuste.csv', index=False)


def a_md(t, dec=4):
    fmt = lambda v: f'{v:.{dec}g}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, t.columns)) + ' |', '|' + '---|'*len(t.columns)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in t.itertuples(index=False)])


md = ['# E-016 — Controles de la convergencia BMS–PySR (exploratorio, post hoc)', '',
      '## 1. CV (25 particiones agrupadas; d = forma − F1; IC con la corrección de Nadeau y Bengio)', '', a_md(CVT), '',
      '## 2. Ajuste con todos los datos (distancia de la superficie a F1 en la rejilla nominal; R² de perfiles)', '',
      a_md(TD.drop(columns=['parametros'])), '', f'Umbral y unidades: {json.dumps(UMB)}', '',
      '## 3. PDP de modelos no simbólicos (E-015): RMSE (kN) del ajuste 1D de cada forma', '', a_md(PD), '',
      f'Entorno: Python {platform.python_version()}, {platform.platform()}.']
(OUT/'E-016_informe.md').write_text('\n'.join(md), encoding='utf-8')
(OUT/'entorno.json').write_text(json.dumps(dict(python=platform.python_version(), plataforma=platform.platform(), **UMB)), encoding='utf-8')
print('\n'.join(md))
