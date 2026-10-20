"""E-010 (EXPLORATORIO, post hoc; 2026-10-18). Dos preguntas surgidas al ver C-005:
(a) ¿Rinden en la CV las formas fijas construidas con los motivos recurrentes del BMS?
    M_aditivo: a·ln(WC) + b/WT + c (3 parámetros; motivo dominante en los MDL de C-005).
    M_A:       (a − b/WT)·ln(WC + WT) (2 parámetros; candidato de E-009, elegido con TODOS los datos → fuga).
    Estructuras elegidas tras ver los datos: su error de CV es optimista y NO es confirmatorio.
(b) Comparaciones por pares entre los mejores modelos (TabPFN, EBM, GBM, BMS-promedio), no previstas en el protocolo.
Mismo t corregido (1/25 + 0.25) y test bayesiano correlacionado (ROPE ±0.03, y ±0.01 como sensibilidad).
Uso: python analisis/exploratorio/E-010_motivos_y_pares.py
"""
import sys, itertools, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from bms_comun import cargar_datos, ROOT

OUT = ROOT/'resultados'/'2026-10-18_E-010'; OUT.mkdir(parents=True, exist_ok=True)
df = cargar_datos(); folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
W = pd.read_csv(ROOT/'resultados'/'C-005_analisis_cv'/'mse_por_particion_ancho.csv', index_col=0)
formas = {'M_aditivo': (lambda X, a, b, c: a*np.log(X[1]) + b/X[0] + c, [3.0, -0.05, 2.0]),
          'M_A': (lambda X, a, b: (a - b/X[0])*np.log(X[1] + X[0]), [3.9, 0.047])}
filas = []
for p in range(25):
    te_ids = set(folds.query('repeticion == @p // 5 and particion == @p % 5').id)
    te, tr = df[df.id.isin(te_ids)], df[~df.id.isin(te_ids)]
    for nom, (f, p0) in formas.items():
        th, _ = curve_fit(f, np.vstack([tr.WT, tr.WC]), tr['T'].values, p0=p0, maxfev=20000)
        q = f(np.vstack([te.WT, te.WC]), *th)
        filas.append(dict(modelo=nom, particion=p, mse=float(np.mean((te['T'].values - q)**2)), parametros=th.tolist()))
F = pd.DataFrame(filas); F.to_csv(OUT/'formas_fijas_por_particion.csv', index=False)
for nom in formas:
    W[nom] = F[F.modelo == nom].set_index('particion').mse


def par(a, b, rope):
    d = (W[a] - W[b]).values; n = len(d); dbar = d.mean(); se = np.sqrt((1/n + 0.25)*d.var(ddof=1))
    post = stats.t(n - 1, loc=dbar, scale=se)
    return dict(A=a, B=b, rope=rope, mse_A=W[a].mean(), mse_B=W[b].mean(), d=dbar, ic95=[dbar - stats.t.ppf(.975, n-1)*se, dbar + stats.t.ppf(.975, n-1)*se],
                p_bilateral=2*stats.t.sf(abs(dbar/se), n - 1), P_A_mejor=post.cdf(-rope), P_equiv=post.cdf(rope) - post.cdf(-rope),
                P_A_peor=post.sf(rope), frac_A_mejor=float(np.mean(d < 0)))


pares = [par(a, b, r) for r in (0.03, 0.01) for a, b in
         list(itertools.combinations(['TabPFN_v2', 'EBM', 'GBM', 'BMS_promedio'], 2)) +
         [(m, c) for m in formas for c in ('BMS_MDL', 'EBM', 'cuadratico_EN', 'GAM')]]
T = pd.DataFrame(pares); T.to_csv(OUT/'comparaciones_por_pares.csv', index=False)
pd.set_option('display.width', 250)
print(F.groupby('modelo').mse.agg(['mean', 'std']).round(4))
print(T.assign(ic95=T.ic95.map(lambda v: f'[{v[0]:.3f}, {v[1]:.3f}]')).round(3).drop(columns=['mse_A', 'mse_B']).to_string(index=False))
