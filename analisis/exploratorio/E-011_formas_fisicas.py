"""E-011 (EXPLORATORIO, post hoc; 2026-10-18). Contraste de formas con motivación física, sugeridas por la revisión
bibliográfica de la interpretación física (articulo/interpretacion_fisica.md). Mismas 25 particiones agrupadas.
  F_inv:   a·ln WC − b/WT + c                      (motivo aditivo de C-005; saturación hiperbólica en t)
  F_lnt:   a·ln WC + b·ln WT + c                   (TSLBC ∝ ln del calor, sin pérdidas: ln(I²t))
  F_exp:   a·ln WC + b·(1 − exp(−WT/τ)) + c        (saturación exponencial en t)
  F_KE:    a·ln(WT·(WC² − q)) + c                  (ln del calor neto de Kim y Eagar 1988: t·(RI² − Q_L))
  F_KEinv: a·ln(WC² − q) − b/WT + c                (pérdidas en la corriente + saturación hiperbólica en t)
  F_M_A:   (a − b/WT)·ln(WC + WT)                  (candidato de E-009)
Estructuras elegidas tras ver los datos → MSE optimistas, no confirmatorios.
Uso: python analisis/exploratorio/E-011_formas_fisicas.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from bms_comun import cargar_datos, ROOT

OUT = ROOT/'resultados'/'2026-10-18_E-011'; OUT.mkdir(parents=True, exist_ok=True)
df = cargar_datos(); folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
W = pd.read_csv(ROOT/'resultados'/'C-005_analisis_cv'/'mse_por_particion_ancho.csv', index_col=0)
sl = lambda z: np.log(np.maximum(z, 1e-9))
F = {'F_inv': (lambda X, a, b, c: a*np.log(X[1]) - b/X[0] + c, [2.5, 0.05, 3.0], 3),
     'F_lnt': (lambda X, a, b, c: a*np.log(X[1]) + b*np.log(X[0]) + c, [2.5, 1.0, 6.0], 3),
     'F_exp': (lambda X, a, b, tau, c: a*np.log(X[1]) + b*(1 - np.exp(-X[0]/tau)) + c, [2.5, 2.0, 0.05, 1.0], 4),
     'F_KE': (lambda X, a, q, c: a*sl(X[0]*(X[1]**2 - q)) + c, [1.5, 1.0, 8.0], 3),
     'F_KEinv': (lambda X, a, q, b, c: a*sl(X[1]**2 - q) - b/X[0] + c, [1.2, 1.0, 0.05, 3.0], 4),
     'F_M_A': (lambda X, a, b: (a - b/X[0])*np.log(X[1] + X[0]), [3.9, 0.047], 2)}
filas = []
for p in range(25):
    ids = set(folds.query('repeticion == @p // 5 and particion == @p % 5').id)
    te, tr = df[df.id.isin(ids)], df[~df.id.isin(ids)]
    for nom, (f, p0, k) in F.items():
        try:
            th, _ = curve_fit(f, np.vstack([tr.WT, tr.WC]), tr['T'].values, p0=p0, maxfev=50000)
            q = f(np.vstack([te.WT, te.WC]), *th); mse = float(np.mean((te['T'].values - q)**2))
        except Exception as e:
            th, mse = [np.nan]*len(p0), np.nan
        filas.append(dict(forma=nom, particion=p, mse=mse, parametros=list(th)))
R = pd.DataFrame(filas); R.to_csv(OUT/'mse_por_particion.csv', index=False)
for nom in F:
    W[nom] = R[R.forma == nom].set_index('particion').mse
# ajuste con todos los datos (parámetros y R² en muestra, descriptivo)
todo = []
for nom, (f, p0, k) in F.items():
    th, _ = curve_fit(f, np.vstack([df.WT, df.WC]), df['T'].values, p0=p0, maxfev=50000)
    res = df['T'].values - f(np.vstack([df.WT, df.WC]), *th); n = len(res)
    todo.append(dict(forma=nom, k=k, parametros=np.round(th, 4).tolist(), mse_muestra=float(np.mean(res**2)),
                     bic=float(n*np.log(np.mean(res**2)) + (k + 1)*np.log(n))))
T = pd.DataFrame(todo)


def par(a, b):
    d = (W[a] - W[b]).values; n = len(d); se = np.sqrt((1/n + 0.25)*d.var(ddof=1)); post = stats.t(n - 1, loc=d.mean(), scale=se)
    return dict(d=d.mean(), p=2*stats.t.sf(abs(d.mean()/se), n - 1), P_eq=post.cdf(0.03) - post.cdf(-0.03), mejor_en=int((d < 0).sum()))


T['mse_cv'] = T.forma.map(lambda m: W[m].mean()); T['de_cv'] = T.forma.map(lambda m: W[m].std())
for ref in ('F_inv', 'EBM', 'BMS_MDL'):
    T[f'd_vs_{ref}'] = T.forma.map(lambda m: par(m, ref)['d'] if m != ref else 0.0)
    T[f'Peq_vs_{ref}'] = T.forma.map(lambda m: par(m, ref)['P_eq'] if m != ref else 1.0)
T['mejor_que_F_inv_en'] = T.forma.map(lambda m: par(m, 'F_inv')['mejor_en'] if m != 'F_inv' else np.nan)
T = T.sort_values('mse_cv'); T.to_csv(OUT/'resumen.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(T.round(4).to_string(index=False))
