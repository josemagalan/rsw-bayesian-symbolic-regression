"""E-001 (EXPLORATORIO, 2026-10-14). Formas cerradas sencillas vs. baselines.
No es confirmatorio: las formas se eligieron mirando todos los datos. Ver 00_gestion/DIARIO.md.
Requiere: numpy, pandas, openpyxl, scipy, scikit-learn.
Uso (desde la raíz del proyecto): python analisis/exploratorio/E-001_formas_cerradas.py
"""
import hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

DATA = Path('Dataset/datos_corregido_junio16_2_puntos20-69-88.xlsx')
SHA = 'ae881203f8f6027daf1baa3bccc56aa0d9d528b8d6cd2aaa0a87af47fc02e99b'
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == SHA, 'El fichero de datos ha cambiado'

df = pd.read_excel(DATA, header=0).iloc[:, 2:9]
df.columns = ['id', 'cyc', 'WT', 'WC', 'EF', 'T', 'Q']
df = df.dropna(subset=['id'])
X = df[['WT', 'WC', 'EF']].values.astype(float); X[:, 2] /= 1000  # EF en kN
y = df['T'].values.astype(float)

def m_sat(X, a, b, c, e):
    t, I, F = X.T; return a*(1-np.exp(-b*(I**2*t)**c*F**e))
def m_logi(X, a, m, s, e):
    t, I, F = X.T; q = np.log(I**2*t)+e*np.log(F); return a/(1+np.exp(-(q-m)/s))
def m_loss(X, a, b, c, e):
    t, I, F = X.T; return a*(1-np.exp(-b*np.clip(I**2-c*F**e, 0, None)*t))
def m_loss2(X, a, b, c, d, e):
    t, I, F = X.T; return a*(1-np.exp(-b*np.clip(I**2-c*F**e, 0, None)*t**d))
def m_pow(X, a, b, c, d, e):
    t, I, F = X.T; return a*(1-np.exp(-b*I**c*t**d*F**e))

MODELS = {
    'sat (I2t)^c F^e': (m_sat, [7, 1, 1, 0]),
    'logistica log(I2t)': (m_logi, [7, 0, 0.5, 0]),
    'sat (I2-c F^e) t': (m_loss, [7, 0.1, 4, 1]),
    'sat (I2-c F^e) t^d': (m_loss2, [7, 0.1, 4, 0.5, 1]),
    'sat I^c t^d F^e': (m_pow, [7, 0.1, 2, 0.5, 0]),
}
res = {k: [] for k in list(MODELS) + ['cuadratico OLS', 'GBM']}
for tr, te in RepeatedKFold(n_splits=10, n_repeats=10, random_state=1).split(X):
    for k, (f, p0) in MODELS.items():
        try:
            p, _ = curve_fit(f, X[tr], y[tr], p0=p0, maxfev=50000)
            res[k].append(np.mean((f(X[te], *p)-y[te])**2))
        except Exception:
            res[k].append(np.nan)
    pf = PolynomialFeatures(2)
    lr = LinearRegression().fit(pf.fit_transform(X[tr]), y[tr])
    res['cuadratico OLS'].append(np.mean((lr.predict(pf.transform(X[te]))-y[te])**2))
    g = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=0).fit(X[tr], y[tr])
    res['GBM'].append(np.mean((g.predict(X[te])-y[te])**2))

print(f'var(TSLBC) = {y.var():.4f} kN^2')
for k, v in res.items():
    print(f'{k:22s} MSE = {np.nanmean(v):.4f} +/- {np.nanstd(v):.4f} kN^2  (fallos: {int(np.isnan(v).sum())})')
for k, (f, p0) in MODELS.items():
    p, _ = curve_fit(f, X, y, p0=p0, maxfev=50000)
    print(k, np.round(p, 3), 'R2 en muestra =', round(1-np.mean((f(X, *p)-y)**2)/y.var(), 4))
