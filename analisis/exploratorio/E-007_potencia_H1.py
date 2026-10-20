"""E-007 (EXPLORATORIO, calibración de potencia de H1). 2026-10-16.
USA LOS DATOS REALES, pero solo con comparadores (sin BMS): estima la desviación típica s_d de las diferencias
de MSE por partición entre pares de modelos con la CV 5×5 agrupada por celda (pares gemelos EF1000/EF1500)
y calcula la potencia del contraste de no inferioridad de H1 para varios márgenes δ.
Comparadores simplificados (sin ajuste interno): cuadrático OLS, RF, GBM, saturación de E-001.
Uso: python analisis/exploratorio/E-007_potencia_H1.py
"""
import hashlib, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'Dataset'/'datos_corregido_junio16_2_puntos20-69-88.xlsx'
assert hashlib.sha256(DATA.read_bytes()).hexdigest().startswith('ae881203')
df = pd.read_excel(DATA).iloc[:, 2:9]; df.columns = ['id','cyc','WT','WC','EF','T','Q']; df = df.dropna(subset=['id']).reset_index(drop=True)
df['EF'] = df['EF']/1000
df['rk'] = df.groupby(['cyc','EF'])['WC'].rank(method='first').astype(int)
df['celda'] = (df['cyc']*100 + df['rk']).astype(int)
cm = df.groupby('celda')['T'].transform('mean')
df['estrato'] = pd.qcut(cm.rank(method='first'), 5, labels=False)
X = df[['WT','WC','EF']].values; y = df['T'].values

def folds(rep):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=20261016+rep).split(X, df['estrato'], df['celda']))

def sat(Xa, a, b, c, d):
    t, I = Xa[:,0], Xa[:,1]; return a*(1-np.exp(-b*I**c*t**d))

res = {k: [] for k in ['cuadratico','RF','GBM','E001','media']}
ntest = []
for rep in range(5):
    for tr, te in folds(rep):
        ntest.append(len(te))
        pf = PolynomialFeatures(2); m = LinearRegression().fit(pf.fit_transform(X[tr]), y[tr])
        res['cuadratico'].append(np.mean((m.predict(pf.transform(X[te]))-y[te])**2))
        rf = RandomForestRegressor(500, max_features=2, random_state=0).fit(X[tr], y[tr])
        res['RF'].append(np.mean((rf.predict(X[te])-y[te])**2))
        gb = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
        res['GBM'].append(np.mean((gb.predict(X[te])-y[te])**2))
        p, _ = curve_fit(sat, X[tr], y[tr], p0=[7, .3, 1.4, .35], maxfev=50000)
        res['E001'].append(np.mean((sat(X[te], *p)-y[te])**2))
        res['media'].append(np.mean((y[tr].mean()-y[te])**2))
R = {k: np.array(v) for k, v in res.items()}
r = np.mean(ntest)/(len(y)-np.mean(ntest)); J = 25
fac = np.sqrt(1/J + r); tcrit = stats.t.ppf(0.95, J-1)
print(f'n_test medio = {np.mean(ntest):.1f}; n_test/n_train = {r:.3f}; factor corrección = {fac:.3f}')
for k, v in R.items():
    print(f'{k:10s} MSE = {v.mean():.4f} ± {v.std(ddof=1):.4f} kN²')
print('\nPotencia del contraste de no inferioridad (verdad: ΔMSE = 0) para cada par:')
out = {'mse': {k: v.tolist() for k, v in R.items()}, 'pares': {}}
for a, b in [('GBM','RF'), ('cuadratico','GBM'), ('cuadratico','RF'), ('E001','GBM'), ('E001','cuadratico')]:
    d = R[a]-R[b]; sd = d.std(ddof=1); se = fac*sd
    pot = {dl: float(stats.norm.cdf(dl/se - tcrit)) for dl in (0.0127, 0.02, 0.03, 0.05)}
    out['pares'][f'{a}-{b}'] = dict(media_d=float(d.mean()), sd_d=float(sd), se_corr=float(se), potencia=pot)
    print(f'{a:>10s}−{b:<10s} d̄ = {d.mean():+.4f}  s_d = {sd:.4f}  SE corr = {se:.4f}  potencia δ=0.0127/0.02/0.03/0.05: ' +
          ' / '.join(f'{pot[x]:.2f}' for x in pot))
dst = ROOT/'resultados'/'2026-10-16_E-007'; dst.mkdir(parents=True, exist_ok=True)
(dst/'resultados.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
