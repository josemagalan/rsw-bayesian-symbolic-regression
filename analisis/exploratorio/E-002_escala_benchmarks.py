"""E-002 (EXPLORATORIO / verificación de benchmarks, F0). 2026-10-14.
Objetivo: resolver Q-002 empíricamente. ¿El MSE = 0.031 del EBM (Santos et al. 2022) está en kN² o en escala estandarizada?
Se replica aproximadamente su protocolo: datos estandarizados (media 0, var 1 en todas las variables),
CV externa 10-fold (anidada 5-fold para GBM/RF), y se reporta el MSE en ambas escalas.
Además (E-003): se evalúa la ecuación (6) de Martín et al. 2016 sobre los datos para fijar sus unidades.
Requiere: numpy, pandas, openpyxl, scikit-learn, interpret-core.
Uso (desde la raíz del proyecto): python analisis/exploratorio/E-002_escala_benchmarks.py
"""
import hashlib, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from interpret.glassbox import ExplainableBoostingRegressor
warnings.filterwarnings('ignore')

DATA = Path('Dataset/datos_corregido_junio16_2_puntos20-69-88.xlsx')
SHA = 'ae881203f8f6027daf1baa3bccc56aa0d9d528b8d6cd2aaa0a87af47fc02e99b'
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == SHA, 'El fichero de datos ha cambiado'
df = pd.read_excel(DATA, header=0).iloc[:, 2:9]
df.columns = ['id', 'cyc', 'WT', 'WC', 'EF', 'T', 'Q']
df = df.dropna(subset=['id'])
y = df['T'].values.astype(float)

# ---------- E-003: ecuación (6) de Martín et al. 2016 ----------
def eq6(WT, WC, EF):
    return (-1.93914 + 0.34904*WT - 0.02266*WT**2 + 1.95763*WC - 0.14341*WC**2
            - 0.36273*EF + 0.01260*WT*WC + 0.03211*WT*EF + 0.01422*WC*EF)
print('== E-003: ecuacion (6) de 2016 evaluada en muestra con distintas unidades ==')
for wt_name, WT in [('WT en ciclos', df['cyc'].values), ('WT en s', df['WT'].values)]:
    for ef_name, EF in [('EF en kN', df['EF'].values/1000), ('EF en N', df['EF'].values)]:
        p = eq6(WT, df['WC'].values, EF)
        print(f'  {wt_name:13s} {ef_name:9s} MSE en muestra = {np.mean((p-y)**2):10.4f} kN^2')

# ---------- E-002: escala del MSE del EBM ----------
X = df[['WT', 'WC', 'EF']].values.astype(float)
mx, sx = X.mean(0), X.std(0, ddof=1)
my, sy = y.mean(), y.std(ddof=1)
Xs, ys = (X-mx)/sx, (y-my)/sy
print(f'\nvar(TSLBC) ddof=1 = {sy**2:.4f} kN^2  -> 0.031 * var = {0.031*sy**2:.4f} kN^2')

# Nota: rejillas reducidas por coste computacional (2 núcleos). EBM con hiperparámetros por defecto
# (sin búsqueda interna); GBM/RF con CV interna de 5 particiones. Suficiente para decidir la ESCALA del error.
models = {
    'EBM': (ExplainableBoostingRegressor(random_state=0, n_jobs=1), None),
    'GBM': (GradientBoostingRegressor(random_state=0, learning_rate=0.05), {'n_estimators': [150, 400], 'max_depth': [2, 3]}),
    'RF':  (RandomForestRegressor(n_estimators=300, random_state=0), {'max_features': [1, 2, 3]}),
}
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
res = {k: [] for k in models}
for r in range(REPS):
    outer = KFold(10, shuffle=True, random_state=100+r)
    for tr, te in outer.split(Xs):
        for k, (est, grid) in models.items():
            if grid is None:
                gs = est.fit(Xs[tr], ys[tr])
            else:
                gs = GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=r), scoring='neg_mean_squared_error', n_jobs=-1)
                gs.fit(Xs[tr], ys[tr])
            res[k].append(np.mean((gs.predict(Xs[te])-ys[te])**2))
print(f'\n== E-002: CV externa 10-fold, {REPS} repeticion(es) externas, datos estandarizados ==')
print(f'{"modelo":6s} {"MSE estandarizado":>22s} {"MSE en kN^2":>22s}')
for k, v in res.items():
    v = np.array(v)
    print(f'{k:6s} {v.mean():10.4f} +/- {v.std():.4f} {v.mean()*sy**2:12.4f} +/- {v.std()*sy**2:.4f}')
print('Referencia Santos et al. 2022: EBM 0.031 (0.013), GBM 0.033 (0.015), RF 0.034 (0.017)')
