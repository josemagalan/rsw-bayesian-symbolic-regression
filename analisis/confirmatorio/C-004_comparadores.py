"""C-004 (CONFIRMATORIO, protocolo §3): comparadores 1–6, 8 y 9 en las 25 particiones agrupadas.
1 media · 2 cuadrático + elastic net · 3 GAM te(WT,WC)+l(EF) · 4 RF · 5 GBM · 6 EBM · 8 TabPFN v2 · 9 saturación E-001.
CV interna agrupada por celda (GroupKFold 5, barajada, semilla 20261016 + partición), métrica MSE.
Predicciones no finitas → media del entrenamiento (se cuentan). Escribe predicciones y MSE por partición.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-004_comparadores.py
     ... --sin-ef         (sensibilidad §9: comparadores sin EF; carpeta aparte)
     ... --prueba-sintetica (respuesta sintética, 2 particiones; salida en resultados\\_pruebas)
     ... --sin-tabpfn     (solo para pruebas en entornos sin TabPFN; NO en la ejecución confirmatoria)"""
import json, sys, time, warnings, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA
warnings.filterwarnings('ignore')

prueba = '--prueba-sintetica' in sys.argv; sin_ef = '--sin-ef' in sys.argv; sin_tabpfn = '--sin-tabpfn' in sys.argv
df = cargar_datos()
folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
if prueba:
    rng = np.random.default_rng(7); df['T'] = 7.0*(1-np.exp(-0.6*df['WC']**2*df['WT'])) + rng.normal(0, 0.28, len(df))
    out = ROOT/'resultados'/'_pruebas'/('C-004_sintetica' + ('_sin_ef' if sin_ef else ''))
else:
    out = ROOT/'resultados'/('C-004_comparadores' + ('_sin_ef' if sin_ef else ''))
out.mkdir(parents=True, exist_ok=True)
VARS = ['WT', 'WC'] if sin_ef else ['WT', 'WC', 'EF']


def inner(Xtr, ytr, grp, seed):
    return list(GroupKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr, grp))


def m_media(Xtr, ytr, Xte, cv, seed, dtr):
    return np.full(len(Xte), ytr.mean()), {}


def m_en(Xtr, ytr, Xte, cv, seed, dtr):
    m = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                      ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=50, cv=cv, max_iter=200000))
    m.fit(Xtr, ytr); en = m[-1]
    return m.predict(Xte), dict(alpha=float(en.alpha_), l1_ratio=float(en.l1_ratio_))


def m_gam(Xtr, ytr, Xte, cv, seed, dtr):
    from pygam import LinearGAM, te, l
    terms = te(0, 1) + l(2) if Xtr.shape[1] == 3 else te(0, 1)
    g = LinearGAM(terms).gridsearch(Xtr, ytr, progress=False)
    return g.predict(Xte), dict(lam=str(g.lam))


def m_rf(Xtr, ytr, Xte, cv, seed, dtr):
    mf = [1, 2, 3] if Xtr.shape[1] == 3 else [1, 2]
    gs = GridSearchCV(RandomForestRegressor(500, random_state=0, n_jobs=-1),
                      {'max_features': mf, 'min_samples_leaf': [1, 3, 5]}, cv=cv, scoring='neg_mean_squared_error')
    gs.fit(Xtr, ytr); return gs.predict(Xte), gs.best_params_


def m_gbm(Xtr, ytr, Xte, cv, seed, dtr):
    gs = GridSearchCV(GradientBoostingRegressor(random_state=0),
                      {'n_estimators': [100, 300, 600], 'max_depth': [2, 3, 4], 'learning_rate': [0.03, 0.1]},
                      cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(Xtr, ytr); return gs.predict(Xte), gs.best_params_


def m_ebm(Xtr, ytr, Xte, cv, seed, dtr):
    from interpret.glassbox import ExplainableBoostingRegressor
    m = ExplainableBoostingRegressor(random_state=0); m.fit(Xtr, ytr); return m.predict(Xte), {}


def m_tabpfn(Xtr, ytr, Xte, cv, seed, dtr):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2)   # sin respaldo silencioso (protocolo §3)
    m.fit(Xtr, ytr); return m.predict(Xte), dict(version='V2')


def m_e001(Xtr, ytr, Xte, cv, seed, dtr):
    f = lambda X, a, b, c, d: a*(1-np.exp(-b*X[:, 1]**c*X[:, 0]**d))
    p, _ = curve_fit(f, Xtr[:, :2], ytr, p0=[7, .3, 1.4, .35], maxfev=50000)
    return f(Xte[:, :2], *p), dict(par=[float(v) for v in p])


MODELOS = [('media', m_media), ('cuadratico_EN', m_en), ('GAM', m_gam), ('RF', m_rf), ('GBM', m_gbm),
           ('EBM', m_ebm), ('TabPFN_v2', m_tabpfn), ('E001_exploratorio', m_e001)]
if sin_tabpfn:
    MODELOS = [m for m in MODELOS if m[0] != 'TabPFN_v2']

filas, preds, detalles = [], [], []
hechas = set()
fp = out/'mse_por_particion.csv'
if fp.exists():
    prev = pd.read_csv(fp).fillna({'fallo': ''})
    ok = prev[prev['fallo'] == '']                 # las filas con fallo técnico se repiten al relanzar
    filas = ok.to_dict('records'); hechas = {(r['modelo'], r['particion']) for r in filas}
    pp = pd.read_csv(out/'predicciones.csv')
    preds = pp.merge(ok[['modelo', 'particion']], on=['modelo', 'particion']).to_dict('records')
for rep in range(5):
    for k in range(5):
        p = 5*rep + k
        if prueba and p >= 2:
            continue
        ids_te = set(folds.query('repeticion == @rep and particion == @k')['id'])
        te = df[df.id.isin(ids_te)]; tr = df[~df.id.isin(ids_te)]
        Xtr, ytr, Xte, yte = tr[VARS].values, tr['T'].values, te[VARS].values, te['T'].values
        cv = inner(Xtr, ytr, tr['celda'].values, 20261016 + p)
        for nombre, fn in MODELOS:
            if (nombre, p) in hechas:
                continue
            t0 = time.process_time(); tw = time.time()
            try:
                yp, info = fn(Xtr, ytr, Xte, cv, 20261016 + p, tr); fallo = ''
            except Exception as e:
                yp, info, fallo = np.full(len(Xte), np.nan), {}, repr(e)
            yp = np.asarray(yp, dtype=float); nf = int((~np.isfinite(yp)).sum())
            yp = np.where(np.isfinite(yp), yp, ytr.mean())
            mse = float(np.mean((yp - yte)**2))
            filas.append(dict(modelo=nombre, particion=p, repeticion=rep, k=k, mse=mse, n_test=len(yte), no_finitos=nf,
                              fallo=fallo, cpu_s=round(time.process_time()-t0, 2), reloj_s=round(time.time()-tw, 2),
                              info=json.dumps(info, default=str)))
            preds += [dict(modelo=nombre, particion=p, id=int(i), y=float(a), pred=float(b)) for i, a, b in zip(te['id'], yte, yp)]
            print(f'p{p:02d} {nombre:18s} MSE={mse:.4f} no_finitos={nf} {fallo[:60]}', flush=True)
        pd.DataFrame(filas).to_csv(fp, index=False); pd.DataFrame(preds).to_csv(out/'predicciones.csv', index=False)
(out/'entorno.json').write_text(json.dumps(dict(python=platform.python_version(), plataforma=platform.platform(),
                                                protocolo_sha=PROTOCOLO_SHA, variables=VARS)), encoding='utf-8')
r = pd.DataFrame(filas).groupby('modelo')['mse'].agg(['mean', 'std', 'count']).sort_values('mean')
print(r)
