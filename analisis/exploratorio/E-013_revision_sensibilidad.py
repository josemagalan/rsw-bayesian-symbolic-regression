"""E-013 (EXPLORATORIO, post hoc): comprobaciones de la revisión adversaria de A-018 y A-019 (2026-10-19).
No ajusta nada; lee las salidas de C-005, C-007 y C-009.
  1. Extrapolación: MSE de cada método en cada división frente al MSE de la CV principal EN LOS MISMOS PUNTOS
     (media sobre las 5 repeticiones del error cuadrático de cada punto); desglose por nivel extremo; sin el punto TSLBC = 0;
     MDL y promedio de cada cadena del BMS; bootstrap pareado por celdas (2000 réplicas, semilla 20261019).
  2. Orden de métodos: tau de Kendall entre la ordenación del análisis principal y la de cada variante (25 particiones).
Uso: python analisis/exploratorio/E-013_revision_sensibilidad.py   (salida en resultados/2026-10-19_E-013)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from bms_comun import cargar_datos, ROOT

R = ROOT/'resultados'; OUT = R/'2026-10-19_E-013'; OUT.mkdir(parents=True, exist_ok=True)
df = cargar_datos().set_index('id')
id_cero = int(df.index[df['T'] == 0][0])
L9 = pd.read_csv(R/'C-009_analisis_sensibilidad'/'predicciones_sensibilidad.csv')
P5 = pd.read_csv(R/'C-005_analisis_cv'/'predicciones_todas.csv')
EXT = {0: 'WT_extremos', 1: 'WC_extremos'}
E = L9[L9.variante == 'extrapolacion'].copy(); E['division'] = E.particion.map(EXT)
E['e2'] = (E.y - E.pred)**2
E['nivel'] = np.where(E.division == 'WT_extremos', 'ciclos_' + df.loc[E.id, 'cyc'].astype(str).values,
                      'nivel_WC_' + df.loc[E.id, 'nivel'].astype(str).values)
cv_punto = P5.assign(e2=(P5.y - P5.pred)**2).groupby(['modelo', 'id']).e2.mean()   # media sobre las 5 repeticiones

filas = []
for (dv, m), g in E.groupby(['division', 'modelo']):
    cv = cv_punto.loc[m].reindex(g.id).values if m in cv_punto.index.get_level_values(0) else np.full(len(g), np.nan)
    sin0 = g[g.id != id_cero]
    fila = dict(division=dv, modelo=m, mse_extrap=g.e2.mean(), mse_cv_mismos_puntos=np.mean(cv),
                razon_mismos_puntos=g.e2.mean()/np.mean(cv), mse_extrap_sin_cero=sin0.e2.mean(),
                frac_error_3_peores=np.sort(g.e2.values)[-3:].sum()/g.e2.sum())
    for nv, gg in g.groupby('nivel'):
        fila[f'mse_{nv}'] = gg.e2.mean()
    filas.append(fila)
T1 = pd.DataFrame(filas).sort_values(['division', 'mse_extrap']); T1.to_csv(OUT/'extrapolacion_detalle.csv', index=False)

# MDL y promedio de cada cadena del BMS
filas = []
for s, dv in EXT.items():
    for f in sorted((R/'C-007_bms_sensibilidad'/'extrapolacion').glob(f'{dv}_cadena?.json')):
        d = json.loads(f.read_text(encoding='utf-8')); ids = d['cfg']['ids_test']; y = df.loc[ids, 'T'].values
        mtr = df.loc[d['cfg']['ids_train'], 'T'].mean()
        aj = lambda v: np.where(np.isfinite(a := np.array([np.nan if x is None else x for x in v], float)), a, mtr)
        B = np.array([aj(m['pred']) for m in d['muestras']])
        filas.append(dict(division=dv, cadena=f.stem[-1], dl_mdl=d['dl_mdl'], canon=d['canon_mdl'],
                          mse_mdl=np.mean((aj(d['pred_mdl_test']) - y)**2), mse_promedio=np.mean((B.mean(0) - y)**2),
                          mse_mediana=np.mean((np.median(B, 0) - y)**2)))
T2 = pd.DataFrame(filas); T2.to_csv(OUT/'bms_extrapolacion_por_cadena.csv', index=False)

# bootstrap pareado por celdas
rng = np.random.default_rng(20261019)
PARES = [('WC_extremos', 'EBM', 'cuadratico_EN'), ('WC_extremos', 'BMS_MDL', 'EBM'), ('WC_extremos', 'GAM', 'BMS_MDL'),
         ('WC_extremos', 'BMS_MDL', 'cuadratico_EN'), ('WC_extremos', 'BMS_promedio', 'BMS_MDL'),
         ('WT_extremos', 'RF', 'BMS_promedio'), ('WT_extremos', 'BMS_MDL', 'cuadratico_EN')]
filas = []
for dv, a, b in PARES:
    W = E[E.division == dv].pivot(index='id', columns='modelo', values='e2')
    celda = df.loc[W.index, 'celda'].values; cel = np.unique(celda)
    dif = (W[a] - W[b]).values
    sumas = np.array([dif[celda == c].sum() for c in cel]); ns = np.array([(celda == c).sum() for c in cel])
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(cel), len(cel)); bs.append(sumas[k].sum()/ns[k].sum())
    filas.append(dict(division=dv, a=a, b=b, d_media=dif.mean(), ic95_inf=np.percentile(bs, 2.5), ic95_sup=np.percentile(bs, 97.5),
                      n_celdas=len(cel)))
T3 = pd.DataFrame(filas); T3.to_csv(OUT/'bootstrap_celdas.csv', index=False)

# tau de Kendall entre ordenaciones (25 particiones)
res = pd.read_csv(R/'C-009_analisis_sensibilidad'/'mse_modelo_por_variante.csv').set_index('modelo')
filas = []
for v in ['aleatoria', 'wc_nominal', 'sin_cero', 'sin_ef']:
    x = res[['principal', v]].drop(index=['media', 'np4'], errors='ignore').dropna()
    tau, p = stats.kendalltau(x.principal, x[v]); filas.append(dict(variante=v, n_metodos=len(x), tau=tau))
T4 = pd.DataFrame(filas); T4.to_csv(OUT/'kendall.csv', index=False)

with pd.option_context('display.width', 250, 'display.max_columns', 30):
    for t in (T1, T2, T3, T4):
        print(t.round(4).to_string(index=False)); print()
