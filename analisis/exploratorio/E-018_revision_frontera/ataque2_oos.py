"""Ataque 2: residuos con signo fuera de muestra cerca de la frontera (C-005 + compacta reajustada por partición, E-010)."""
from comun import *
p = pd.read_csv(R/'resultados'/'C-005_analisis_cv'/'predicciones_todas.csv')
e010 = pd.read_csv(R/'resultados'/'2026-10-18_E-010'/'formas_fijas_por_particion.csv')
pars = {int(r.particion): json.loads(r.parametros) for r in e010[e010.modelo == 'M_A'].itertuples()}
folds = pd.read_csv(R/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
d = df.set_index('id')
rows = []
for k, (a, b) in pars.items():
    ids = folds.query('repeticion == @k // 5 and particion == @k % 5').id.values
    te = d.loc[ids]
    rows.append(pd.DataFrame(dict(modelo='compacta_OOS', particion=k, id=ids, y=te['T'].values, pred=MA(te.WT.values, te.WC.values, a, b))))
    rows.append(pd.DataFrame(dict(modelo='compacta_IN', particion=k, id=ids, y=te['T'].values, pred=MA(te.WT.values, te.WC.values))))
P = pd.concat([p[['modelo', 'particion', 'id', 'y', 'pred']]] + rows)
# check: compacta OOS MSE vs E-010
chk = P[P.modelo == 'compacta_OOS'].groupby('particion').apply(lambda g: np.mean((g.y - g.pred)**2))
print('max |MSE compacta OOS - E-010|:', float(np.max(np.abs(chk.values - e010[e010.modelo=='M_A'].sort_values('particion').mse.values))))
cel = celdas(df); cel['Tc'] = cel['T']
P = P.merge(df[['id', 'cyc', 'nivel', 'WT', 'WC', 'EF']], on='id').merge(cel[['cyc', 'nivel', 'Tc']], on=['cyc', 'nivel'])
# definición B: niveles que rodean el primer cruce empírico por ciclo (y uno más arriba)
emp = pd.read_csv(R/'resultados'/'2026-10-19_E-017'/'frontera_empirica.csv')
brk = {}
for r in emp.itertuples():
    if np.isnan(r.WCstar): continue
    g = cel[cel.cyc == r.ciclos].sort_values('nivel'); i = int(np.argmax(g['T'].values >= UMB))
    brk[r.ciclos] = set(g.nivel.values[max(i-1, 0):i+2])  # nivel bajo, alto y siguiente
P['cerca_A'] = (P.Tc - UMB).abs() < 0.6
P['cerca_B'] = [nv in brk.get(c, set()) for c, nv in zip(P.cyc, P.nivel)]
P['tramo'] = np.where(P.WT >= 0.12, 'WT>=0.12', 'WT<0.12')
P['r'] = P.y - P.pred
P['ok'] = (P.y >= UMB) == (P.pred >= UMB)
P['acept_pred'] = P.pred >= UMB; P['acept_obs'] = P.y >= UMB
orden = ['compacta_OOS', 'compacta_IN', 'BMS_MDL', 'BMS_promedio', 'PySR', 'E001_exploratorio', 'cuadratico_EN', 'GAM', 'EBM', 'RF', 'GBM', 'TabPFN_v2', 'media']
pd.set_option('display.width', 250)
for cdef in ['cerca_A', 'cerca_B']:
    S = P[P[cdef]]
    print(f'\n=== {cdef}: n puntos únicos = {S.id.nunique()} (por tramo: {S.drop_duplicates("id").tramo.value_counts().to_dict()})')
    t = S.groupby(['modelo', 'tramo']).agg(res_medio=('r', 'mean'), rmse=('r', lambda x: np.sqrt(np.mean(x**2))),
                                          acierto=('ok', 'mean'), frac_acept_pred=('acept_pred', 'mean'), frac_acept_obs=('acept_obs', 'mean')).unstack('tramo')
    print(t.reindex(orden).round(3).to_string())
    # IC del residuo medio por bootstrap de celdas (WT>=0.12), para compacta_OOS y RF
    rng = np.random.default_rng(0)
    for m in ['compacta_OOS', 'RF', 'GBM', 'GAM', 'EBM', 'TabPFN_v2', 'BMS_MDL', 'PySR']:
        Q = S[(S.modelo == m) & (S.tramo == 'WT>=0.12')]
        cm = Q.groupby(['cyc', 'nivel']).r.mean().values
        bs = [rng.choice(cm, len(cm)).mean() for _ in range(4000)]
        print(f'  {m:14s} WT>=0.12 residuo medio {cm.mean():+.3f} IC95 celdas [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}] (n celdas {len(cm)})')
# Diferencia pareada compacta - RF etc por punto-partición
S = P[P.cerca_B & (P.tramo == 'WT>=0.12')].pivot_table(index=['particion', 'id'], columns='modelo', values='r')
for m in ['RF', 'GBM', 'GAM', 'EBM', 'TabPFN_v2', 'BMS_MDL', 'PySR']:
    dd = (S['compacta_OOS'].abs() - S[m].abs())
    print(f'|r| compacta - |r| {m}: media {dd.mean():+.3f}; MSE compacta {np.mean(S.compacta_OOS**2):.3f} vs {np.mean(S[m]**2):.3f}')
P.to_csv(OUT/'ataque2_residuos.csv', index=False)
