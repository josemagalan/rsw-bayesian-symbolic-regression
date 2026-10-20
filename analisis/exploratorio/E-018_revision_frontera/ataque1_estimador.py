"""Ataque 1: robustez del estimador de la frontera empírica y sesgo de la compacta con IC."""
from comun import *
from scipy.optimize import minimize
rng = np.random.default_rng(1)
cy = sorted(df.cyc.unique())
def fronteras(d, metodo='primer'):
    c = celdas(d); out = {}
    for k, g in c.groupby('cyc'):
        g = g.sort_values('nivel'); wc, t = g.WC.values, g['T'].values
        if metodo == 'primer': out[k] = primer_cruce(wc, t)
        elif metodo == 'ultimo': out[k] = ultimo_cruce(wc, t)
        elif metodo == 'iso': out[k] = primer_cruce(wc, pava(t))
    return pd.Series(out)
WT = df.groupby('cyc').WT.first()
res = pd.DataFrame({'WT': WT, 'primer': fronteras(df), 'ultimo': fronteras(df, 'ultimo'), 'iso': fronteras(df, 'iso'),
                    'EF1.0': fronteras(df[df.EF == 1.0]), 'EF1.5': fronteras(df[df.EF == 1.5]),
                    'EF1.0_iso': fronteras(df[df.EF == 1.0], 'iso'), 'EF1.5_iso': fronteras(df[df.EF == 1.5], 'iso')})
# logística: logit p = alpha_cyc + beta*WC (pendiente común), por ciclo WC50 = -alpha/beta
acc = (df['T'] >= UMB).astype(float).values; ci = df.cyc.map({k: i for i, k in enumerate(cy)}).values; wc = df.WC.values
def nll(th):
    al, be = th[:-1], th[-1]; z = al[ci] + be*wc
    return np.sum(np.logaddexp(0, z) - acc*z) + 1e-4*np.sum(th**2)
th = minimize(nll, np.r_[np.full(len(cy), -10.0), 2.0], method='BFGS').x
res['logit_comun'] = pd.Series(-th[:-1]/th[-1], index=cy)
# probit/normal: T ~ cruce desde regresión local lineal en WC de los 4 niveles que rodean el umbral (ajuste local)
def local_lin(d):
    out = {}
    for k, g in celdas(d).groupby('cyc'):
        g = g.sort_values('nivel'); w, t = g.WC.values, g['T'].values
        x = primer_cruce(w, t)
        if np.isnan(x): out[k] = np.nan; continue
        m = np.argsort(np.abs(w - x))[:4]; s, i0 = np.polyfit(w[m], t[m], 1)
        out[k] = (UMB - i0)/s
    return pd.Series(out)
res['local_lin4'] = local_lin(df)
res['compacta'] = fr_MA(res.WT.values)
print(res.round(3).to_string())
L = res.WT >= 0.12
for c in ['primer', 'ultimo', 'iso', 'EF1.0', 'EF1.5', 'EF1.0_iso', 'EF1.5_iso', 'logit_comun', 'local_lin4']:
    d = (res.compacta - res[c]); ok = d.notna()
    print(f"{c:12s} sesgo compacta {d[ok].mean():+.3f}  (WT>=0.12: {d[ok & L].mean():+.3f}, n={int((ok&L).sum())})  RMSE {np.sqrt((d[ok]**2).mean()):.3f}")
# bootstrap: (i) remuestreo de gemelas dentro de celda; (ii) bootstrap por celdas completas no tiene sentido en una curva 1D,
# así que (iii) bootstrap paramétrico: medias de celda + N(0, 0.26^2/2)
B = 2000; bs = {'gemelas': [], 'param': [], 'gemelas_iso': []}
cel = celdas(df)
for b in range(B):
    idx = df.groupby('celda').sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).index
    d = df.loc[idx]
    f1 = fronteras(d); f2 = fronteras(d, 'iso')
    bs['gemelas'].append(f1.values); bs['gemelas_iso'].append(f2.values)
    c2 = cel.copy(); c2['T'] = c2['T'] + rng.normal(0, 0.26/np.sqrt(2), len(c2))
    bs['param'].append(np.array([primer_cruce(g.sort_values('nivel').WC.values, g.sort_values('nivel')['T'].values) for _, g in c2.groupby('cyc')]))
Lv = (res.WT >= 0.12).values; comp = res.compacta.values
for k, v in bs.items():
    v = np.array(v)
    lo, hi = np.nanpercentile(v, [2.5, 97.5], axis=0)
    bias_L = np.nanmean(comp[Lv] - v[:, Lv], axis=1)
    bias_all = np.nanmean((comp - v)[:, 2:], axis=1)
    print(f"\nBootstrap {k}: IC95 por ciclo (cyc, lo, hi, compacta):")
    for j, c in enumerate(cy): print(f"  {c:2d} {lo[j]:.2f} {hi[j]:.2f}  comp {comp[j]:.2f}  {'FUERA' if comp[j] > hi[j] else ''}")
    print(f"  sesgo medio WT>=0.12: mediana {np.median(bias_L):+.3f}, IC95 [{np.percentile(bias_L,2.5):+.3f}, {np.percentile(bias_L,97.5):+.3f}]; "
          f"todos: IC95 [{np.nanpercentile(bias_all,2.5):+.3f}, {np.nanpercentile(bias_all,97.5):+.3f}]; frac NaN {np.isnan(v).mean():.3f}")
res.to_csv(OUT/'ataque1_fronteras.csv')
