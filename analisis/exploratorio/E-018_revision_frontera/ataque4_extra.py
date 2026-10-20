"""Ataque 3b/4: I^2 (t - t0) = K, I^n t = K, rejilla de E-014 y comprobaciones de código."""
from comun import *
from scipy.optimize import curve_fit
f1 = pd.read_csv(OUT/'ataque1_fronteras.csv', index_col=0)
s = f1.primer.dropna(); t = f1.loc[s.index, 'WT'].values; y = s.values
for nom, f, p0 in [('WC^2=K/(t-t0)', lambda t, K, t0: np.sqrt(K/np.clip(t - t0, 1e-6, None)), [2, 0.03]),
                   ('WC^n t = K (n libre)', lambda t, K, n: (K/t)**(1/n), [3, 2]),
                   ('lobulo', lambda t, A, B: np.sqrt(A + B/t), [10, 2])]:
    th, _ = curve_fit(f, t, y, p0=p0, maxfev=20000); r = y - f(t, *th)
    print(f'{nom:22s} th={np.round(th,3)} rmse={np.sqrt(np.mean(r**2)):.3f} R2={1-(r**2).sum()/((y-y.mean())**2).sum():.3f}')
sup = pd.read_csv(R/'resultados'/'2026-10-19_E-014'/'superficies.csv')
wt = np.sort(sup.WT.unique()); wc = np.sort(sup.WC.unique())
print('rejilla E-014: nWT', len(wt), 'paso', np.round(np.diff(wt)[:3], 4), '¿contiene 0.08..0.24?', np.isin(np.round(f1.WT.values, 4), np.round(wt, 4)).all(),
      'nWC', len(wc), 'paso', np.round(np.diff(wc)[:2], 3), 'WC rango', wc.min(), wc.max())
# Rejilla: ¿cuántos puntos de frontera de cada método en 0.08–0.24 caen en el mismo escalón (árboles)?
fr = pd.read_csv(R/'resultados'/'2026-10-19_E-017'/'fronteras.csv')
for m in ['RF', 'GBM', 'EBM', 'GAM', 'TabPFN_v2', 'compacta']:
    d = fr[(fr.metodo == m) & (fr.WT >= 0.08) & (fr.WT <= 0.24)].WCstar.round(3)
    print(f'{m:10s} n={len(d)} valores distintos={d.nunique()}  lag-1 autocorr de residuos lóbulo:', end=' ')
    tt = fr[(fr.metodo == m) & (fr.WT >= 0.08) & (fr.WT <= 0.24)].WT.values; yy = d.values; ok = np.isfinite(yy)
    b, a = np.polyfit(1/tt[ok], yy[ok]**2, 1); r = yy[ok]**2 - a - b/tt[ok]; print(round(np.corrcoef(r[:-1], r[1:])[0, 1], 2))
# cyc 7: sensibilidad al no monotónico
c = celdas(df); print(c[c.cyc == 7][['nivel', 'WC', 'T']].round(3).T.to_string())
