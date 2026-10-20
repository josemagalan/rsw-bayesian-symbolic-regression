"""Ataque 3: ¿discriminan 9 puntos empíricos la forma del lóbulo WC*^2 = A + B/t frente a alternativas?"""
from comun import *
from scipy.optimize import curve_fit
f1 = pd.read_csv(OUT/'ataque1_fronteras.csv', index_col=0)
modelos = {
 'lobulo WC2=A+B/t':  (lambda t, A, B: np.sqrt(np.clip(A + B/t, 1e-9, None)), [10, 2]),
 'I2t=cte WC2=B/t':   (lambda t, B: np.sqrt(B/t), [3]),
 'WC=A+B/t':          (lambda t, A, B: A + B/t, [4, .1]),
 'WC=A+B*exp(-t/tau)':(lambda t, A, B, tau: A + B*np.exp(-t/tau), [4.5, 20, .03]),
 'WC=A+B*t^-p':       (lambda t, A, B, p: A + B*t**(-p), [4, .01, 2]),
 'WC=A+B*t':          (lambda t, A, B: A + B*t, [6, -8]),
 'constante':         (lambda t, A: A + 0*t, [5]),
 'escalon (corte 0.11)': (None, None),
}
def ajusta(t, y):
    out = []
    n = len(y)
    for nom, (f, p0) in modelos.items():
        if f is None:
            k = 2; yh = np.where(t < 0.11, y[t < 0.11].mean(), y[t >= 0.11].mean())
            loo = []
            for i in range(n):
                m = np.arange(n) != i; tt, yy = t[m], y[m]
                loo.append(y[i] - (yy[tt < 0.11].mean() if t[i] < 0.11 else yy[tt >= 0.11].mean()))
        else:
            k = len(p0)
            try:
                th, _ = curve_fit(f, t, y, p0=p0, maxfev=20000); yh = f(t, *th)
            except Exception as e:
                out.append(dict(modelo=nom, err=str(e)[:40])); continue
            loo = []
            for i in range(n):
                m = np.arange(n) != i
                try: thi, _ = curve_fit(f, t[m], y[m], p0=th, maxfev=20000); loo.append(y[i] - f(t[i], *thi))
                except Exception: loo.append(np.nan)
        rss = float(((y - yh)**2).sum()); k1 = k + 1
        aicc = n*np.log(rss/n) + 2*k1 + 2*k1*(k1 + 1)/max(n - k1 - 1, 1)
        out.append(dict(modelo=nom, k=k, rmse_kA=np.sqrt(rss/n), R2_WC=1 - rss/((y - y.mean())**2).sum(), AICc=aicc,
                        loo_rmse=float(np.sqrt(np.nanmean(np.array(loo)**2)))))
    o = pd.DataFrame(out); o['dAICc'] = o.AICc - o.AICc.min(); return o
pd.set_option('display.width', 200)
for col in ['primer', 'iso', 'logit_comun', 'local_lin4', 'EF1.0', 'EF1.5']:
    s = f1[col].dropna(); s = s[(f1.WT >= 0.04) & (f1.WT <= 0.24)].dropna()
    t, y = f1.loc[s.index, 'WT'].values, s.values
    print(f'\n--- frontera empírica "{col}" (n={len(y)}, WT {t.min()}–{t.max()})'); print(ajusta(t, y).round(3).to_string(index=False))
# R2 del script original (en WC^2, centrado) y su reproducción
t = f1.primer.dropna().index; tt = f1.loc[t, 'WT'].values; y2 = f1.loc[t, 'primer'].values**2
B1, A1 = np.polyfit(1/tt, y2, 1); print('\nReproducción R2 empírica (WC^2):', 1 - ((y2 - A1 - B1/tt)**2).sum()/((y2 - y2.mean())**2).sum())
# R2 de 1/t solo tras 0.12 s (¿hay forma en la parte plana?)
m = tt >= 0.12
from scipy.stats import pearsonr, spearmanr
print('WT>=0.12: corr(WC*^2, 1/WT) =', pearsonr(1/tt[m], y2[m]), ' Spearman', spearmanr(tt[m], np.sqrt(y2[m])))
print('pendiente A+B/t en WT>=0.12:', np.polyfit(1/tt[m], y2[m], 1))
# compacta: R2 restringido a los 9 WT empíricos y a la rejilla 0.08–0.24
for lo in [0.064, 0.08, 0.12]:
    wt = np.round(np.arange(0.02, 0.3001, 0.002), 4); wt = wt[(wt >= lo - 1e-9) & (wt <= 0.24 + 1e-9)]
    w = fr_MA(wt); ok = w <= df.WC.max(); x, yy = 1/wt[ok], w[ok]**2
    b, a = np.polyfit(x, yy, 1); r = yy - a - b*x
    lin = np.polyfit(wt[ok], w[ok], 1); rl = w[ok] - np.polyval(lin, wt[ok])
    print(f'compacta WT>={lo}: R2 lóbulo {1 - (r**2).sum()/((yy - yy.mean())**2).sum():.4f}; R2 de WC=A+B*t (recta) {1 - (rl**2).sum()/((w[ok]-w[ok].mean())**2).sum():.4f}; n={ok.sum()}')
wt9 = f1.WT.values[2:]; w9 = fr_MA(wt9); b, a = np.polyfit(1/wt9, w9**2, 1); r = w9**2 - a - b/wt9
print('compacta en los 9 WT empíricos: R2 lóbulo', 1 - (r**2).sum()/((w9**2 - (w9**2).mean())**2).sum())
