"""E-017 (EXPLORATORIO, post hoc; 2026-10-19): frontera de aceptación (TSLBC = 5.93 kN) en forma cerrada y su relación con la
frontera de corriente mínima del lóbulo de soldabilidad con pérdidas, I² = A + B/t (Kim y Eagar 1988).
  1. Forma compacta M_A = (a − b/WT)·ln(WC + WT): despejada, WC*(WT) = exp(5.93/(a − b/WT)) − WT (si a − b/WT > 0).
     Parámetros: ajuste con los 242 puntos y, como banda, los 25 juegos de E-010 (reajuste por partición).
  2. Frontera empírica sin modelo: por ciclo, medias de las gemelas por nivel de corriente; primer cruce ascendente de 5.93 kN,
     interpolando linealmente en WC (horquilla = WC de los dos niveles que lo rodean).
  3. Fronteras de los demás métodos: primer cruce de 5.93 kN en la rejilla de E-014 (ajustes con todos los datos).
  4. Ajuste de la forma del lóbulo WC*² = A + B/WT (y de I²t = cte, A = 0) a cada frontera dentro del dominio ensayado.
  5. Acuerdo entre fronteras: RMSE (kA) frente a la frontera empírica y frente a la de la forma compacta.
Contaminación: M_A se eligió con todos los datos (E-009/E-010); la frontera empírica usa los mismos datos. Es una ilustración
de uso de una expresión cerrada, no una validación.
Uso: python analisis/exploratorio/E-017_frontera_aceptacion.py     (salida en resultados/2026-10-19_E-017)"""
import json, platform, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from bms_comun import cargar_datos, ROOT
warnings.filterwarnings('ignore')

OUT = ROOT/'resultados'/'2026-10-19_E-017'; OUT.mkdir(parents=True, exist_ok=True)
UMB = 5.93
df = cargar_datos()
WCMIN, WCMAX = df.WC.min(), df.WC.max()
MA = lambda X, a, b: (a - b/X[0])*np.log(X[1] + X[0])


def frontera_MA(wt, a, b):
    den = a - b/wt
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        w = np.exp(UMB/den) - wt
    return np.where(den > 0, w, np.inf)


def primer_cruce(wc, pred):
    """WC del primer cruce ascendente de UMB (interpolación lineal); NaN si no cruza."""
    for i in range(1, len(wc)):
        if pred[i-1] < UMB <= pred[i]:
            return wc[i-1] + (UMB - pred[i-1])*(wc[i] - wc[i-1])/(pred[i] - pred[i-1])
    return np.nan


def ajuste_lobulo(wt, wc):
    ok = np.isfinite(wc) & (wc >= WCMIN) & (wc <= WCMAX)
    x, y = 1/wt[ok], wc[ok]**2
    if ok.sum() < 3:
        return dict(n=int(ok.sum()))
    B1, A1 = np.polyfit(x, y, 1); r1 = y - (A1 + B1*x)
    B0 = (x @ y)/(x @ x); r0 = y - B0*x
    ss = ((y - y.mean())**2).sum()
    rmse_kA = lambda A, B: float(np.sqrt(np.mean((np.sqrt(np.clip(A + B*x, 0, None)) - wc[ok])**2)))
    return dict(n=int(ok.sum()), A=float(A1), B=float(B1), R2=float(1 - (r1**2).sum()/ss), rmse_kA=rmse_kA(A1, B1),
                B_I2t=float(B0), R2_I2t=float(1 - (r0**2).sum()/ss), rmse_kA_I2t=rmse_kA(0.0, B0),
                WT_min=float(wt[ok].min()), WT_max=float(wt[ok].max()))


if __name__ == '__main__':
    res = {}
    # 1. forma compacta
    X = (df.WT.values, df.WC.values)
    (a, b), _ = curve_fit(MA, X, df['T'].values, p0=[3.9, 0.047])
    res['MA_todos'] = dict(a=float(a), b=float(b))
    e010 = pd.read_csv(ROOT/'resultados'/'2026-10-18_E-010'/'formas_fijas_por_particion.csv')
    pars = np.array([json.loads(p) for p in e010.loc[e010.modelo == 'M_A', 'parametros']])
    WTg = np.round(np.arange(0.02, 0.3001, 0.002), 4)
    f_all = frontera_MA(WTg, a, b)
    f_part = np.array([frontera_MA(WTg, *p) for p in pars])
    filas = [dict(WT=w, metodo='compacta_todos', WCstar=v) for w, v in zip(WTg, f_all)]
    for q, nom in [(0, 'compacta_part_min'), (50, 'compacta_part_mediana'), (100, 'compacta_part_max')]:
        filas += [dict(WT=w, metodo=nom, WCstar=v) for w, v in zip(WTg, np.percentile(f_part, q, axis=0))]

    # 2. frontera empírica (medias de gemelas por celda)
    cel = df.groupby(['cyc', 'nivel']).agg(WT=('WT', 'first'), WC=('WC', 'mean'), T=('T', 'mean')).reset_index()
    emp = []
    for cyc, g in cel.groupby('cyc'):
        g = g.sort_values('nivel'); wc, t = g.WC.values, g['T'].values
        x = primer_cruce(wc, t)
        i = np.argmax(t >= UMB) if (t >= UMB).any() else None
        emp.append(dict(ciclos=int(cyc), WT=float(g.WT.iloc[0]), WCstar=float(x),
                        WC_bajo=float(wc[i-1]) if i else np.nan, WC_alto=float(wc[i]) if i is not None else np.nan,
                        n_no_monot_sobre_umbral=int(((t[i:] < UMB).sum()) if i is not None else 0)))
    emp = pd.DataFrame(emp); emp.to_csv(OUT/'frontera_empirica.csv', index=False)

    # 3. demás métodos (rejilla de E-014)
    sup = pd.read_csv(ROOT/'resultados'/'2026-10-19_E-014'/'superficies.csv')
    metodos = [c for c in sup.columns if c not in ('WT', 'WC')]
    fr = {}
    for m in metodos:
        v = []
        for wt, g in sup.groupby('WT'):
            g = g.sort_values('WC'); v.append((wt, primer_cruce(g.WC.values, g[m].values)))
        fr[m] = pd.DataFrame(v, columns=['WT', 'WCstar'])
        filas += [dict(WT=w, metodo=m, WCstar=s) for w, s in v]
    pd.DataFrame(filas).to_csv(OUT/'fronteras.csv', index=False)

    # 4. forma del lóbulo, dentro del dominio de tiempo (0.04–0.24 s)
    dom = lambda wt: (wt >= 0.04 - 1e-9) & (wt <= 0.24 + 1e-9)
    lob = {'empirica': ajuste_lobulo(emp.WT.values, emp.WCstar.values),
           'compacta_todos': ajuste_lobulo(WTg[dom(WTg)], f_all[dom(WTg)])}
    for m, d in fr.items():
        d = d[dom(d.WT.values)]; lob[m] = ajuste_lobulo(d.WT.values, d.WCstar.values)
    lp = np.array([[ajuste_lobulo(WTg[dom(WTg)], fp[dom(WTg)]).get(k, np.nan) for k in ('A', 'B', 'R2')] for fp in f_part])
    res['lobulo_compacta_particiones'] = dict(A_rango=[float(np.nanmin(lp[:, 0])), float(np.nanmax(lp[:, 0]))],
                                              B_rango=[float(np.nanmin(lp[:, 1])), float(np.nanmax(lp[:, 1]))],
                                              R2_min=float(np.nanmin(lp[:, 2])))
    pd.DataFrame(lob).T.to_csv(OUT/'ajuste_lobulo.csv')
    res['lobulo'] = lob

    # 5. acuerdo entre fronteras en los tiempos ensayados
    ok = emp.WCstar.notna()
    acu = {}
    for m, d in list(fr.items()) + [('compacta_todos', pd.DataFrame({'WT': WTg, 'WCstar': f_all}))]:
        fi = np.interp(emp.WT[ok], d.WT, d.WCstar)
        fc = np.interp(emp.WT[ok], WTg, f_all)
        acu[m] = dict(rmse_vs_empirica_kA=float(np.sqrt(np.nanmean((fi - emp.WCstar[ok])**2))),
                      sesgo_vs_empirica_kA=float(np.nanmean(fi - emp.WCstar[ok])),
                      sesgo_WT_ge_012_kA=float(np.nanmean((fi - emp.WCstar[ok])[emp.WT[ok].values >= 0.12])),
                      rmse_vs_compacta_kA=float(np.sqrt(np.nanmean((fi - fc)**2))))
    res['acuerdo'] = acu
    band = np.interp(emp.WT[ok], WTg, np.percentile(f_part, 100, axis=0)) - np.interp(emp.WT[ok], WTg, np.percentile(f_part, 0, axis=0))
    res['ancho_banda_particiones_kA'] = dict(mediana=float(np.median(band)), max=float(band.max()))
    res['entorno'] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, plataforma=platform.platform())
    json.dump(res, open(OUT/'resumen.json', 'w'), indent=1, default=float)

    # Figura
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    ACC, BAND, GRIS, TINTA = '#1f5f8b', '#c9dcea', '#b5b5b5', '#222222'
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6))
    A = ax[0]
    A.add_patch(plt.Rectangle((0.04, WCMIN), 0.20, WCMAX - WCMIN, fill=False, ls=':', lw=0.8, ec='#777'))
    acc = df['T'] >= UMB
    A.scatter(df.WT[~acc], df.WC[~acc], s=9, facecolor='white', edgecolor='#999', lw=0.6, label='Joint < 5.93 kN', zorder=2)
    A.scatter(df.WT[acc], df.WC[acc], s=9, color='#999', lw=0, label='Joint ≥ 5.93 kN', zorder=2)
    for i, (m, d) in enumerate(fr.items()):
        if m in ('compacta', 'BMS_MDL_CV_mediana'):
            continue
        A.plot(d.WT, d.WCstar, color=GRIS, lw=0.8, label='Other methods' if i == 0 else None, zorder=3)
    A.fill_between(WTg, np.percentile(f_part, 0, axis=0), np.percentile(f_part, 100, axis=0), color=BAND, lw=0, zorder=3,
                   label='Compact form, 25 CV refits')
    A.plot(WTg, f_all, color=ACC, lw=2, label='Compact form, all data', zorder=4)
    e = emp[ok]
    A.errorbar(e.WT, e.WCstar, yerr=[e.WCstar - e.WC_bajo, e.WC_alto - e.WCstar], fmt='o', ms=4, color=TINTA, lw=0.8,
               capsize=2, label='Empirical crossing (cell means)', zorder=5)
    A.set_xlim(0.02, 0.30); A.set_ylim(1.0, 7.5); A.set_xlabel('Weld time WT (s)'); A.set_ylabel('Weld current WC (kA)')
    A.set_title('(a) 5.93 kN acceptance boundary', loc='left', fontsize=9)
    A.grid(color='#eee', lw=0.5)
    fig.legend(*A.get_legend_handles_labels(), fontsize=6.5, frameon=False, loc='lower center', ncol=3)
    B = ax[1]
    L = lob['compacta_todos']
    m = (WTg >= L['WT_min'] - 1e-9) & (WTg <= 0.24 + 1e-9)
    B.plot(1/WTg[m], f_all[m]**2, color=ACC, lw=2, label='Compact form')
    B.plot(1/e.WT, e.WCstar**2, 'o', ms=4, color=TINTA, label='Empirical crossing')
    xs = np.linspace(1/0.24, 1/L['WT_min'], 50)
    B.plot(xs, L['A'] + L['B']*xs, ls='--', color='#d0782a', lw=1.2, label=f"I² = A + B/t (R² = {L['R2']:.3f})")
    B.plot(xs, L['B_I2t']*xs, ls=':', color='#777', lw=1.2, label=f"I²t = const (R² = {L['R2_I2t']:.2f})")
    B.set_xlabel('1 / WT (s⁻¹)'); B.set_ylabel('WC*² (kA²)'); B.set_title('(b) Lobe-boundary form', loc='left', fontsize=9)
    B.grid(color='#eee', lw=0.5); B.legend(fontsize=6.5, frameon=False, loc='upper left')
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT/'Fig_frontera.png', dpi=200); fig.savefig(OUT/'Fig_frontera.pdf')
    print(json.dumps({k: res[k] for k in ('MA_todos', 'lobulo_compacta_particiones', 'ancho_banda_particiones_kA', 'acuerdo')}, indent=1, default=float))
    print(pd.DataFrame(lob).T.round(4).to_string())
    print(emp.round(3).to_string())
