"""E-018, figura candidata a Fig. 8 (sustituye a la de E-017): (a) fronteras de 5.93 kN; (b) residuo con signo fuera de muestra
cerca de la frontera (niveles que rodean el cruce empírico y el siguiente; CV de C-005 y compacta reajustada por partición),
con IC 95 % por bootstrap de celdas. Requiere haber ejecutado ataque2_oos.py. Salida: resultados/2026-10-19_E-018/Fig_frontera_v2."""
from comun import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E17 = R/'resultados'/'2026-10-19_E-017'
fr = pd.read_csv(E17/'fronteras.csv'); emp = pd.read_csv(E17/'frontera_empirica.csv')
P = pd.read_csv(OUT/'ataque2_residuos.csv'); P = P[P.cerca_B]
MET = [('compacta_OOS', 'Compact form'), ('BMS_MDL', 'BMS-MDL'), ('PySR', 'PySR'), ('cuadratico_EN', 'Quadratic-EN'),
       ('GAM', 'GAM'), ('RF', 'RF'), ('GBM', 'GBM'), ('EBM', 'EBM'), ('TabPFN_v2', 'TabPFN v2')]
rng = np.random.default_rng(0); filas = []
for m, et in MET:
    for tramo in ['WT>=0.12', 'WT<0.12']:
        cm = P[(P.modelo == m) & (P.tramo == tramo)].groupby(['cyc', 'nivel']).r.mean().values
        bs = [rng.choice(cm, len(cm)).mean() for _ in range(4000)]
        filas.append(dict(modelo=m, etiqueta=et, tramo=tramo, n_celdas=len(cm), residuo=cm.mean(),
                          lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
T = pd.DataFrame(filas); T.to_csv(OUT/'fig_residuos_frontera.csv', index=False); print(T.round(3).to_string())

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
ACC, BAND, GRIS, TINTA = '#1f5f8b', '#c9dcea', '#b5b5b5', '#222222'
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
A = ax[0]
A.add_patch(plt.Rectangle((0.04, df.WC.min()), 0.20, df.WC.max() - df.WC.min(), fill=False, ls=':', lw=0.8, ec='#777'))
acc = df['T'] >= UMB
A.scatter(df.WT[~acc], df.WC[~acc], s=9, facecolor='white', edgecolor='#999', lw=0.6, label='Joint < 5.93 kN', zorder=2)
A.scatter(df.WT[acc], df.WC[acc], s=9, color='#999', lw=0, label='Joint ≥ 5.93 kN', zorder=2)
for i, m in enumerate(['cuadratico_EN', 'GAM', 'RF', 'GBM', 'EBM', 'TabPFN_v2', 'PySR', 'BMS_MDL']):
    d = fr[fr.metodo == m]; A.plot(d.WT, d.WCstar, color=GRIS, lw=0.8, label='Other methods' if i == 0 else None, zorder=3)
lo, hi = fr[fr.metodo == 'compacta_part_min'], fr[fr.metodo == 'compacta_part_max']
A.fill_between(lo.WT, lo.WCstar, hi.WCstar, color=BAND, lw=0, zorder=3, label='Compact form, 25 CV refits')
d = fr[fr.metodo == 'compacta_todos']; A.plot(d.WT, d.WCstar, color=ACC, lw=2, label='Compact form, all data', zorder=4)
e = emp[emp.WCstar.notna()]
A.errorbar(e.WT, e.WCstar, yerr=[e.WCstar - e.WC_bajo, e.WC_alto - e.WCstar], fmt='o', ms=4, color=TINTA, lw=0.8, capsize=2,
           label='Empirical crossing (cell means)', zorder=5)
A.set_xlim(0.02, 0.30); A.set_ylim(1.0, 7.5); A.set_xlabel('Weld time WT (s)'); A.set_ylabel('Weld current WC (kA)')
A.set_title('(a) 5.93 kN acceptance boundary', loc='left', fontsize=9); A.grid(color='#eee', lw=0.5)
fig.legend(*A.get_legend_handles_labels(), fontsize=6.5, frameon=False, loc='lower center', ncol=3)
B = ax[1]; y = np.arange(len(MET))[::-1]
for tramo, off, col, mk, lab in [('WT>=0.12', 0.15, TINTA, 'o', 'WT ≥ 0.12 s'), ('WT<0.12', -0.15, '#999', 's', 'WT < 0.12 s')]:
    t = T[T.tramo == tramo].set_index('modelo').loc[[m for m, _ in MET]]
    B.errorbar(t.residuo, y + off, xerr=[t.residuo - t.lo, t.hi - t.residuo], fmt=mk, ms=4, color=col, lw=0.9, capsize=2, label=lab)
B.axvline(0, color='#777', lw=0.8); B.set_yticks(y); B.set_yticklabels([et for _, et in MET])
B.set_xlabel('Out-of-sample residual y − ŷ (kN)'); B.grid(axis='x', color='#eee', lw=0.5)
B.set_title('(b) Bias near the boundary (CV)', loc='left', fontsize=9); B.legend(fontsize=6.5, frameon=False, loc='center right')
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT/'Fig_frontera_v2.png', dpi=200); fig.savefig(OUT/'Fig_frontera_v2.pdf')
