"""C-012: figuras del artículo y del material suplementario, generadas por script (regla 2). No calcula resultados nuevos
(salvo reajustar la forma compacta de E-010 para dibujarla):
solo lee las salidas versionadas (C-001, C-003, C-005, C-006, C-009, C-011, E-012, E-013). Etiquetas en inglés.
  Fig1  datos: TSLBC frente a WC por nivel de WT (rampa secuencial), un panel por EF; umbral de aceptación.
  Fig2  CV: MSE medio ± DE entre particiones por método; suelo de ruido (C-006).
  Fig3  diferencias BMS − comparador (IC 95 % corregido) con la ROPE ±0.03 (C-005).
  Fig4  datos y modelos del análisis A: MDL de A y la forma compacta (a − b/WT)·ln(WC + WT) (E-010, exploratoria).
  Fig5  motivos 1/WT y log en el prior, en el posterior y en los MDL por partición (E-012).
  Fig6  extrapolación por nivel extremo (C-009 + desglose exploratorio de E-013).
  FigS1 trazas de DL de las 8 cadenas de D (C-001) y DL de las 4 cadenas de cada partición (C-003).
  FigS2 residuos e influencia del MDL de A (C-011).
Paleta: la de referencia de la skill de visualización (categórica en orden fijo: azul, naranja, aqua; secuencial azul);
gris neutro para lo que no es BMS. Salida: PDF (vectorial) y PNG 300 ppp en resultados/C-012_figuras/.
Revisión 2026-10-19 (D-016): etiquetas «posterior-sample average» y «estimated noise variance»; C012_OUT opcional.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-012_figuras.py   (después de C-011)"""
import glob, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bms_comun import cargar_datos, ROOT

import os
R = ROOT/'resultados'; OUT = Path(os.environ.get('C012_OUT', R/'C-012_figuras')); OUT.mkdir(parents=True, exist_ok=True)
AZUL, NARANJA, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRIS, GRIS_CLARO, TINTA, TINTA2, REJILLA = '#8a8984', '#c9c8c2', '#0b0b0b', '#52514e', '#e6e5e0'
RAMPA = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b', '#0a2b57']
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7.5, 'axes.edgecolor': GRIS_CLARO, 'axes.labelcolor': TINTA2, 'xtick.color': TINTA2,
                     'ytick.color': TINTA2, 'text.color': TINTA, 'axes.grid': True, 'grid.color': REJILLA, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 1.6,
                     'pdf.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 110})
df = cargar_datos(); df['EF_N'] = (df.EF*1000).round().astype(int)
NOMBRE = {'TabPFN_v2': 'TabPFN v2', 'EBM': 'EBM', 'GBM': 'GBM', 'BMS_promedio': 'BMS posterior-sample average', 'RF': 'Random forest',
          'GAM': 'GAM', 'cuadratico_EN': 'Quadratic (elastic net)', 'BMS_MDL': 'BMS MDL', 'E001_exploratorio': 'Saturation (E-001)',
          'PySR': 'PySR', 'media': 'Mean'}
UMBRAL, SIGMA2, ROPE = 5.93, 0.066, 0.03
hechas = []


def guardar(fig, nombre):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'{nombre}.{ext}', bbox_inches='tight')
    plt.close(fig); hechas.append(nombre)


# ---------------------------------------------------------------- Fig1: datos
fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
for ax, ef in zip(axs, [1000, 1500]):
    for i, cyc in enumerate(range(2, 13)):
        g = df[(df.EF_N == ef) & (df.cyc == cyc)].sort_values('WC')
        ax.plot(g.WC, g['T'], color=RAMPA[i], lw=1.2, marker='o', ms=3, mec='white', mew=0.4, zorder=2)
    ax.axhline(UMBRAL, color=TINTA2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_title(f'EF = {ef} N', loc='left'); ax.set_xlabel('Welding current WC (kA)')
    ax.text(1.4, UMBRAL + 0.12, 'acceptance 5.93 kN', color=TINTA2, fontsize=7)
axs[0].set_ylabel('TSLBC (kN)')
sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(RAMPA), norm=matplotlib.colors.Normalize(0.03, 0.25))
cb = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02, ticks=[0.04, 0.08, 0.12, 0.16, 0.20, 0.24]); cb.set_label('Welding time WT (s)')
cb.outline.set_visible(False)
guardar(fig, 'Fig1_datos')

# ---------------------------------------------------------------- Fig2: MSE en la CV
res = pd.read_csv(R/'C-005_analisis_cv'/'resumen_modelos.csv'); M = pd.read_csv(R/'C-005_analisis_cv'/'metricas_por_particion.csv')
res = res[res.modelo != 'media'].sort_values('mse', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(3.6, 3.1))
ax.axvspan(0, SIGMA2, color=REJILLA, lw=0, zorder=0)
ax.text((0.05 + SIGMA2)/2, (len(res) - 1)/2, 'estimated noise variance σ̂² ≈ 0.066 kN²', rotation=90, ha='center', va='center', fontsize=6.5,
        color=TINTA2)
for i, r in res.iterrows():
    col = AZUL if r.modelo.startswith('BMS') else GRIS
    pts = M[M.modelo == r.modelo].mse.values
    ax.scatter(pts, np.full(len(pts), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(pts)), s=5, color=col, alpha=0.25, lw=0)
    ax.errorbar(r.mse, i, xerr=r.mse_de, fmt='o', ms=5, color=col, ecolor=col, elinewidth=1.4, capsize=0, mec='white', mew=0.6)
ax.set_yticks(range(len(res))); ax.set_yticklabels([NOMBRE[m] for m in res.modelo]); ax.grid(axis='y', visible=False)
ax.set_xlabel('Out-of-sample MSE (kN²), 5×5 grouped CV'); ax.set_xlim(0.05, None)
guardar(fig, 'Fig2_mse_cv')

# ---------------------------------------------------------------- Fig3: diferencias con ROPE
C = pd.read_csv(R/'C-005_analisis_cv'/'comparaciones.csv')
orden = [m for m in ['TabPFN_v2', 'EBM', 'GBM', 'RF', 'GAM', 'cuadratico_EN', 'E001_exploratorio', 'PySR'] if m in set(C.comparador)]
fig, ax = plt.subplots(figsize=(3.6, 3.0))
ax.axvspan(-ROPE, ROPE, color=REJILLA, lw=0, zorder=0); ax.axvline(0, color=GRIS_CLARO, lw=0.8)
for k, (b, col, dy) in enumerate([('BMS_MDL', AZUL, -0.14), ('BMS_promedio', NARANJA, 0.14)]):
    for i, c in enumerate(orden[::-1]):
        r = C[(C.bms == b) & (C.comparador == c)]
        if len(r):
            r = r.iloc[0]
            ax.errorbar(r.d_media, i + dy, xerr=[[r.d_media - r.ic95_inf], [r.ic95_sup - r.d_media]], fmt='o', ms=4.5, color=col,
                        elinewidth=1.4, capsize=0, mec='white', mew=0.5)
ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE[c] for c in orden[::-1]]); ax.grid(axis='y', visible=False)
ax.set_xlabel('MSE difference, BMS − comparator (kN²)\n95% CI, corrected resampled t')
ax.text(ROPE - 0.002, -0.55, 'ROPE ±0.03', ha='right', va='bottom', fontsize=6.5, color=TINTA2)
ax.legend(handles=[Line2D([], [], color=AZUL, marker='o', lw=1.4, label='BMS MDL'),
                   Line2D([], [], color=NARANJA, marker='o', lw=1.4, label='BMS posterior-sample average')],
          loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=2, handlelength=1.2)
guardar(fig, 'Fig3_diferencias_rope')

# ---------------------------------------------------------------- Fig4: modelos del análisis A
WT_s, WC_s = sp.symbols('WT WC')
caD = []
for f in sorted(glob.glob(str(R/'C-001_analisis_A'/'D_cadena*.json'))):
    if not f.endswith('.progreso.json'):
        caD.append(json.loads(Path(f).read_text(encoding='utf-8')))
mA = min(caD, key=lambda d: d['dl_mdl']); ex = sp.sympify(mA['modelo_mdl'])
us = sorted(str(s) for s in ex.free_symbols if str(s) not in ('WT', 'WC'))
fA = sp.lambdify([WT_s, WC_s] + [sp.Symbol(u) for u in us], ex, 'numpy'); thA = [mA['parametros_mdl']['d0'][u] for u in us]
from scipy.optimize import curve_fit
fC = lambda X, a, b: (a - b/X[0])*np.log(X[1] + X[0])
thC, _ = curve_fit(fC, np.vstack([df.WT, df.WC]), df['T'].values, p0=[3.9, 0.05])
fig, axs = plt.subplots(1, 4, figsize=(7, 2.3), sharey=True)
wc = np.linspace(1.4, 6.6, 200)
for ax, cyc in zip(axs, [2, 4, 7, 12]):
    g = df[df.cyc == cyc]; wt = cyc*0.02
    ax.scatter(g.WC, g['T'], s=9, color=GRIS, lw=0, alpha=0.8, zorder=2)
    ax.plot(wc, fA(np.full_like(wc, wt), wc, *thA), color=AZUL, zorder=3)
    ax.plot(wc, fC(np.vstack([np.full_like(wc, wt), wc]), *thC), color=NARANJA, ls=(0, (5, 2)), zorder=3)
    ax.set_title(f'WT = {wt:.2f} s', loc='left'); ax.set_xlabel('WC (kA)')
axs[0].set_ylabel('TSLBC (kN)')
axs[-1].legend(handles=[Line2D([], [], color=AZUL, label=f'MDL, analysis A ({len(us)} parameters)'),
                        Line2D([], [], color=NARANJA, ls=(0, (5, 2)), label='(a − b/WT)·ln(WC + WT), 2 parameters'),
                        Line2D([], [], color=GRIS, marker='o', lw=0, ms=3, label='Data (both EF)')],
               loc='upper right', bbox_to_anchor=(1.0, -0.3), ncol=3)
guardar(fig, 'Fig4_modelos_A')

# ---------------------------------------------------------------- Fig5: motivos frente al prior
E12 = pd.read_csv(R/'2026-10-18_E-012'/'resumen.csv')
fuentes = [('prior (sin datos)', 'Prior (no data)', GRIS_CLARO), ('posterior CV (muestras C-003)', 'Posterior samples (CV)', AQUA),
           ('MDL por partición (C-005)', 'MDL per partition (CV)', AZUL)]
motivos = [('inv_WT_si_ambas', '1/WT term'), ('log_si_ambas', 'log term'), ('ambos_motivos_si_ambas', 'Both')]
fig, ax = plt.subplots(figsize=(3.6, 2.4)); x = np.arange(len(motivos)); w = 0.26
for k, (fu, et, col) in enumerate(fuentes):
    v = E12[E12.fuente == fu].iloc[0]
    ax.bar(x + (k - 1)*w, [v[m] for m, _ in motivos], w - 0.03, color=col, label=et, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([e for _, e in motivos]); ax.grid(axis='x', visible=False)
ax.set_ylabel('Fraction of expressions\nthat use both WT and WC'); ax.set_ylim(0, 1)
ax.legend(loc='upper right')
guardar(fig, 'Fig5_motivos_prior')

# ---------------------------------------------------------------- Fig6: extrapolación por nivel
X13 = pd.read_csv(R/'2026-10-19_E-013'/'extrapolacion_detalle.csv')
fil = lambda dv: X13[(X13.division == dv) & (X13.modelo != 'media')].set_index('modelo')
paneles = [('WC_extremos', 'Held-out current extremes', ('mse_nivel_WC_1', 'lowest level (1.4–1.7 kA)'),
            ('mse_nivel_WC_11', 'highest level (6.4–6.6 kA)')),
           ('WT_extremos', 'Held-out time extremes', ('mse_ciclos_2', '2 cycles (0.04 s)'), ('mse_ciclos_12', '12 cycles (0.24 s)'))]
orden = fil('WC_extremos').sort_values('mse_extrap').index.tolist()
fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
for ax, (dv, tit, (c1, e1), (c2, e2)) in zip(axs, paneles):
    t = fil(dv)
    for i, m in enumerate(orden[::-1]):
        ax.plot([t.loc[m, c1], t.loc[m, c2]], [i, i], color=GRIS_CLARO, lw=1.0, zorder=1)
        ax.scatter(t.loc[m, c1], i, s=24, color=AZUL, zorder=2, edgecolor='white', lw=0.5)
        ax.scatter(t.loc[m, c2], i, s=24, color=NARANJA, zorder=2, edgecolor='white', lw=0.5)
    ax.set_title(tit, loc='left'); ax.grid(axis='y', visible=False); ax.set_xlim(0, None)
    ax.set_xlabel('MSE on the held-out level (kN²)')
    ax.legend(handles=[Line2D([], [], color=AZUL, marker='o', lw=0, label=e1), Line2D([], [], color=NARANJA, marker='o', lw=0, label=e2)],
              loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3)
axs[0].set_yticks(range(len(orden))); axs[0].set_yticklabels([NOMBRE[m] for m in orden[::-1]])
for lab in axs[0].get_yticklabels():
    if lab.get_text().startswith('BMS'):
        lab.set_fontweight('bold'); lab.set_color(TINTA)
guardar(fig, 'Fig6_extrapolacion')

# ---------------------------------------------------------------- FigS1: trazas y DL por cadena
fig, axs = plt.subplots(1, 2, figsize=(7, 2.7), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax = axs[0]
for d in caD:
    tr = np.array(d['traza_cada10'], float); es = d is mA
    ax.plot(np.arange(len(tr))*10, tr, color=AZUL if es else GRIS_CLARO, lw=0.9 if es else 0.6, zorder=3 if es else 2)
ax.set_ylim(90, 260); ax.set_xlabel('MCMC step'); ax.set_ylabel('Description length at T = 1')
ax.set_title('Analysis A: 8 chains (MDL chain in blue)', loc='left')
ax = axs[1]; B = pd.read_csv(R/'C-005_analisis_cv'/'bms_cadenas.csv')
for p, g in B.groupby('particion'):
    ax.scatter(np.full(len(g), p), g.dl_mdl, s=8, color=GRIS, lw=0, zorder=2)
    ax.scatter(p, g.dl_mdl.min(), s=14, color=AZUL, zorder=3, edgecolor='white', lw=0.4)
ax.set_xlabel('CV partition'); ax.set_ylabel('Best DL per chain'); ax.set_title('CV: 4 chains per partition (minimum in blue)', loc='left')
guardar(fig, 'FigS1_cadenas')

# ---------------------------------------------------------------- FigS2: residuos del MDL de A
RA = pd.read_csv(R/'C-011_informe_s10'/'residuos_MDL_A.csv'); n = len(RA)
fig, axs = plt.subplots(1, 4, figsize=(7, 2.0))
for ax, (xc, xl) in zip(axs[:3], [('prediccion', 'Fitted TSLBC (kN)'), ('WT', 'WT (s)'), ('WC', 'WC (kA)')]):
    ax.axhline(0, color=GRIS_CLARO, lw=0.8); ax.scatter(RA[xc], RA.residuo, s=6, color=AZUL, lw=0, alpha=0.7)
    ax.set_xlabel(xl)
axs[0].set_ylabel('Residual (kN)')
ax = axs[3]; ax.vlines(np.arange(n), 0, RA.cook, color=AZUL, lw=0.7); ax.axhline(4/n, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
ax.set_xlabel('Observation'); ax.set_ylabel("Cook's distance"); ax.text(n, 4/n, ' 4/n', va='bottom', ha='right', fontsize=6.5, color=TINTA2)
fig.tight_layout(w_pad=1.2)
guardar(fig, 'FigS2_residuos_A')

(OUT/'indice.json').write_text(json.dumps(dict(figuras=hechas, parametros_forma_compacta=list(map(float, thC))), indent=1),
                               encoding='utf-8')
print('Figuras:', ', '.join(hechas))
