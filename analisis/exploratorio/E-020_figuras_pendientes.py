"""E-020 (exploratorio): figuras pendientes del borrador. No ajusta modelos: solo lee salidas guardadas y evalúa
expresiones ya ajustadas sobre los datos. Estilo y paleta de C-012 (azul/naranja/aqua validados; gris para no simbólicos).
  Fig7_efectos_parciales  dependencia parcial en WC y en WT: 25 MDL del BMS (C-003, vía E-015/ecuaciones.csv),
                          25 PySR (C-002), PDP de GAM, EBM, RF, GBM y TabPFN (E-015/pdp.csv) y medias marginales de los datos.
  FigS1_sintetico         estudio sintético previo a la congelación (E-005, E-006): ΔL = L(mejor) − L(verdad) y RMSE funcional.
  FigS4_umbral            MSE de CV de formas cerradas con y sin desplazamiento en WC, y desplazamiento c por partición (E-016).
La dependencia parcial de una expresión simbólica se calcula como la de los modelos no simbólicos en E-015: media de f sobre
las 242 filas de datos con la variable fijada en cada punto de la rejilla nominal.
Uso: python analisis/exploratorio/E-020_figuras_pendientes.py"""
import glob, json, os, platform, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bms_comun import cargar_datos, ROOT

R = ROOT/'resultados'; OUT = R/'2026-10-19_E-020'; OUT.mkdir(parents=True, exist_ok=True)
AZUL, NARANJA, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRIS, GRIS_CLARO, TINTA, TINTA2, REJILLA = '#8a8984', '#c9c8c2', '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7.5, 'axes.edgecolor': GRIS_CLARO, 'axes.labelcolor': TINTA2, 'xtick.color': TINTA2,
                     'ytick.color': TINTA2, 'text.color': TINTA, 'axes.grid': True, 'grid.color': REJILLA, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 1.6,
                     'pdf.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 110})


def guardar(fig, nombre):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'{nombre}.{ext}', bbox_inches='tight')
    plt.close(fig)


df = cargar_datos()
GWC = np.round(np.arange(1.5, 6.51, 0.5), 2); GWT = np.round(np.arange(0.04, 0.241, 0.02), 2)
WT_s, WC_s = sp.symbols('WT WC')

# ---------------------------------------------------------------- Fig7: efectos parciales
EQ = pd.read_csv(R/'2026-10-19_E-015'/'ecuaciones.csv')
PDP = pd.read_csv(R/'2026-10-19_E-015'/'pdp.csv')


def dep_parcial(expr):
    f = sp.lambdify([WT_s, WC_s], sp.sympify(expr, locals={'WT': WT_s, 'WC': WC_s}), 'numpy')
    with np.errstate(all='ignore'):
        pc = [np.mean(f(df.WT.values, np.full(len(df), v))) for v in GWC]
        pt = [np.mean(f(np.full(len(df), v), df.WC.values)) for v in GWT]
    return np.real(np.array(pc, dtype=complex)), np.real(np.array(pt, dtype=complex))


curvas = {}
for (met, fuente), col in [(('BMS', 'C-003 (CV)'), AZUL), (('PySR', 'C-002 (CV)'), NARANJA)]:
    g = EQ[(EQ.metodo == met) & (EQ.fuente == fuente) & (EQ.variante == 'principal')]
    curvas[met] = [dep_parcial(e) for e in g.expr]
    assert len(curvas[met]) == 25, (met, len(curvas[met]))
marg_c = df.groupby('nivel').agg(x=('WC', 'mean'), y=('T', 'mean'))
marg_t = df.groupby('cyc').agg(x=('WT', 'first'), y=('T', 'mean'))
fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
for k, (ax, var, grid, marg, xl) in enumerate([(axs[0], 'WC', GWC, marg_c, 'Welding current WC (kA)'),
                                              (axs[1], 'WT', GWT, marg_t, 'Welding time WT (s)')]):
    for met, col in [('PySR', NARANJA), ('BMS', AZUL)]:
        for c in curvas[met]:
            ax.plot(grid, c[k], color=col, lw=0.8, alpha=0.35, zorder=2)
    for met, g in PDP[PDP.variable == var].groupby('metodo'):
        ax.plot(g.x, g.pdp, color=TINTA2, lw=1.1, alpha=0.9, zorder=3)
    ax.scatter(marg.x, marg.y, s=16, color=TINTA, zorder=4, lw=0.6, edgecolor='white')
    ax.set_xlabel(xl); ax.set_title(['a  Effect of current', 'b  Effect of weld time'][k], loc='left')
axs[0].set_ylabel('Partial dependence of TSLBC (kN)')
fig.legend(handles=[Line2D([], [], color=AZUL, lw=1.2, label='BMS MDL, 25 partitions'),
                    Line2D([], [], color=NARANJA, lw=1.2, label='PySR, 25 partitions'),
                    Line2D([], [], color=TINTA2, lw=1.2, label='GAM, EBM, RF, GBM, TabPFN (all data)'),
                    Line2D([], [], color=TINTA, marker='o', lw=0, ms=4, label='Data, marginal means')],
           loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=4)
fig.subplots_adjust(bottom=0.24)
guardar(fig, 'Fig7_efectos_parciales')

# ---------------------------------------------------------------- FigS1: estudio sintético
filas = []
for carp in ['E-005_recuperacion_sintetica', 'E-006_recuperacion_sintetica_D_EF']:
    for f in glob.glob(str(R/carp/'T[123]_*_s0.*_seed[0-9].json')):
        d = json.loads(Path(f).read_text(encoding='utf-8')); nom = Path(f).stem.split('_')
        filas.append(dict(verdad=nom[0], opcion=nom[2], sigma=d['sigma'], delta=d.get('delta_dl'), rmse=d.get('rmse_funcional')))
S = pd.DataFrame(filas)
S.to_csv(OUT/'sintetico_por_cadena.csv', index=False)
filas_y = [('T3', 'A'), ('T3', 'B'), ('T3', 'D'), ('T1', 'B'), ('T1', 'D'), ('T2', 'B'), ('T2', 'D')]
npar = {'T3': 2, 'T1': 4, 'T2': 5}
etq = [f'{v} ({npar[v]} par.), config. {o}' for v, o in filas_y]
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for i, (v, o) in enumerate(filas_y[::-1]):
    g = S[(S.verdad == v) & (S.opcion == o)]
    for s, col, mk in [(0.10, AZUL, 'o'), (0.28, NARANJA, '^')]:
        h = g[np.isclose(g.sigma, s)]
        jit = np.linspace(-0.15, 0.15, max(len(h), 1))[:len(h)]
        axs[0].scatter(h.delta, i + jit + (-0.06 if s < 0.2 else 0.06), s=18, color=col, marker=mk, lw=0.5, edgecolor='white', zorder=3)
        axs[1].scatter(h.rmse, i + jit + (-0.06 if s < 0.2 else 0.06), s=18, color=col, marker=mk, lw=0.5, edgecolor='white', zorder=3)
axs[0].axvline(0, color=TINTA2, lw=0.8); axs[0].set_yticks(range(len(filas_y))); axs[0].set_yticklabels(etq[::-1])
axs[0].grid(axis='y', visible=False); axs[1].grid(axis='y', visible=False)
axs[0].set_xlabel('ΔL = L(best expression) − L(truth)'); axs[0].set_title('a  Plausibility of the truth', loc='left')
axs[0].text(-1, -0.9, '← best expression more plausible', ha='right', va='bottom', fontsize=6.5, color=TINTA2)
for s in (0.10, 0.28):
    axs[1].axvline(s, color=GRIS_CLARO, lw=0.9, ls=(0, (4, 2)))
    axs[1].text(s, len(filas_y) - 0.45, f'σ = {s:.2f}', ha='center', va='bottom', fontsize=6.5, color=TINTA2)
axs[1].set_xlabel('Functional RMSE of the best expression (kN)'); axs[1].set_title('b  Distance to the truth', loc='left', pad=12); axs[0].title.set_y(axs[1].title.get_position()[1])
axs[0].set_title('a  Plausibility of the truth', loc='left', pad=12); axs[1].set_xlim(0, 0.30)
axs[0].set_ylim(-0.9, len(filas_y) - 0.3)
fig.legend(handles=[Line2D([], [], color=AZUL, marker='o', lw=0, ms=5, label='noise σ = 0.10 kN'),
                    Line2D([], [], color=NARANJA, marker='^', lw=0, ms=5, label='noise σ = 0.28 kN')],
           loc='lower center', bbox_to_anchor=(0.55, -0.04), ncol=2)
fig.subplots_adjust(bottom=0.22, wspace=0.08)
guardar(fig, 'FigS1_sintetico')

# ---------------------------------------------------------------- FigS4: formas con y sin desplazamiento en WC
CV = pd.read_csv(R/'2026-10-19_E-016'/'cv_por_particion.csv')
FORMAS = [('P6', '1 − exp(−k(WC − c))', 'si'), ('P1', 'ln(WC/c)', 'si'), ('P8', 'Michaelis–Menten in (WC − c)', 'si'),
          ('P4', '(WC − c)$^{d}$', 'si'), ('P3', '√(WC − c)', 'si'), ('F2', 'ln(WC + WT), compact form', 'implicito'),
          ('F1', 'ln WC, PySR form', 'implicito'), ('P5', '1 − exp(−k·WC)', 'no'), ('P7', 'Michaelis–Menten in WC', 'no'),
          ('P2', 'WC$^{d}$', 'no'), ('P9', 'Linear in WC', 'no'), ('C1', '√WC', 'no')]
COL = {'si': AZUL, 'implicito': NARANJA, 'no': GRIS}
res = []
for cod, et, tipo in FORMAS:
    g = CV[CV.forma.str.startswith(cod + ' ')]
    assert g.particion.nunique() == 25, cod
    res.append(dict(codigo=cod, etiqueta=et, tipo=tipo, mse=g.mse.mean(), mse_de=g.mse.std(), c_mediana=g.umbral_c.median()))
RS = pd.DataFrame(res); RS.to_csv(OUT/'formas_umbral_resumen.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(7, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
for i, r in RS[::-1].reset_index(drop=True).iterrows():
    ax.errorbar(r.mse, i, xerr=r.mse_de, fmt='o', ms=4.5, color=COL[r.tipo], elinewidth=1.3, capsize=0, mec='white', mew=0.5)
ax.set_yticks(range(len(RS))); ax.set_yticklabels(RS.etiqueta[::-1]); ax.grid(axis='y', visible=False)
ax.set_xlabel('Out-of-sample MSE (kN²), mean ± SD\nover 25 grouped partitions'); ax.set_title('a  Current dependence × (a − b/WT)', loc='left')
ax.legend(handles=[Line2D([], [], color=COL[t], marker='o', lw=1.3, label=l) for t, l in
                   [('si', 'explicit offset c'), ('implicito', 'implicit offset'), ('no', 'no offset')]], loc='upper right')
ax = axs[1]
con = [f for f in FORMAS if f[2] == 'si']
ax.axvspan(df.WC.min(), 2.0, color=REJILLA, lw=0, zorder=0)
ax.text(df.WC.min() + 0.03, len(con) - 0.55, 'tested\ncurrents', fontsize=6.5, color=TINTA2, va='top')
for i, (cod, et, _) in enumerate(con[::-1]):
    c = CV[CV.forma.str.startswith(cod + ' ')].umbral_c.values
    ax.scatter(c, i + np.random.default_rng(i).uniform(-0.15, 0.15, len(c)), s=7, color=AZUL, alpha=0.45, lw=0, zorder=2)
    ax.scatter(np.median(c), i, s=36, marker='|', color=TINTA, lw=1.6, zorder=3)
ax.set_yticks(range(len(con))); ax.set_yticklabels([e for _, e, _ in con[::-1]]); ax.grid(axis='y', visible=False)
ax.set_xlim(0, 2.0); ax.set_xlabel('Offset c (kA) per partition;\nbar = median'); ax.set_title('b  Implied zero-strength current', loc='left')
fig.subplots_adjust(wspace=0.62)
guardar(fig, 'FigS4_umbral')

import matplotlib as _m, scipy as _s
json.dump(dict(python=platform.python_version(), plataforma=platform.platform(), matplotlib=_m.__version__, numpy=np.__version__,
               pandas=pd.__version__, sympy=sp.__version__, scipy=_s.__version__), open(OUT/'entorno.json', 'w'), indent=1)
print(RS.round(4).to_string(index=False)); print(S.groupby(['verdad', 'opcion']).agg(n=('delta', 'size'), dmin=('delta', 'min'), dmax=('delta', 'max'), rmax=('rmse', 'max')))
