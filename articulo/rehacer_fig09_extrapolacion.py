"""Fig. 9 (extrapolación) rehecha para legibilidad. No calcula nada nuevo: lee
resultados/2026-10-19_E-013/extrapolacion_detalle.csv (los mismos números que
C-012_figuras.py, Fig6_extrapolacion).
Cambios frente a la versión anterior:
  - eje x logarítmico COMPARTIDO por los dos paneles (antes escalas lineales distintas, 0-1.9 y 0-0.4);
  - referencia por método: MSE de CV en las mismas uniones (ambos extremos juntos), y varianza de ruido estimada;
  - leyenda única y genérica (extremo inferior / superior); los niveles concretos van en el título de cada panel;
  - bandas alternas por fila y fuente algo mayor.
Uso: python3 articulo/rehacer_fig09_extrapolacion.py
Salida: articulo/borrador/figuras_revisadas/Fig09_extrapolation_v2.{pdf,png}"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT/'resultados'/'2026-10-19_E-013'/'extrapolacion_detalle.csv'
OUT = ROOT/'articulo'/'borrador'/'figuras_revisadas'; OUT.mkdir(parents=True, exist_ok=True)

AZUL, NARANJA = '#2a78d6', '#eb6834'
GRIS, GRIS_CLARO, TINTA, TINTA2, REJILLA, BANDA = '#8a8984', '#c9c8c2', '#0b0b0b', '#52514e', '#e6e5e0', '#f4f3ef'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8.5,
                     'legend.fontsize': 8, 'axes.edgecolor': GRIS_CLARO, 'axes.labelcolor': TINTA2, 'xtick.color': TINTA2,
                     'ytick.color': TINTA2, 'text.color': TINTA, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 300})
NOMBRE = {'TabPFN_v2': 'TabPFN v2', 'EBM': 'EBM', 'GBM': 'GBM', 'BMS_promedio': 'BMS posterior-sample average', 'RF': 'Random forest',
          'GAM': 'GAM', 'cuadratico_EN': 'Quadratic (elastic net)', 'BMS_MDL': 'BMS MDL', 'E001_exploratorio': 'Saturation (E-001)',
          'PySR': 'PySR'}
SIGMA2 = 0.066  # varianza de ruido estimada sin modelo (C-006)

X = pd.read_csv(CSV)
fil = lambda dv: X[(X.division == dv) & (X.modelo != 'media')].set_index('modelo')
paneles = [('WC_extremos', ('a  Weld current extremes held out', 'lower: 1.4–1.7 kA    upper: 6.4–6.6 kA'), 'mse_nivel_WC_1', 'mse_nivel_WC_11'),
           ('WT_extremos', ('b  Weld time extremes held out', 'lower: 2 cycles (0.04 s)    upper: 12 cycles (0.24 s)'), 'mse_ciclos_2', 'mse_ciclos_12')]
orden = fil('WC_extremos').sort_values('mse_extrap').index.tolist()   # mismo orden que la versión anterior
filas = orden[::-1]

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True, sharex=True)
for ax, (dv, tit, c_inf, c_sup) in zip(axs, paneles):
    t = fil(dv)
    for i, m in enumerate(filas):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BANDA, zorder=0, lw=0)
        lo, hi = t.loc[m, c_inf], t.loc[m, c_sup]
        ax.plot([min(lo, hi), max(lo, hi)], [i, i], color=GRIS_CLARO, lw=1.2, zorder=1)
        ax.plot(t.loc[m, 'mse_cv_mismos_puntos'], i, marker='|', ms=9, mew=1.6, color=TINTA, zorder=3)
        ax.scatter(lo, i, s=34, color=AZUL, zorder=4, edgecolor='white', lw=0.6)
        ax.scatter(hi, i, s=34, color=NARANJA, zorder=4, edgecolor='white', lw=0.6)
    ax.axvline(SIGMA2, color=GRIS, ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax.text(0, 1.10, tit[0], transform=ax.transAxes, fontsize=9, fontweight='bold', color=TINTA, va='bottom')
    ax.text(0, 1.03, tit[1], transform=ax.transAxes, fontsize=8, color=TINTA2, va='bottom')
    ax.set_xscale('log'); ax.set_xlim(0.05, 2.6)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis='x', color=REJILLA, lw=0.6, zorder=0); ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)
axs[0].set_ylim(-0.6, len(filas) - 0.4)
axs[0].set_yticks(range(len(filas))); axs[0].set_yticklabels([NOMBRE[m] for m in filas])
for lab in axs[0].get_yticklabels():
    if lab.get_text().startswith('BMS'):
        lab.set_fontweight('bold'); lab.set_color(TINTA)
axs[1].text(SIGMA2 * 1.06, len(filas) - 0.55, 'noise\nvariance', fontsize=7, color=GRIS, va='top')

leyenda = [Line2D([], [], color=AZUL, marker='o', lw=0, ms=6, label='Lower extreme held out'),
           Line2D([], [], color=NARANJA, marker='o', lw=0, ms=6, label='Upper extreme held out'),
           Line2D([], [], color=TINTA, marker='|', lw=0, ms=9, mew=1.6, label='Cross-validation error at the same joints (both extremes pooled)'),
           Line2D([], [], color=GRIS, ls=(0, (3, 2)), lw=0.9, label='Estimated noise variance (0.066 kN²)')]
fig.supxlabel('MSE on the held-out joints (kN², log scale; same scale in both panels)', fontsize=8.5, color=TINTA2, x=0.6, y=0.115)
fig.legend(handles=leyenda, loc='lower center', ncol=2, bbox_to_anchor=(0.6, -0.04), handletextpad=0.4, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=4.0)
for ext in ('pdf', 'png'):
    fig.savefig(OUT/f'Fig09_extrapolation_v2.{ext}', bbox_inches='tight')
print('ok', OUT)
