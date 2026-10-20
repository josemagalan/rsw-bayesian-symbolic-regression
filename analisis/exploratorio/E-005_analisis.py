"""Análisis de E-005 (recuperación sintética). Lee resultados/E-005_recuperacion_sintetica/*.json
y escribe en la misma carpeta: resumen.csv, resumen.md y trazas.png. Se puede ejecutar con resultados parciales.
Uso: python analisis/exploratorio/E-005_analisis.py [carpeta_de_resultados]
     (por defecto E-005_recuperacion_sintetica; para E-006: E-006_recuperacion_sintetica_D_EF)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'resultados' / (sys.argv[1] if len(sys.argv) > 1 else 'E-005_recuperacion_sintetica')
rows, traces = [], {}
for f in sorted(p for p in D.glob('*.json') if not p.name.endswith('.progreso.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    tr = np.array(d['traza_cada10'], float); tr[~np.isfinite(tr)] = np.nan
    n = len(tr)
    mejora_ultimo_cuarto = np.nanmin(tr[:int(0.75*n)]) - np.nanmin(tr)
    c = d['cfg']
    if d['delta_dl'] is None:
        diag = 'verdad no representable'
    elif d['delta_dl'] > 1:
        diag = 'fallo de muestreo' + (' (aún bajando)' if mejora_ultimo_cuarto > 0 else ' (atascada)')
    else:
        diag = 'verdad no preferible / encontrada'
    rows.append(dict(verdad=c['truth'], opcion=c['option'], sigma=c['sigma'], semilla=c['seed'],
                     dl_mdl=round(d['dl_mdl'], 2),
                     dl_verdad=None if d['dl_verdad'] is None else round(d['dl_verdad'], 2),
                     delta_dl=None if d['delta_dl'] is None else round(d['delta_dl'], 2),
                     rmse_funcional=round(d['rmse_funcional'], 3), paso_mdl=d['paso_mdl'],
                     mejora_ultimo_cuarto=round(float(mejora_ultimo_cuarto), 2),
                     parche_complejas=d['parches']['expresion_compleja'],
                     pasos_por_s=round(d['pasos_por_s'], 2), minutos=round(d['tiempo_s']/60, 1),
                     diagnostico=diag, modelo_mdl=d['modelo_mdl']))
    traces[d['nombre']] = (tr, d['dl_verdad'])
df = pd.DataFrame(rows).sort_values(['verdad', 'opcion', 'sigma', 'semilla'])
df.to_csv(D / 'resumen.csv', index=False)
cols = ['verdad', 'opcion', 'sigma', 'semilla', 'dl_mdl', 'dl_verdad', 'delta_dl', 'rmse_funcional',
        'paso_mdl', 'mejora_ultimo_cuarto', 'diagnostico']
md = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
md += ['| ' + ' | '.join('' if pd.isna(r[c]) else str(r[c]) for c in cols) + ' |' for _, r in df.iterrows()]
(D / 'resumen.md').write_text(f'# E-005 — resumen ({len(df)} configuraciones)\n\n' + '\n'.join(md) + '\n', encoding='utf-8')

# Trazas: un panel por configuración (una sola serie, sin leyenda); línea discontinua = DL de la verdad
k = len(traces); ncol = 4; nrow = int(np.ceil(k / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 2.6*nrow), squeeze=False)
for ax, (name, (tr, dlv)) in zip(axes.flat, sorted(traces.items())):
    steps = np.arange(len(tr)) * 10
    ax.plot(steps, np.fmin.accumulate(np.nan_to_num(tr, nan=np.inf)), color='#2a78d6', lw=2)
    if dlv is not None:
        ax.axhline(dlv, color='#52514e', lw=1, ls='--')
    ax.set_title(name, fontsize=8, color='#0b0b0b')
    ax.tick_params(labelsize=7, colors='#52514e')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.6)
for ax in list(axes.flat)[k:]:
    ax.axis('off')
fig.suptitle(f'{D.name}: mínima longitud de descripción alcanzada vs. paso MCMC (discontinua = DL de la verdad)', fontsize=10)
fig.tight_layout()
fig.savefig(D / 'trazas.png', dpi=130)
print(df[cols].to_string(index=False))
