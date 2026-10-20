"""E-021 (exploratorio, descriptivo; 2026-10-19). Dispersión entre particiones para las tablas de la v2 del borrador
y recuento de motivos temporales con dos definiciones. No ajusta ni reevalúa modelos: solo resume salidas guardadas.
  - Tabla 4 (principal) y tabla de métricas secundarias del SM: media y DE de las 25 particiones agrupadas de
    C-005/metricas_por_particion.csv (comprueba que las medias coinciden con C-005/resumen_modelos.csv).
  - Tabla de sensibilidad del SM: media y DE por variante de C-009/mse_por_particion.csv (BMS en aleatoria,
    wc_nominal y np4: solo la primera repetición, 5 particiones).
  - Tabla de formas con y sin umbral: MSE y DE de E-020/formas_umbral_resumen.csv.
  - Motivos en los 25 MDL por partición (C-005/bms_por_particion.csv): 1/WT puro y WT en un denominador con el
    detector de E-012 (incluye 1/(WT ± …)).
La DE describe la variación entre particiones solapadas, no el error estándar de la media.
Uso: python3 analisis/exploratorio/E-021_dispersion_tablas.py"""
import json, platform, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; R = ROOT/'resultados'
OUT = R/'2026-10-19_E-021'; OUT.mkdir(parents=True, exist_ok=True)
ORDEN = ['TabPFN_v2', 'EBM', 'GBM', 'BMS_promedio', 'RF', 'GAM', 'cuadratico_EN', 'BMS_MDL', 'E001_exploratorio', 'PySR', 'media']
CORTO = {'TabPFN_v2': 'TabPFN v2', 'EBM': 'EBM', 'GBM': 'GBM', 'BMS_promedio': 'BMS average', 'RF': 'RF', 'GAM': 'GAM',
         'cuadratico_EN': 'Quadratic EN', 'BMS_MDL': 'BMS-MDL', 'E001_exploratorio': 'Saturation (expl.)', 'PySR': 'PySR',
         'media': 'Training mean'}
f3 = lambda m, s: f'{m:.3f} ({s:.3f})'.replace('-', '−')
def tabla_md(cab, filas):
    return '\n'.join(['| ' + ' | '.join(cab) + ' |', '| ' + ' | '.join(['---'] * len(cab)) + ' |'] + ['| ' + ' | '.join(f) + ' |' for f in filas])

# --- Tabla 4 y métricas secundarias (C-005)
M = pd.read_csv(R/'C-005_analisis_cv'/'metricas_por_particion.csv'); RES = pd.read_csv(R/'C-005_analisis_cv'/'resumen_modelos.csv').set_index('modelo')
met = ['mse', 'rmse', 'mae', 'r2', 'error_clas', 'exactitud_equilibrada', 'f1_aceptable', 'f1_rechazo']
g = M.groupby('modelo')[met]; media, de, n = g.mean(), g.std(ddof=1), g.size()
assert set(n) == {25}, n
for c in met:
    assert np.allclose(media.loc[ORDEN, c], RES.loc[ORDEN, c]), c   # mismas medias que la tabla publicada
assert np.allclose(de.loc[ORDEN, 'mse'], RES.loc[ORDEN, 'mse_de'])
pd.concat({'media': media, 'de': de}, axis=1).loc[ORDEN].to_csv(OUT/'metricas_media_de.csv')
t4 = tabla_md(['Model', 'MSE (kN²)', 'RMSE (kN)', 'Classification error', 'Balanced accuracy'],
              [[CORTO[m]] + [f3(media.loc[m, c], de.loc[m, c]) for c in ['mse', 'rmse', 'error_clas', 'exactitud_equilibrada']] for m in ORDEN])
ts = tabla_md(['Model', 'MAE (kN)', 'R²', 'F1, acceptable', 'F1, rejected'],
              [[CORTO[m]] + [f3(media.loc[m, c], de.loc[m, c]) for c in ['mae', 'r2', 'f1_aceptable', 'f1_rechazo']] for m in ORDEN if m != 'media'])

# --- Sensibilidad (C-009)
S = pd.read_csv(R/'C-009_analisis_sensibilidad'/'mse_por_particion.csv')
VAR = [('principal', 'Main'), ('aleatoria', 'Random partitions'), ('wc_nominal', 'Nominal current'), ('sin_cero', 'No zero-strength joint'), ('sin_ef', 'No EF input')]
gs = S.groupby(['variante', 'modelo']).mse.agg(['mean', 'std', 'size'])
gs.to_csv(OUT/'sensibilidad_media_de.csv')
filas = []
for m in ORDEN[:-1]:
    f = [CORTO[m]]
    for v, _ in VAR:
        r = gs.loc[(v, m)]; f.append(f3(r['mean'], r['std']) + ('†' if r['size'] == 5 else ''))
    filas.append(f)
tsens = tabla_md(['Model'] + [e for _, e in VAR], filas)
np4 = {m: dict(mse=gs.loc[('np4', m), 'mean'], de=gs.loc[('np4', m), 'std'], n=int(gs.loc[('np4', m), 'size']),
               mse_recortado=S[(S.variante == 'np4') & (S.modelo == m)].mse_rec.mean(),
               de_recortado=S[(S.variante == 'np4') & (S.modelo == m)].mse_rec.std(ddof=1)) for m in ['BMS_MDL', 'BMS_promedio']}

# --- Formas con y sin umbral (E-020)
F = pd.read_csv(R/'2026-10-19_E-020'/'formas_umbral_resumen.csv')

# --- Motivos temporales (C-005), definiciones de E-012
B = pd.read_csv(R/'C-005_analisis_cv'/'bms_por_particion.csv')
puro = B.canon.str.contains(r'/WT(?![\w*])', regex=True)
den = B.canon.apply(lambda s: bool(re.search(r'/WT(?![\w*])|/\(WT[+\-]', s)))
motivos = dict(n=len(B), inv_WT_puro=int(puro.sum()), WT_en_denominador_E012=int(den.sum()), log=int(B.canon.str.contains('log(', regex=False).sum()),
               cadenas_misma_estructura=B.cadenas_misma_estructura.value_counts().sort_index().to_dict())

(OUT/'tablas.md').write_text('# E-021: tablas con dispersión\n\n## Tabla 4\n\n' + t4 + '\n\n## Métricas secundarias (SM)\n\n' + ts +
                             '\n\n## Sensibilidad (SM; † = 5 particiones, primera repetición)\n\n' + tsens + '\n', encoding='utf-8')
json.dump(dict(np4=np4, motivos=motivos, formas=F[['etiqueta', 'mse', 'mse_de']].to_dict('records'),
               python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
          open(OUT/'resumen.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print((OUT/'tablas.md').read_text(encoding='utf-8')); print(json.dumps(dict(np4=np4, motivos=motivos), indent=1, default=float))
