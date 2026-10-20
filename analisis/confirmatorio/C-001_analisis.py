"""Análisis de C-001 (protocolo §2A, §6 H2/H3b/H4b, §7, §10). Lee resultados/C-001_analisis_A/*.json.
Produce: resumen_cadenas.csv, estructuras_posterior.csv, C-001_informe.json y C-001_informe.md.
  - Cadenas: DL mínima, paso del MDL, estructura canónica, aceptación de swaps, parches, truncadas/relanzadas.
  - MDL global por configuración (mínima DL entre cadenas completas; relanzadas _r incluidas).
  - H4b: Δ = DL_D − DL_B (tres desenlaces, umbral 20).
  - Posterior (D): las 10 estructuras canónicas más frecuentes en las muestras MCMC (todas las cadenas).
  - H2 (descriptiva): nº de parámetros del MDL y frecuencia de k ≤ 4 en las muestras.
  - H3b en el MDL del análisis A (descriptivo aquí; el contraste es en la CV): γ̂, R²_ui, p_T.
  - §10: bootstrap de residuos (1000 réplicas) de los parámetros del MDL de D; comprobaciones físicas.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-001_analisis.py [carpeta_resultados]
"""
import json, re, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, sympy as sp
from scipy.optimize import curve_fit
from bms_comun import cargar_datos, ROOT

D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'resultados'/'C-001_analisis_A'


def a_md(t):   # tabla markdown sin dependencias (tabulate no está en el lock)
    cols = list(t.columns)
    filas = ['| ' + ' | '.join(cols) + ' |', '|' + '---|'*len(cols)]
    filas += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in t.itertuples(index=False)]
    return '\n'.join(filas)
cad = {}
for f in sorted(D.glob('*_cadena*.json')):
    if f.name.endswith('.progreso.json'):
        continue
    cad[f.stem] = json.loads(f.read_text(encoding='utf-8'))
trunc = [p.name for p in D.glob('*_TIEMPO_AGOTADO.txt')]; err = [p.name for p in D.glob('*_ERROR.txt')]
filas = []
for n, d in cad.items():
    k = len(set(re.findall(r'_a\d_', d['modelo_mdl'])))
    filas.append(dict(cadena=n, config=n.split('_')[0], dl_mdl=d['dl_mdl'], paso_mdl=d['paso_mdl'], n_param=k,
                      canon=d['canon_mdl'], modelo=d['modelo_mdl'], aceptacion_swaps=round(d['aceptacion_swaps'], 3),
                      timeouts=d['parches'].get('timeout_sympify', 0), complejas=d['parches'].get('expresion_compleja', 0),
                      minutos=round(d['tiempo_s']/60, 1)))
tc = pd.DataFrame(filas).sort_values(['config', 'dl_mdl'])
tc.to_csv(D/'resumen_cadenas.csv', index=False)
inf = dict(cadenas_completas=len(cad), truncadas=trunc, errores=err)
mejor = {}
for conf in ('D', 'B'):
    t = tc[tc.config == conf]
    if len(t) == 0:
        continue
    r = t.iloc[0]; mejor[conf] = cad[r.cadena]
    inf[f'MDL_{conf}'] = dict(cadena=r.cadena, dl=r.dl_mdl, modelo=r.modelo, canon=r.canon, n_param=int(r.n_param),
                             parametros=cad[r.cadena]['parametros_mdl'],
                             cadenas_misma_estructura=int((t.canon == r.canon).sum()), n_cadenas=len(t),
                             dl_rango=[float(t.dl_mdl.min()), float(t.dl_mdl.max())])
# H4b
if 'D' in mejor and 'B' in mejor:
    delta = mejor['D']['dl_mdl'] - mejor['B']['dl_mdl']
    inf['H4b'] = dict(delta=delta, desenlace=('modelar EF compensa (efecto)' if delta > 20 else
                                              'no compensa modelar EF' if delta < -20 else 'no concluyente'))
# posterior (D)
cnt = collections.Counter(); dls = collections.defaultdict(list)
for n, d in cad.items():
    if n.startswith('D_'):
        for m in d['muestras']:
            cnt[m['canon']] += 1; dls[m['canon']].append(m['E'])
tot = sum(cnt.values())
post = pd.DataFrame([dict(canon=c, frecuencia=v/tot, n=v, dl_min=min(dls[c]),
                          n_param=len(set(re.findall(r'c\d+', c)))) for c, v in cnt.most_common()])
post.to_csv(D/'estructuras_posterior.csv', index=False)
inf['posterior_top10'] = post.head(10).to_dict('records'); inf['n_muestras_D'] = tot
inf['H2_descriptiva'] = dict(n_param_MDL_D=inf.get('MDL_D', {}).get('n_param'),
                             frac_muestras_k_le_4=float((post.n_param <= 4).dot(post.frecuencia)) if tot else None)
# Funciones del MDL de D
df = cargar_datos(); y = df['T'].values
WT_s, WC_s = sp.symbols('WT WC')
if 'D' in mejor:
    md = mejor['D']; pv = md['parametros_mdl']['d0']
    ex = sp.sympify(md['modelo_mdl'], locals={'WT': WT_s, 'WC': WC_s})
    usados = sorted({str(s) for s in ex.free_symbols} - {'WT', 'WC'})
    psyms = [sp.Symbol(u) for u in usados]
    f_par = sp.lambdify([WT_s, WC_s] + psyms, ex, 'numpy')
    theta = np.array([pv[u] for u in usados])
    f = lambda t, c, th=theta: np.asarray(f_par(t, c, *th), dtype=float) * np.ones_like(np.asarray(t, dtype=float))
    pred = f(df.WT.values, df.WC.values); res = y - pred
    inf['MDL_D_ajuste_en_muestra'] = dict(mse=float(np.mean(res**2)), r2=float(1 - np.mean(res**2)/np.var(y)),
                                         nota='EN MUESTRA: no es error de predicción (ver C-003)')
    # H3b descriptivo en A
    WTn = np.arange(2, 13)*0.02; WCn = np.linspace(1.5, 6.5, 11); GT, GC = np.meshgrid(WTn, WCn, indexing='ij')
    h = 1e-4
    dT = np.ravel((f(GT*np.exp(h), GC) - f(GT*np.exp(-h), GC))/(2*h))
    dC = np.ravel((f(GT, GC*np.exp(h)) - f(GT, GC*np.exp(-h)))/(2*h))
    g = float(np.sum(dT*dC)/np.sum(dC**2)); sT = np.sum(dT**2)
    inf['H3b_en_A'] = dict(gamma=g, R2_ui=float(1-np.sum((dT-g*dC)**2)/sT) if sT > 0 else None,
                           p_T=float(sT/(sT+np.sum(dC**2))))
    # Bootstrap de residuos (estructura fija)
    rng = np.random.default_rng(20261016); X = np.vstack([df.WT.values, df.WC.values])
    fit = lambda Xa, *th: np.asarray(f_par(Xa[0], Xa[1], *th), dtype=float) * np.ones(Xa.shape[1])
    boot = []
    for b in range(1000):
        yb = pred + rng.choice(res, len(res), replace=True)
        try:
            th, _ = curve_fit(fit, X, yb, p0=theta, maxfev=20000); boot.append(th)
        except Exception:
            pass
    boot = np.array(boot)
    inf['bootstrap_parametros'] = dict(n_ok=len(boot), parametros={u: dict(est=float(theta[i]),
        ic95=[float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]) for i, u in enumerate(usados)}
        if len(boot) else {})
    # Comprobaciones físicas en una rejilla densa del dominio y fuera de él
    tt, cc = np.meshgrid(np.linspace(0.04, 0.24, 41), np.linspace(1.44, 6.56, 41), indexing='ij')
    with np.errstate(all='ignore'):
        F = f(tt, cc); dFt = np.gradient(F, axis=0); dFc = np.gradient(F, axis=1)
        inf['comprobaciones_fisicas'] = dict(
            minimo_en_dominio=float(np.nanmin(F)), maximo_en_dominio=float(np.nanmax(F)),
            no_finitos_en_dominio=int(np.sum(~np.isfinite(F))),
            frac_creciente_en_WT=float(np.mean(dFt >= -1e-9)), frac_creciente_en_WC=float(np.mean(dFc >= -1e-9)),
            f_WT_0p001=float(np.nanmax(np.abs(f(np.full(5, 1e-3), np.linspace(1.5, 6.5, 5))))),
            f_WC_0p01=float(np.nanmax(np.abs(f(np.linspace(0.04, 0.24, 5), np.full(5, 1e-2))))),
            f_fuera_WT_0p4=f(np.full(3, 0.4), np.array([2.0, 4.0, 6.5])).tolist(),
            f_fuera_WC_8=f(np.array([0.06, 0.12, 0.24]), np.full(3, 8.0)).tolist())
(D/'C-001_informe.json').write_text(json.dumps(inf, indent=1, default=float), encoding='utf-8')
md = ['# C-001 — Informe automático', '', f'Cadenas completas: {len(cad)}; truncadas: {trunc}; errores: {err}', '',
      a_md(tc[['cadena', 'dl_mdl', 'paso_mdl', 'n_param', 'aceptacion_swaps', 'timeouts', 'minutos', 'modelo']]), '',
      '## Posterior (D): 10 estructuras más frecuentes', '', a_md(post.head(10)), '',
      '```json', json.dumps({k: v for k, v in inf.items() if k not in ('posterior_top10',)}, indent=1, default=float), '```']
(D/'C-001_informe.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md[:6])); print(json.dumps({k: inf.get(k) for k in ('MDL_D', 'MDL_B', 'H4b', 'H3b_en_A', 'H2_descriptiva')}, indent=1, default=float))
