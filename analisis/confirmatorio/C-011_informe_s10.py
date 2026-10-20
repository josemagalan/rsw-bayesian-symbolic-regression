"""C-011 (CONFIRMATORIO, protocolo §10): informe adicional obligatorio que faltaba tras C-001.
(El bootstrap de residuos de los parámetros del MDL de A y sus comprobaciones físicas ya están en C-001.)
  1. Diagnóstico de residuos del MDL del análisis A (D, cadena de menor DL; en muestra):
     - heterocedasticidad: test de Breusch-Pagan en la versión de Koenker (n·R² de e² sobre [1, ŷ] y sobre [1, WT, WC, EF]),
       Spearman de |e| con ŷ, varianza de e por terciles de ŷ; normalidad (Shapiro, descriptiva);
     - puntos influyentes: apalancamiento no lineal h = diag(J(JᵀJ)⁺Jᵀ), con J el jacobiano numérico en θ̂; distancia de
       Cook D = e²/(p·s²)·h/(1−h)²; se marcan D > 4/n; borrado de casos para los 3 más influyentes (reajuste y cambio de las
       predicciones en las 242 condiciones).
     Lo mismo, como referencia (exploratoria, E-010), para la forma compacta (a − b/WT)·ln(WC + WT) ajustada a todos los datos.
  2. Residuos fuera de muestra (CV de C-005): por punto, error cuadrático medio en las 5 repeticiones para BMS-MDL,
     BMS-promedio, EBM, TabPFN, GAM y cuadrático; heterocedasticidad respecto a la respuesta; puntos más difíciles.
  3. Comprobaciones físicas (§10) de TODOS los MDL de D: los 8 de C-001 y los 25 de C-003 (más la forma compacta):
     f ≥ 0 y finita en el dominio; monotonía en WT y WC; límites WT → 0 (0.001 s) y WC → 0 (0.01 kA); saturación y
     acotación fuera del dominio (WT 0.5 y 1 s; WC 10 y 20 kA), como advertencia.
  4. Material suplementario: tabla de TODAS las cadenas (C-001, C-003, C-007).
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-011_informe_s10.py     (salida en resultados\\C-011_informe_s10)"""
import glob, json, re, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, sympy as sp
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import factorial
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA

warnings.filterwarnings('ignore')
R = ROOT/'resultados'; OUT = R/'C-011_informe_s10'; OUT.mkdir(parents=True, exist_ok=True)
df = cargar_datos(); y = df['T'].values; n = len(y)
WT_s, WC_s = sp.symbols('WT WC')
inf = dict(protocolo_sha=PROTOCOLO_SHA)


def a_md(t, dec=4):
    fmt = lambda v: (f'{v:.{dec}g}' if isinstance(v, (float, np.floating)) else
                     '[' + ', '.join(f'{x:.3g}' for x in v) + ']' if isinstance(v, list) else str(v))
    cols = list(t.columns)
    return '\n'.join(['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|'*len(cols)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in t.itertuples(index=False)])


def funcion(modelo):
    ex = sp.sympify(modelo, locals={'WT': WT_s, 'WC': WC_s})
    usados = sorted(str(s) for s in ex.free_symbols if str(s) not in ('WT', 'WC'))
    fl = sp.lambdify([WT_s, WC_s] + [sp.Symbol(u) for u in usados], ex, ['numpy', {'fac': factorial}])

    def f(t, c, *th):
        t, c = np.asarray(t, float), np.asarray(c, float)
        with np.errstate(all='ignore'):
            v = np.asarray(fl(t, c, *th))
            if np.iscomplexobj(v):
                v = np.where(np.abs(v.imag) < 1e-12, v.real, np.nan)
            return v.astype(float)*np.ones(np.broadcast(t, c).shape)
    return f, usados


def cadenas(patron):
    cs = {}
    for f in sorted(glob.glob(str(R/patron))):
        if f.endswith('.progreso.json'):
            continue
        p = Path(f); m = re.match(r'(.+)_cadena(\d+)(_r)?\.json$', p.name)
        k = (p.parent.name, m.group(1), int(m.group(2)))
        if k not in cs or m.group(3):
            cs[k] = (p.name, json.loads(p.read_text(encoding='utf-8')))
    return cs


# ---------------------------------------------------------------- 1. residuos e influencia (en muestra)
def jacobiano(f, X, th):
    J = np.empty((X.shape[1], len(th)))
    for j in range(len(th)):
        h = 1e-6*max(abs(th[j]), 1e-12)
        tp, tm = th.copy(), th.copy(); tp[j] += h; tm[j] -= h
        J[:, j] = (f(X[0], X[1], *tp) - f(X[0], X[1], *tm))/(2*h)
    return J


def koenker(e2, Z):
    Z = np.column_stack([np.ones(len(e2)), Z]); b, *_ = np.linalg.lstsq(Z, e2, rcond=None)
    r2 = 1 - np.sum((e2 - Z@b)**2)/np.sum((e2 - e2.mean())**2); lm = len(e2)*r2
    return dict(LM=float(lm), gl=Z.shape[1] - 1, p=float(stats.chi2.sf(lm, Z.shape[1] - 1)))


def diagnostico(nombre, f, th, etiqueta):
    X = np.vstack([df.WT.values, df.WC.values]); th = np.asarray(th, float)
    yhat = f(X[0], X[1], *th); e = y - yhat; p = len(th); s2 = np.sum(e**2)/(n - p)
    J = jacobiano(f, X, th); H = J @ np.linalg.pinv(J.T @ J, rcond=1e-12) @ J.T; h = np.clip(np.diag(H), 0, 0.999999)
    cook = e**2/(p*s2)*h/(1 - h)**2
    ter = pd.qcut(yhat, 3, labels=['bajo', 'medio', 'alto'])
    res = dict(modelo=etiqueta, n_parametros=p, mse_en_muestra=float(np.mean(e**2)), s2=float(s2),
               BP_Koenker_yhat=koenker(e**2, yhat), BP_Koenker_WT_WC_EF=koenker(e**2, df[['WT', 'WC', 'EF']].values),
               spearman_abs_e_yhat=dict(zip(['rho', 'p'], map(float, stats.spearmanr(np.abs(e), yhat)))),
               var_e_por_tercil_yhat={k: float(np.var(e[ter == k], ddof=1)) for k in ['bajo', 'medio', 'alto']},
               shapiro_p=float(stats.shapiro(e).pvalue), suma_h=float(h.sum()),
               n_cook_mayor_4_n=int(np.sum(cook > 4/n)))
    tabla = df[['id', 'cyc', 'WT', 'WC', 'EF', 'T']].assign(prediccion=yhat, residuo=e, h=h, cook=cook)
    tabla.to_csv(OUT/f'residuos_{nombre}.csv', index=False)
    top = tabla.sort_values('cook', ascending=False).head(10)
    # borrado de casos de los 3 más influyentes
    borr = []
    fit = lambda Xa, *t: f(Xa[0], Xa[1], *t)
    for i in top.index[:3]:
        m = np.arange(n) != i
        try:
            t2, _ = curve_fit(fit, X[:, m], y[m], p0=th, maxfev=20000)
            dpred = f(X[0], X[1], *t2) - yhat
            borr.append(dict(id=int(df.id[i]), rms_cambio_pred=float(np.sqrt(np.mean(dpred[m]**2))),
                             max_cambio_pred=float(np.max(np.abs(dpred))), cambio_pred_propio=float(dpred[i])))
        except Exception as ex:
            borr.append(dict(id=int(df.id[i]), error=repr(ex)[:80]))
    res['borrado_de_casos'] = borr
    return res, top


mdlA = min(((k, d) for k, (_, d) in cadenas('C-001_analisis_A/D_cadena*.json').items()), key=lambda kd: kd[1]['dl_mdl'])
fA, usA = funcion(mdlA[1]['modelo_mdl']); thA = [mdlA[1]['parametros_mdl']['d0'][u] for u in usA]
inf['MDL_A'] = dict(cadena=mdlA[0][2], dl=mdlA[1]['dl_mdl'], canon=mdlA[1]['canon_mdl'])
diag_A, top_A = diagnostico('MDL_A', fA, thA, f"MDL de A: {mdlA[1]['canon_mdl']}")
# forma compacta (E-010, exploratoria), ajustada a todos los datos
fC = lambda t, c, a, b: (a - b/np.asarray(t, float))*np.log(np.asarray(c, float) + np.asarray(t, float))
thC, _ = curve_fit(lambda X, a, b: fC(X[0], X[1], a, b), np.vstack([df.WT, df.WC]), y, p0=[3.9, 0.05])
diag_C, top_C = diagnostico('compacta', fC, thC, f'(a − b/WT)·ln(WC + WT), a = {thC[0]:.3f}, b = {thC[1]:.4f} (exploratoria)')
inf['residuos'] = [diag_A, diag_C]

# ---------------------------------------------------------------- 2. residuos fuera de muestra (CV)
P = pd.read_csv(R/'C-005_analisis_cv'/'predicciones_todas.csv')
MOD = ['BMS_MDL', 'BMS_promedio', 'EBM', 'TabPFN_v2', 'GAM', 'cuadratico_EN']
P = P[P.modelo.isin(MOD)].assign(e=lambda d: d.y - d.pred)
pp = P.groupby(['modelo', 'id']).agg(e_medio=('e', 'mean'), ecm=('e', lambda v: np.mean(v**2))).reset_index()
ancho = pp.pivot(index='id', columns='modelo', values='ecm')
pts = df.set_index('id')[['cyc', 'WT', 'WC', 'EF', 'T']].join(ancho)
pts['ecm_medio_metodos'] = ancho[MOD].mean(axis=1)
pts.reset_index().to_csv(OUT/'residuos_cv_por_punto.csv', index=False)
het = []
for m in MOD:
    g = pp[pp.modelo == m].set_index('id').join(df.set_index('id')['T'])
    rho, pv = stats.spearmanr(np.abs(g.e_medio), g['T'])
    clase = g['T'] >= 5.93
    het.append(dict(modelo=m, spearman_abs_e_y=rho, p=pv, mse_aceptables=g.ecm[clase].mean(), mse_no_aceptables=g.ecm[~clase].mean(),
                    frac_mse_10_peores=np.sort(g.ecm.values)[-10:].sum()/g.ecm.sum()))
het = pd.DataFrame(het); het.to_csv(OUT/'heterocedasticidad_cv.csv', index=False)
dificiles = pts.sort_values('ecm_medio_metodos', ascending=False).head(10).reset_index()

# ---------------------------------------------------------------- 3. comprobaciones físicas de todos los MDL de D
tt, cc = np.meshgrid(np.linspace(0.04, 0.24, 41), np.linspace(1.44, 6.56, 41), indexing='ij')
CWC = np.linspace(1.5, 6.5, 6); CWT = np.linspace(0.04, 0.24, 6)


def fisica(f, th):
    with np.errstate(all='ignore'):
        F = f(tt, cc, *th); dFt = np.diff(F, axis=0); dFc = np.diff(F, axis=1)
        ev = lambda t, c: f(np.asarray(t, float), np.asarray(c, float), *th)
        return dict(min_dominio=float(np.nanmin(F)), max_dominio=float(np.nanmax(F)), no_finitos=int(np.sum(~np.isfinite(F))),
                    f_no_negativa=bool(np.nanmin(F) >= 0 and np.all(np.isfinite(F))),
                    frac_creciente_WT=float(np.mean(dFt >= -1e-9)), frac_creciente_WC=float(np.mean(dFc >= -1e-9)),
                    max_abs_WT_0p001=float(np.nanmax(np.abs(ev(np.full(6, 1e-3), CWC)))),
                    max_abs_WC_0p01=float(np.nanmax(np.abs(ev(CWT, np.full(6, 1e-2))))),
                    rango_WT_0p001=[float(np.nanmin(ev(np.full(6, 1e-3), CWC))), float(np.nanmax(ev(np.full(6, 1e-3), CWC)))],
                    rango_WC_0p01=[float(np.nanmin(ev(CWT, np.full(6, 1e-2)))), float(np.nanmax(ev(CWT, np.full(6, 1e-2))))],
                    rango_WT_0p5=[float(np.nanmin(ev(np.full(6, .5), CWC))), float(np.nanmax(ev(np.full(6, .5), CWC)))],
                    rango_WT_1=[float(np.nanmin(ev(np.full(6, 1.), CWC))), float(np.nanmax(ev(np.full(6, 1.), CWC)))],
                    rango_WC_10=[float(np.nanmin(ev(CWT, np.full(6, 10.)))), float(np.nanmax(ev(CWT, np.full(6, 10.))))],
                    rango_WC_20=[float(np.nanmin(ev(CWT, np.full(6, 20.)))), float(np.nanmax(ev(CWT, np.full(6, 20.))))])


filas = []
for (carp, pref, c), (fn, d) in cadenas('C-001_analisis_A/D_cadena*.json').items():
    f, us = funcion(d['modelo_mdl'])
    filas.append(dict(fuente='C-001 (A)', modelo=f'{pref}_cadena{c}', dl=d['dl_mdl'], canon=d['canon_mdl'],
                      es_mdl_global=(c == mdlA[0][2]), **fisica(f, [d['parametros_mdl']['d0'][u] for u in us])))
c3 = cadenas('C-003_bms_cv/p??_cadena*.json'); parts = sorted({k[1] for k in c3})
for p in parts:
    k, (fn, d) = min(((k, v) for k, v in c3.items() if k[1] == p), key=lambda kv: kv[1][1]['dl_mdl'])
    f, us = funcion(d['modelo_mdl'])
    filas.append(dict(fuente='C-003 (CV)', modelo=f'{p} (cadena {k[2]})', dl=d['dl_mdl'], canon=d['canon_mdl'],
                      es_mdl_global=False, **fisica(f, [d['parametros_mdl']['d0'][u] for u in us])))
filas.append(dict(fuente='E-010 (exploratoria)', modelo='forma compacta', dl=np.nan, canon='(a − b/WT)·ln(WC + WT)',
                  es_mdl_global=False, **fisica(fC, thC)))
FIS = pd.DataFrame(filas); FIS.to_csv(OUT/'comprobaciones_fisicas.csv', index=False)
cv = FIS[FIS.fuente == 'C-003 (CV)']
lim = lambda v: v < 0.5          # "tiende a 0": |f| < 0.5 kN en el límite
inf['fisica_resumen_CV'] = dict(
    n=len(cv), f_no_negativa=int(cv.f_no_negativa.sum()), monotona_WT=int((cv.frac_creciente_WT == 1).sum()),
    monotona_WC=int((cv.frac_creciente_WC == 1).sum()), tiende_a_0_WT=int(lim(cv.max_abs_WT_0p001).sum()),
    tiende_a_0_WC=int(lim(cv.max_abs_WC_0p01).sum()),
    acotada_fuera=int(cv.apply(lambda r: max(abs(np.array(r.rango_WT_1 + r.rango_WC_20))) <= 2*y.max(), axis=1).sum()))

# ---------------------------------------------------------------- 4. todas las cadenas (material suplementario)
filas = []
for fuente, patron in [('C-001', 'C-001_analisis_A/*_cadena*.json'), ('C-003', 'C-003_bms_cv/p??_cadena*.json'),
                       ('C-007', 'C-007_bms_sensibilidad/*/*_cadena*.json')]:
    for (carp, pref, c), (fn, d) in cadenas(patron).items():
        cfg = d['cfg']
        filas.append(dict(fuente=fuente, carpeta=carp, grupo=pref, cadena=c, fichero=fn, semilla=cfg['seed'], prior=cfg['prior'],
                          configuracion='B (por EF)' if cfg['multi'] else 'D', variables='+'.join(cfg['variables']),
                          wc=cfg.get('wc', 'WC'), dl_mdl=d['dl_mdl'], paso_mdl=d['paso_mdl'],
                          n_parametros=len(set(re.findall(r'_a\d+_', d['modelo_mdl']))), canon_mdl=d['canon_mdl'],
                          aceptacion_swaps=d['aceptacion_swaps'], timeouts_sympify=d['parches'].get('timeout_sympify', 0),
                          expresiones_complejas=d['parches'].get('expresion_compleja', 0), minutos=d['tiempo_s']/60,
                          relanzada=fn.endswith('_r.json'), plataforma=d.get('plataforma')))
TODAS = pd.DataFrame(filas).sort_values(['fuente', 'carpeta', 'grupo', 'cadena'])
TODAS.to_csv(OUT/'todas_las_cadenas.csv', index=False)
inf['cadenas'] = TODAS.groupby('fuente').agg(n=('dl_mdl', 'size'), cpu_h=('minutos', lambda m: m.sum()/60)).to_dict('index')

# ---------------------------------------------------------------- informe
(OUT/'C-011_informe.json').write_text(json.dumps(inf, indent=1, default=float), encoding='utf-8')
cols_top = ['id', 'cyc', 'WC', 'EF', 'T', 'prediccion', 'residuo', 'h', 'cook']
cols_fis = ['fuente', 'modelo', 'dl', 'min_dominio', 'frac_creciente_WT', 'frac_creciente_WC', 'rango_WT_0p001',
            'rango_WC_0p01', 'rango_WT_1', 'rango_WC_20', 'canon']
md = ['# C-011 — Informe adicional (§10)', '',
      'El bootstrap de los parámetros del MDL de A y sus comprobaciones físicas están en C-001. Aquí: residuos, influencia, '
      'comprobaciones físicas de todos los MDL de D y la tabla de todas las cadenas.', '',
      '## 1. Residuos e influencia (en muestra, 242 puntos)', '']
for dgn, top in [(diag_A, top_A), (diag_C, top_C)]:
    md += [f"### {dgn['modelo']}", '',
           f"- MSE en muestra {dgn['mse_en_muestra']:.4f} kN²; s² = {dgn['s2']:.4f}; Shapiro p = {dgn['shapiro_p']:.3g}.",
           f"- Breusch-Pagan (Koenker): sobre ŷ LM = {dgn['BP_Koenker_yhat']['LM']:.2f}, p = {dgn['BP_Koenker_yhat']['p']:.3g}; "
           f"sobre WT, WC, EF LM = {dgn['BP_Koenker_WT_WC_EF']['LM']:.2f}, p = {dgn['BP_Koenker_WT_WC_EF']['p']:.3g}.",
           f"- Spearman |e| ~ ŷ: ρ = {dgn['spearman_abs_e_yhat']['rho']:.3f} (p = {dgn['spearman_abs_e_yhat']['p']:.3g}). "
           f"Varianza de e por terciles de ŷ: " + ', '.join(f'{k} {v:.4f}' for k, v in dgn['var_e_por_tercil_yhat'].items()) + '.',
           f"- Cook > 4/n: {dgn['n_cook_mayor_4_n']} puntos. Borrado de casos (3 más influyentes): " +
           '; '.join(f"id {b['id']}: cambio RMS de las demás predicciones {b.get('rms_cambio_pred', float('nan')):.3f} kN, "
                     f"máx. {b.get('max_cambio_pred', float('nan')):.3f}" for b in dgn['borrado_de_casos']) + '.', '',
           a_md(top[cols_top]), '']
md += ['## 2. Residuos fuera de muestra (CV de C-005; por punto, media de las 5 repeticiones)', '', a_md(het), '',
       '### Los 10 puntos más difíciles (ECM medio entre métodos)', '',
       a_md(dificiles[['id', 'cyc', 'WC', 'EF', 'T', 'ecm_medio_metodos'] + MOD]), '',
       '## 3. Comprobaciones físicas de los MDL de D', '',
       f"Resumen de los 25 MDL de la CV: {inf['fisica_resumen_CV']}. 'Tiende a 0' = |f| < 0.5 kN con WT = 0.001 s o WC = 0.01 kA; "
       f"'acotada fuera' = |f| ≤ 2·máx(y) con WT = 1 s y WC = 20 kA. Los rangos son [mín, máx] de f sobre 6 valores de la otra "
       f"variable dentro de su dominio (kN).", '', a_md(FIS[cols_fis]), '',
       '## 4. Todas las cadenas', '', f"`todas_las_cadenas.csv`: {len(TODAS)} cadenas. {inf['cadenas']}"]
(OUT/'C-011_informe.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md[:30]))
