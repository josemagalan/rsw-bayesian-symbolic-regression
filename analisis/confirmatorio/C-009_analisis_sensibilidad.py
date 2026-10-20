"""C-009 (protocolo §9 con D-015): análisis de las sensibilidades. No ajusta ningún modelo; solo lee y evalúa.
Lee:
  resultados/C-005_analisis_cv/predicciones_todas.csv      análisis principal (todas las predicciones, ya con las reglas de C-005)
  resultados/C-007_bms_sensibilidad/<variante>/             BMS: aleatoria, wc_nominal, np4 (repetición 0) y extrapolación
  resultados/C-008_sensibilidad_comparadores/<variante>/    comparadores y PySR: aleatoria, wc_nominal, sin_cero, extrapolación
  resultados/C-004_comparadores_sin_ef/                    comparadores sin EF
Reglas (las de C-005): BMS-MDL = mínima DL entre las cadenas completas (≥ 2; si no, fallo -> media del entrenamiento);
  promedio de muestras = media de las predicciones de todas las muestras de todas las cadenas; no finitos -> media del
  entrenamiento. Se prefiere la cadena relanzada (_r) si existe.
Sensibilidades (§9):
  - aleatoria, wc_nominal: comparadores y PySR en 25 particiones; BMS en las 5 de la repetición 0 (D-015).
  - sin_cero: comparadores y PySR reajustados (C-008); BMS sin reajuste: se excluye el punto de las predicciones de C-003.
  - sin_ef: comparadores sin EF frente al BMS y PySR del análisis principal (que ya no usan EF).
  - np4: BMS con prior nv2.np4 en la repetición 0, frente al BMS principal en las mismas particiones (pareado).
  - recorte: TODAS las predicciones se recortan a [0, 1.2·máx(y_entrenamiento)] (columna mse_rec en todas las tablas).
  - extrapolación: todos los métodos en WT_extremos y WC_extremos.
Todo es secundario y descriptivo (§5: solo H1 y H4a tienen error controlado). Los estadísticos de tipo H1
(t corregido, test bayesiano correlacionado con ROPE ±0.03) se dan como descripción en las variantes de 25 particiones.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-009_analisis_sensibilidad.py   [--prueba-sintetica]"""
import importlib.util, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('c008', AQUI/'C-008_sensibilidad_comparadores.py')
c008 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c008)

PRUEBA = '--prueba-sintetica' in sys.argv
R = ROOT/'resultados'/'_pruebas' if PRUEBA else ROOT/'resultados'
n_ = (lambda real, prueba: R/(prueba if PRUEBA else real))
DIR5, DIR7 = n_('C-005_analisis_cv', 'C-005_sintetica'), n_('C-007_bms_sensibilidad', 'C-007_sintetica')
DIR8, DIR_SINEF = n_('C-008_sensibilidad_comparadores', 'C-008_sintetica'), n_('C-004_comparadores_sin_ef', 'C-004_sintetica_sin_ef')
OUT = n_('C-009_analisis_sensibilidad', 'C-009_sintetica')
DD = R/'datos_derivados' if PRUEBA else ROOT/'analisis'/'datos_derivados'
OUT.mkdir(parents=True, exist_ok=True)
DELTA, RATIO, FACTOR_REC = 0.03, 0.25, 1.2
EXTREMOS = {0: 'WT_extremos', 1: 'WC_extremos'}
inf = dict(protocolo_sha=PROTOCOLO_SHA, protocolo_anexo_sha=c008.PROTOCOLO_ANEXO_SHA, prueba_sintetica=PRUEBA, avisos=[], checks={})


def a_md(t, dec=4):
    cols = list(t.columns)
    fmt = lambda v: (f'{v:.{dec}f}' if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
    filas = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '---|'*len(cols)]
    return '\n'.join(filas + ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in t.itertuples(index=False)])


# ---------------------------------------------------------------- datos y divisiones (las mismas funciones que C-008)
df0 = cargar_datos()
id_cero = int(df0.loc[df0['T'] == 0, 'id'].iloc[0])
if PRUEBA:
    rng = np.random.default_rng(7)
    df0['T'] = 7.0*(1-np.exp(-0.6*df0['WC']**2*df0['WT'])) + rng.normal(0, 0.28, len(df0))
porid = df0.set_index('id')
DIVS = {}
for v in ['principal', 'aleatoria', 'wc_nominal', 'sin_cero', 'extrapolacion']:
    d, divs = c008.preparar(v, df0, id_cero, DD, False)     # todas las particiones (también en la prueba)
    DIVS[v] = {x['p']: dict(media_tr=float(d[~d.id.isin(x['ids'])]['T'].mean()), max_tr=float(d[~d.id.isin(x['ids'])]['T'].max()),
                            ids_ok=set(d[d.id.isin(x['ids'])]['id'])) for x in divs}

largo = []


def anadir(variante, modelo, p, ids, pred, div):
    """div: clave de DIVS del entrenamiento de ese ajuste (da la media y el máximo del entrenamiento)."""
    s = DIVS[div][p]; ids = list(ids)
    pred = np.asarray([np.nan if x is None else x for x in pred], dtype=float)
    malo = ~np.isfinite(pred); pred = np.where(malo, s['media_tr'], pred)
    largo.append(pd.DataFrame(dict(variante=variante, modelo=modelo, particion=p, id=ids, y=porid.loc[ids, 'T'].values,
                                   pred=pred, media_tr=s['media_tr'], max_tr=s['max_tr'], no_finito=malo)))
    return int(malo.sum())


# ---------------------------------------------------------------- análisis principal (C-005)
P = pd.read_csv(DIR5/'predicciones_todas.csv')
inf['checks']['principal_y_max_dif'] = float(np.max(np.abs(P.y.values - porid.loc[P.id, 'T'].values)))
for (m, p), g in P.groupby(['modelo', 'particion']):
    anadir('principal', m, p, g.id, g.pred.values, 'principal')
    if m.startswith('BMS') or m == 'PySR':
        anadir('sin_ef', m, p, g.id, g.pred.values, 'principal')        # el BMS y PySR ya no usan EF
    if m.startswith('BMS'):                                              # D-015: sin reajuste, se quita el punto de la prueba
        gg = g[g.id != id_cero]
        anadir('sin_cero', m, p, gg.id, gg.pred.values, 'principal')


# ---------------------------------------------------------------- comparadores y PySR (C-008, C-004 --sin-ef)
def leer_comparadores(carpeta, variante, div):
    f = carpeta/'predicciones.csv'
    if not f.exists():
        inf['avisos'].append(f'Falta {f.relative_to(ROOT)}: variante {variante} sin comparadores'); return None
    pc = pd.read_csv(f); mse = pd.read_csv(carpeta/'mse_por_particion.csv').fillna({'fallo': ''})
    fallos = mse[mse.fallo != '']
    if len(fallos):
        inf['avisos'].append(f'{variante}: fallos técnicos de comparadores {fallos[["modelo", "particion"]].values.tolist()}')
    for (m, p), g in pc.groupby(['modelo', 'particion']):
        assert set(g.id) == DIVS[div][p]['ids_ok'], (variante, m, p)
        anadir(variante, m, p, g.id, g.pred.values, div)
    return mse


def leer_pysr(carpeta, variante, div):
    fs = sorted(carpeta.glob('p??.json'))
    if not fs:
        inf['avisos'].append(f'{variante}: PySR no ejecutado (sin p??.json en {carpeta.relative_to(ROOT)})'); return []
    hechos, filas = set(), []
    for f in fs:
        d = json.loads(f.read_text(encoding='utf-8')); p = d['p']; hechos.add(p)
        assert set(d['ids_test']) == DIVS[div][p]['ids_ok'], (variante, p)
        anadir(variante, 'PySR', p, d['ids_test'], d['pred'], div)
        filas.append(dict(variante=variante, particion=p, ecuacion=d['ecuacion'], complejidad=d['complejidad'],
                          mse_c008=d['mse'], cpu_s=d.get('cpu_s')))
    for p in sorted(set(DIVS[div]) - hechos):          # fallo de PySR -> media del entrenamiento (como C-005)
        ids = sorted(DIVS[div][p]['ids_ok']); anadir(variante, 'PySR', p, ids, [None]*len(ids), div)
        inf['avisos'].append(f'{variante}: PySR sin resultado en p{p:02d} -> media del entrenamiento')
    return filas


mse8, pysr_filas = {}, []
for v in ['aleatoria', 'wc_nominal', 'sin_cero', 'extrapolacion']:
    mse8[v] = leer_comparadores(DIR8/v, v, v)
    pysr_filas += leer_pysr(DIR8/v/'pysr', v, v)
mse8['sin_ef'] = leer_comparadores(DIR_SINEF, 'sin_ef', 'principal')
pd.DataFrame(pysr_filas).to_csv(OUT/'pysr_sensibilidad.csv', index=False)


# ---------------------------------------------------------------- BMS (C-007)
def cadenas(carpeta, prefijo):
    cs = {}
    for f in sorted(carpeta.glob(f'{prefijo}_cadena*.json')):
        m = re.match(rf'{prefijo}_cadena(\d+)(_r)?\.json$', f.name)
        if m and (int(m.group(1)) not in cs or m.group(2)):
            cs[int(m.group(1))] = json.loads(f.read_text(encoding='utf-8'))
    return cs


def a_vector(v, n, relleno):
    a = np.array([np.nan if x is None else x for x in (v if v is not None else [None]*n)], dtype=float)
    return np.where(np.isfinite(a), a, relleno)


MOTIVOS = {'inv_WT': r'/WT\b|WT\*\*\(-1\)', 'log_WC': r'log\(WC\b', 'periodica': r'\b(sin|cos|tan)\('}
filas_bms = []
for v, div in [('aleatoria', 'aleatoria'), ('wc_nominal', 'wc_nominal'), ('np4', 'principal'), ('extrapolacion', 'extrapolacion')]:
    for p in sorted(DIVS[div]):
        if v != 'extrapolacion' and p >= 5:
            continue
        cs = cadenas(DIR7/v, EXTREMOS[p] if v == 'extrapolacion' else f'p{p:02d}')
        if not cs:
            if not PRUEBA:
                inf['avisos'].append(f'BMS {v} p{p:02d}: sin cadenas')
            continue
        ref = next(iter(cs.values()))['cfg']['ids_test']
        for c, d in cs.items():
            assert d['cfg']['ids_test'] == ref and set(ref) == DIVS[div][p]['ids_ok'], (v, p, c)
        s = DIVS[div][p]
        fila = dict(variante=v, particion=p, division=EXTREMOS[p] if v == 'extrapolacion' else '', n_cadenas=len(cs),
                    minutos=sum(d['tiempo_s'] for d in cs.values())/60,
                    timeouts=sum(len(d['expresiones_timeout']) for d in cs.values()))
        if len(cs) < 2:   # fallo del BMS (§7)
            nf = anadir(v, 'BMS_MDL', p, ref, [None]*len(ref), div); anadir(v, 'BMS_promedio', p, ref, [None]*len(ref), div)
            fila.update(fallo_bms=True)
        else:
            cmin = min(cs, key=lambda c: cs[c]['dl_mdl']); d = cs[cmin]
            nf = anadir(v, 'BMS_MDL', p, ref, d['pred_mdl_test'], div)
            B = np.array([a_vector(m.get('pred'), len(ref), s['media_tr']) for dc in cs.values() for m in dc['muestras']])
            anadir(v, 'BMS_promedio', p, ref, B.mean(0), div)
            dls = sorted(dc['dl_mdl'] for dc in cs.values())
            fila.update(fallo_bms=False, cadena_mdl=cmin, dl_mdl=d['dl_mdl'], canon=d['canon_mdl'],
                        n_param=len(set(re.findall(r'_a\d+_', d['modelo_mdl']))), dl_rango=dls[-1] - dls[0],
                        cadenas_misma_estructura=sum(dc['canon_mdl'] == d['canon_mdl'] for dc in cs.values()),
                        n_muestras=len(B), muestras_fuera_de_rango=int(np.sum(np.any((B < 0) | (B > FACTOR_REC*s['max_tr']), axis=1))),
                        max_abs_muestra=float(np.max(np.abs(B))),
                        **{k: bool(re.search(r, d['canon_mdl'])) for k, r in MOTIVOS.items()})
        fila['no_finitos_mdl'] = nf
        filas_bms.append(fila)
tb = pd.DataFrame(filas_bms); tb.to_csv(OUT/'bms_sensibilidad.csv', index=False)
inf['bms_cpu_h_C007'] = float(tb.minutos.sum()/60) if len(tb) else 0.0

L = pd.concat(largo, ignore_index=True)
L['pred_rec'] = np.clip(L.pred, 0, FACTOR_REC*L.max_tr)
L['fuera_de_rango'] = (L.pred < 0) | (L.pred > FACTOR_REC*L.max_tr)
L.to_csv(OUT/'predicciones_sensibilidad.csv', index=False)
M = (L.assign(e2=(L.y - L.pred)**2, e2r=(L.y - L.pred_rec)**2)
      .groupby(['variante', 'modelo', 'particion'])
      .agg(mse=('e2', 'mean'), mse_rec=('e2r', 'mean'), n_test=('y', 'size'), fuera_de_rango=('fuera_de_rango', 'sum'),
           no_finitos=('no_finito', 'sum')).reset_index())
M.to_csv(OUT/'mse_por_particion.csv', index=False)

# comprobaciones: MSE recalculados frente a los ficheros de origen
m5 = pd.read_csv(DIR5/'metricas_por_particion.csv')
x = M[M.variante == 'principal'].merge(m5[['modelo', 'particion', 'mse']], on=['modelo', 'particion'], suffixes=('', '_c005'))
inf['checks']['mse_principal_vs_C005_max_dif'] = float(np.max(np.abs(x.mse - x.mse_c005)))
for v, mse in mse8.items():
    if mse is not None:
        x = M[M.variante == v].merge(mse[['modelo', 'particion', 'mse']], on=['modelo', 'particion'], suffixes=('', '_src'))
        inf['checks'][f'mse_{v}_vs_fuente_max_dif'] = float(np.max(np.abs(x.mse - x.mse_src))) if len(x) else None
if pysr_filas:
    x = M[M.modelo == 'PySR'].merge(pd.DataFrame(pysr_filas), on=['variante', 'particion'])
    inf['checks']['mse_pysr_vs_C008_max_dif'] = float(np.max(np.abs(x.mse - x.mse_c008))) if len(x) else None

# ---------------------------------------------------------------- 1. resumen por variante y modelo
orden = ['principal', 'aleatoria', 'wc_nominal', 'sin_cero', 'sin_ef', 'np4']
Mi = M[M.variante != 'extrapolacion']
res = (Mi.groupby(['variante', 'modelo'])
         .agg(mse=('mse', 'mean'), mse_de=('mse', 'std'), mse_rec=('mse_rec', 'mean'), n_part=('mse', 'size'),
              fuera_de_rango=('fuera_de_rango', 'sum'), no_finitos=('no_finitos', 'sum')).reset_index())
res['variante'] = pd.Categorical(res.variante, orden, ordered=True); res = res.sort_values(['variante', 'mse'])
res.to_csv(OUT/'resumen_variantes.csv', index=False)
tabla_ancha = res.astype({'variante': str}).pivot(index='modelo', columns='variante', values='mse')
tabla_ancha = tabla_ancha[[c for c in orden if c in tabla_ancha]].sort_values('principal')
tabla_ancha.to_csv(OUT/'mse_modelo_por_variante.csv')
tabla_rec = res.astype({'variante': str}).pivot(index='modelo', columns='variante', values='mse_rec')
tabla_rec = tabla_rec[[c for c in orden if c in tabla_rec]].loc[tabla_ancha.index]
tabla_rec.to_csv(OUT/'mse_recortado_modelo_por_variante.csv')
r0 = Mi[Mi.particion < 5].groupby(['variante', 'modelo']).mse.mean().unstack(0)     # las 5 particiones del BMS de C-007
r0 = r0[[c for c in orden if c in r0]].sort_values('principal'); r0.to_csv(OUT/'mse_modelo_por_variante_rep0.csv')


# ---------------------------------------------------------------- 2. estadísticos de tipo H1 (descriptivos, 25 particiones)
def contraste(a, b):
    """a, b: MSE por partición (Series con el mismo índice). d = a − b, como C-005."""
    d = (a - b).values; n = len(d); dbar = d.mean(); sd = d.std(ddof=1)
    se = np.sqrt((1/n + RATIO)*sd**2); gl = n - 1; q95 = stats.t.ppf(0.95, gl); post = stats.t(gl, loc=dbar, scale=se)
    return dict(n=n, mse_a=a.mean(), mse_b=b.mean(), d_media=dbar, d_de=sd, lim_inf_unilateral95=dbar - q95*se,
                lim_sup_unilateral95=dbar + q95*se, P_a_mejor=post.cdf(-DELTA), P_equivalentes=post.cdf(DELTA) - post.cdf(-DELTA),
                P_a_peor=post.sf(DELTA), frac_a_mejor=float(np.mean(d < 0)),
                regla_H1=('no inferior' if dbar + q95*se < DELTA else 'inferior' if dbar - q95*se > DELTA else 'no concluyente'))


filas = []
for v in ['principal', 'sin_cero', 'sin_ef']:
    for col in ['mse', 'mse_rec']:
        W = M[M.variante == v].pivot(index='particion', columns='modelo', values=col)
        if len(W) < 10:
            continue
        for b in [x for x in ['BMS_MDL', 'BMS_promedio'] if x in W]:
            for c in [m for m in W.columns if not m.startswith('BMS') and m != 'media']:
                if W[[b, c]].notna().all(axis=None):
                    filas.append(dict(variante=v + ('_recortado' if col == 'mse_rec' else ''), bms=b, comparador=c,
                                      **contraste(W[b], W[c])))
cmp25 = pd.DataFrame(filas); cmp25.to_csv(OUT/'contrastes_25_particiones.csv', index=False)

# ---------------------------------------------------------------- 3. repetición 0: BMS de cada variante (5 particiones)
filas = []
W0 = M[(M.particion < 5) & (M.variante != 'extrapolacion')]
bp = W0[W0.variante == 'principal'].pivot(index='particion', columns='modelo', values='mse')
for v in ['aleatoria', 'wc_nominal', 'np4']:
    Wv = W0[W0.variante == v].pivot(index='particion', columns='modelo', values='mse')
    if 'BMS_MDL' not in Wv:
        continue
    ref = bp if v == 'np4' else Wv     # comparadores ajustados con los mismos datos y particiones que el BMS de la variante
    for b in ['BMS_MDL', 'BMS_promedio']:
        cands = [(f'{b} principal', bp, b, v != 'aleatoria')] + \
                [(c, ref, c, True) for c in ref.columns if not c.startswith('BMS') and c != 'media']
        for nombre, tabla, col, pareado in cands:
            if col not in tabla:
                continue
            ok = Wv[b].dropna().index.intersection(tabla[col].dropna().index)
            d = Wv.loc[ok, b] - tabla.loc[ok, col]
            filas.append(dict(variante=v, bms=b, frente_a=nombre, pareado=pareado, n=len(d), mse_bms=Wv.loc[ok, b].mean(),
                              mse_ref=tabla.loc[ok, col].mean(), d_media=d.mean(), d_min=d.min(), d_max=d.max(),
                              n_bms_mejor=int((d < 0).sum())))
rep0 = pd.DataFrame(filas); rep0.to_csv(OUT/'repeticion0_bms.csv', index=False)

# ---------------------------------------------------------------- 4. cambio de cada método respecto al análisis principal
filas = []
base = M[M.variante == 'principal'].set_index(['modelo', 'particion'])
for v in ['aleatoria', 'wc_nominal', 'sin_cero', 'sin_ef']:
    for m, g in M[M.variante == v].groupby('modelo'):
        if m not in base.index.get_level_values(0) or len(g) < 10:
            continue
        g = g.set_index('particion'); b = base.loc[m].loc[g.index]; d = g.mse - b.mse
        fila = dict(variante=v, modelo=m, n=len(d), mse_principal=b.mse.mean(), mse_variante=g.mse.mean(), d_media=d.mean(),
                    ic95_inf=np.nan, ic95_sup=np.nan)
        if v != 'aleatoria':   # mismas particiones -> pareado, con la corrección de Nadeau y Bengio
            se = np.sqrt((1/len(d) + RATIO)*d.var(ddof=1)); q = stats.t.ppf(0.975, len(d) - 1)
            fila.update(ic95_inf=d.mean() - q*se, ic95_sup=d.mean() + q*se)
        filas.append(fila)
cambio = pd.DataFrame(filas); cambio.to_csv(OUT/'cambio_respecto_principal.csv', index=False)

# ---------------------------------------------------------------- 5. extrapolación
E = M[M.variante == 'extrapolacion'].copy(); E['division'] = E.particion.map(EXTREMOS)
cvp = res[res.variante == 'principal'].set_index('modelo').mse
if len(E):
    ext = E.pivot_table(index='modelo', columns='division', values=['mse', 'mse_rec'])
    ext.columns = [f'{a}_{b}' for a, b in ext.columns]
    ext['mse_cv_principal'] = cvp.reindex(ext.index).astype(float)
    for dv in EXTREMOS.values():
        if f'mse_{dv}' in ext:
            ext[f'razon_{dv}'] = ext[f'mse_{dv}']/ext.mse_cv_principal
    ext = ext.sort_values(ext.columns[0])
else:
    ext = pd.DataFrame()
ext.to_csv(OUT/'extrapolacion.csv')

# ---------------------------------------------------------------- informe
(OUT/'C-009_informe.json').write_text(json.dumps(inf, indent=1, default=str), encoding='utf-8')
cols_h1 = ['variante', 'bms', 'mse_a', 'mse_b', 'd_media', 'lim_inf_unilateral95', 'lim_sup_unilateral95',
           'P_a_mejor', 'P_equivalentes', 'P_a_peor', 'regla_H1']
md = [f'# C-009 — Análisis de sensibilidad (§9, D-015){" — PRUEBA SINTÉTICA" if PRUEBA else ""}', '',
      'Todo es secundario y descriptivo. MSE en kN². `mse_rec`: predicciones recortadas a [0, 1.2·máx(y_tr)].', '',
      '## Avisos', ''] + [f'- {a}' for a in inf['avisos']] + ['', '## Comprobaciones', ''] + \
     [f'- {k}: {v}' for k, v in inf['checks'].items()] + \
     ['', '## 1. MSE medio por modelo y variante (todas las particiones disponibles)', '', a_md(tabla_ancha.reset_index()),
      '', '### Con recorte', '', a_md(tabla_rec.reset_index()),
      '', '### Repetición 0 (las 5 particiones del BMS de C-007)', '', a_md(r0.reset_index()),
      '', '### Detalle', '', a_md(res.astype({'variante': str})),
      '', '## 2. Estadísticos de tipo H1 frente a EBM (descriptivos; d = BMS − EBM)', '',
      a_md(cmp25[cmp25.comparador == 'EBM'][cols_h1]) if len(cmp25) else '(sin datos)',
      '', '## 3. BMS de cada variante en la repetición 0 (d = BMS − referencia)', '', a_md(rep0) if len(rep0) else '(sin datos)',
      '', '## 4. Cambio de cada método respecto al análisis principal (d = variante − principal)', '',
      a_md(cambio) if len(cambio) else '(sin datos)',
      '', '## 5. Extrapolación', '', a_md(ext.reset_index()) if len(ext) else '(sin datos)',
      '', '## 6. BMS: MDL por variante y partición', '', a_md(tb) if len(tb) else '(sin datos)', '',
      f'Coste del BMS en C-007: {inf["bms_cpu_h_C007"]:.1f} CPU·h.']
(OUT/'C-009_informe.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md[:14]))
print(tabla_ancha.round(4).to_string())
print(ext.round(3).to_string())
