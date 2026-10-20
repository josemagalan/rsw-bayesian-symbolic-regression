"""C-005 (CONFIRMATORIO, protocolo §3–§7): análisis estadístico de la validación cruzada (análisis B).
Lee (sin volver a ajustar nada salvo H3a):
  resultados/C-003_bms_cv/p??_cadena?.json      BMS configuración D, 4 cadenas por partición
  resultados/C-002_pysr_cv/p??.json             PySR
  resultados/C-004_comparadores/predicciones.csv y mse_por_particion.csv   comparadores 1–6, 8 y 9
  resultados/C-001_analisis_A/D_cadena?.json    MDL del análisis A (solo para H3a)
Reglas del protocolo que se aplican aquí:
  - BMS-MDL de cada partición: mínima DL entre las cadenas completas; hacen falta ≥ 2 cadenas completas,
    si no, fallo del BMS (predicción = media del entrenamiento), nunca se descarta la partición (§7).
  - BMS promedio de muestras: media de las predicciones de las muestras (pasos 10 000–20 000, 1 de cada 100)
    juntando las cadenas (§3, secundaria).
  - Predicciones no finitas → media del entrenamiento, y se cuentan (§3).
  - H1 (§6): d_j = MSE_BMS-MDL,j − MSE_EBM,j; se cumple si d̄ + t0.95,24·√[(1/25 + 0.25)·s_d²] < 0.03;
    "inferior" solo si d̄ − t0.95,24·√[…] > 0.03; si no, no concluyente.
  - Test t bayesiano correlacionado (ρ = 0.2, ROPE ±0.03) frente a cada comparador (fórmula cerrada y,
    como comprobación, baycomp). Contrastes secundarios de superioridad bilaterales con Holm (familia: los
    8 comparadores preespecificados; E-001 es exploratorio y queda fuera de la familia).
  - H2 descriptiva; H3a descriptiva (equivalencia funcional con el MDL de A, RMSE < 0.05 kN en las 121 celdas
    nominales); H3b confirmatoria (≥ 70 % de particiones con R²_ui ≥ 0.90 y p_T ≥ 0.01).
  - Métricas secundarias (RMSE, MAE, R²), clasificación con umbral 5.93 kN, complejidad y coste (§4).
  - Diagnósticos del BMS por partición (§7): concordancia entre cadenas, R̂ y ESS de la traza de DL,
    aceptación de intercambios, intervenciones de los parches.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-005_analisis_cv.py
     ... --prueba-sintetica   (lee las salidas de prueba de resultados\\_pruebas y escribe en _pruebas\\C-005_sintetica)
"""
import ast, collections, json, re, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, sympy as sp
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import factorial
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA

warnings.filterwarnings('ignore')
PRUEBA = '--prueba-sintetica' in sys.argv
if PRUEBA:
    R = ROOT/'resultados'/'_pruebas'
    DIR_A, DIR_PYSR, DIR_BMS, DIR_COMP, OUT = (R/'C-001_sintetica', R/'C-002_sintetica', R/'C-003_sintetica',
                                               R/'C-004_sintetica', R/'C-005_sintetica')
else:
    R = ROOT/'resultados'
    DIR_A, DIR_PYSR, DIR_BMS, DIR_COMP, OUT = (R/'C-001_analisis_A', R/'C-002_pysr_cv', R/'C-003_bms_cv',
                                               R/'C-004_comparadores', R/'C-005_analisis_cv')
OUT.mkdir(parents=True, exist_ok=True)

DELTA = 0.03            # margen de no inferioridad y ROPE (kN²)
RATIO = 0.25            # n_te/n_tr (corrección de Nadeau y Bengio)
RHO = 0.2               # ρ = n_te/n; RHO/(1-RHO) = RATIO
UMBRAL = 5.93           # kN, umbral de aceptación (clasificación descriptiva)
TOL_EQUIV = 0.05        # kN, H3a
FAMILIA = ['media', 'cuadratico_EN', 'GAM', 'RF', 'GBM', 'EBM', 'PySR', 'TabPFN_v2']   # Holm
EXPLORATORIOS = ['E001_exploratorio']
BMS = ['BMS_MDL', 'BMS_promedio']
WT_s, WC_s = sp.symbols('WT WC')
GRID_WT = np.arange(2, 13)*0.02
GRID_WC = np.linspace(1.5, 6.5, 11)
G_WT, G_WC = np.meshgrid(GRID_WT, GRID_WC, indexing='ij')
inf = dict(protocolo_sha=PROTOCOLO_SHA, prueba_sintetica=PRUEBA, avisos=[])


def a_md(t, dec=4):
    cols = list(t.columns)
    fmt = lambda v: (f'{v:.{dec}f}' if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
    filas = ['| ' + ' | '.join(cols) + ' |', '|' + '---|'*len(cols)]
    filas += ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in t.itertuples(index=False)]
    return '\n'.join(filas)


# ---------------------------------------------------------------- datos y particiones
df = cargar_datos()
if PRUEBA:   # misma respuesta sintética que C-002/C-003/C-004 --prueba-sintetica
    rng = np.random.default_rng(7)
    df['T'] = 7.0*(1-np.exp(-0.6*df['WC']**2*df['WT'])) + rng.normal(0, 0.28, len(df))
folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
porid = df.set_index('id')
PARTS = sorted({5*r + k for r, k in folds[['repeticion', 'particion']].drop_duplicates().itertuples(index=False)})


def ids_test(p):
    return folds.query('repeticion == @p // 5 and particion == @p % 5')['id'].tolist()


def entrenamiento(p):
    te = set(ids_test(p)); return df[~df.id.isin(te)]


# ---------------------------------------------------------------- comparadores (C-004)
mse_c = pd.read_csv(DIR_COMP/'mse_por_particion.csv')
pred_c = pd.read_csv(DIR_COMP/'predicciones.csv')
partes_disp = sorted(set(mse_c.particion))
if partes_disp != PARTS:
    PARTS = partes_disp
    inf['avisos'].append(f'Particiones disponibles: {PARTS}')
n = len(PARTS)
fallos_c = mse_c[mse_c.fallo.notna() & (mse_c.fallo.astype(str) != '')]
inf['comparadores_fallidos'] = fallos_c[['modelo', 'particion', 'fallo']].to_dict('records')
largo = [pred_c[['modelo', 'particion', 'id', 'y', 'pred']].assign(sustituida=False)]
media_tr = {p: float(entrenamiento(p)['T'].mean()) for p in PARTS}
# comprobaciones de coherencia
chk = pred_c[pred_c.modelo == 'media']
inf['check_media_entrenamiento_max_dif'] = float(max(abs(chk[chk.particion == p].pred - media_tr[p]).max() for p in PARTS))
inf['check_y_max_dif'] = float(np.max(np.abs(pred_c.y.values - porid.loc[pred_c.id, 'T'].values)))
for p in PARTS:
    assert sorted(pred_c[(pred_c.modelo == 'EBM') & (pred_c.particion == p)].id) == sorted(ids_test(p)), p


# ---------------------------------------------------------------- utilidades
def a_finito(v, p):
    a = np.full(len(ids_test(p)), np.nan) if v is None else np.array([np.nan if x is None else x for x in v], dtype=float)
    malo = ~np.isfinite(a); a[malo] = media_tr[p]
    return a, int(malo.sum())


def funcion_bms(modelo, params):
    ex = sp.sympify(modelo)
    usados = sorted(str(s) for s in ex.free_symbols if str(s) not in ('WT', 'WC'))
    fl = sp.lambdify([WT_s, WC_s] + [sp.Symbol(u) for u in usados], ex, ['numpy', {'fac': factorial}])
    th = np.array([params[u] for u in usados], dtype=float)

    def f(t, c, th=th):
        t, c = np.asarray(t, dtype=float), np.asarray(c, dtype=float)
        with np.errstate(all='ignore'):
            v = np.asarray(fl(t, c, *th))
            if np.iscomplexobj(v):
                v = np.where(np.abs(v.imag) < 1e-12, v.real, np.nan)
            return v.astype(float)*np.ones(np.broadcast(t, c).shape)
    return f, fl, usados, th


def nodos(expr):
    tr = ast.parse(expr.replace('^', '**'), mode='eval')
    funcs = {id(x.func) for x in ast.walk(tr) if isinstance(x, ast.Call)}
    return sum(1 for x in ast.walk(tr) if isinstance(x, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant))
               or (isinstance(x, ast.Name) and id(x) not in funcs))


def criterio_h3b(f, h=1e-4):   # idéntico a E-008 (calibración)
    lt, lc = np.log(G_WT), np.log(G_WC)
    dT = np.ravel((f(np.exp(lt + h), G_WC) - f(np.exp(lt - h), G_WC))/(2*h))
    dC = np.ravel((f(G_WT, np.exp(lc + h)) - f(G_WT, np.exp(lc - h)))/(2*h))
    ok = np.isfinite(dT) & np.isfinite(dC); dT, dC = dT[ok], dC[ok]
    sC, sT = np.sum(dC**2), np.sum(dT**2)
    if sC <= 0 or sT + sC <= 0:
        return dict(gamma=np.nan, R2_ui=np.nan, p_T=np.nan, n_celdas=int(ok.sum()))
    g = np.sum(dT*dC)/sC
    return dict(gamma=float(g), R2_ui=float(1 - np.sum((dT - g*dC)**2)/sT) if sT > 0 else np.nan,
                p_T=float(sT/(sT + sC)), n_celdas=int(ok.sum()))


def rhat_ess(trazas):
    """R̂ (split, Gelman-Rubin clásico) y ESS (Geyer, secuencia inicial positiva) de trazas de igual longitud."""
    L = min(len(t) for t in trazas)
    X = np.array([t[:L] for t in trazas], dtype=float)
    if not np.all(np.isfinite(X)) or L < 8:
        return np.nan, np.nan
    h = L//2; S = np.vstack([X[:, :h], X[:, h:2*h]]); m, nn = S.shape
    W = S.var(axis=1, ddof=1).mean(); B = nn*S.mean(axis=1).var(ddof=1)
    rh = np.sqrt(((nn-1)/nn*W + B/nn)/W) if W > 0 else np.nan
    ess = 0.0
    for x in X:
        x = x - x.mean(); v = x.var()
        if v == 0:
            continue
        f = np.fft.rfft(x, 2*L); ac = np.fft.irfft(f*np.conj(f))[:L]/(L*v)
        s, k = 0.0, 0
        while 2*k + 1 < L:
            par = ac[2*k] + ac[2*k+1]
            if par <= 0:
                break
            s += par; k += 1
        ess += L/max(2*s - 1, 1e-12)
    return float(rh), float(ess)


# ---------------------------------------------------------------- BMS (C-003)
cad = collections.defaultdict(dict)
for fch in sorted(DIR_BMS.glob('p??_cadena*.json')):
    if fch.name.endswith('.progreso.json'):
        continue
    m = re.match(r'p(\d\d)_cadena(\d+)(_r)?\.json$', fch.name)
    p, c = int(m.group(1)), int(m.group(2))
    if p in PARTS and (c not in cad[p] or m.group(3)):
        cad[p][c] = json.loads(fch.read_text(encoding='utf-8'))
inf['bms_truncadas'] = sorted(x.name for x in DIR_BMS.glob('*_TIEMPO_AGOTADO.txt'))
inf['bms_errores'] = sorted(x.name for x in DIR_BMS.glob('*_ERROR.txt'))
filas_bms, filas_cad, muestras_k = [], [], []
mdl_func = {}
for p in PARTS:
    te = ids_test(p); cs = cad.get(p, {})
    for c, d in sorted(cs.items()):
        assert sorted(d['cfg']['ids_test']) == sorted(te), (p, c)
        filas_cad.append(dict(particion=p, cadena=c, dl_mdl=d['dl_mdl'], paso_mdl=d['paso_mdl'], canon=d['canon_mdl'],
                              n_param=len(set(re.findall(r'_a\d+_', d['modelo_mdl']))), aceptacion_swaps=d['aceptacion_swaps'],
                              timeouts=d['parches'].get('timeout_sympify', 0), complejas=d['parches'].get('expresion_compleja', 0),
                              excepciones=d['parches'].get('excepcion', 0), minutos=d['tiempo_s']/60,
                              relanzada=d['cfg']['nombre'].endswith('_r'), modelo=d['modelo_mdl']))
        muestras_k += [len(set(re.findall(r'c\d+', s['canon']))) for s in d['muestras']]
    fila = dict(particion=p, n_cadenas_completas=len(cs))
    ord_ids = cs[min(cs)]['cfg']['ids_test'] if cs else te
    if len(cs) < 2:   # fallo del BMS (§7)
        pm, nf = np.full(len(te), media_tr[p]), len(te)
        ppost, nfp, nmu = pm.copy(), len(te), 0
        fila.update(fallo_bms=True)
    else:
        cmin = min(cs, key=lambda c: cs[c]['dl_mdl']); d = cs[cmin]
        pm, nf = a_finito(d['pred_mdl_test'], p)
        bloques = []; nfp = 0
        for c, dc in cs.items():
            for s in dc['muestras']:
                a, k = a_finito(s.get('pred'), p); bloques.append(a); nfp += k
        ppost = np.mean(bloques, axis=0); nmu = len(bloques)
        f, fl, usados, th = funcion_bms(d['modelo_mdl'], d['parametros_mdl']['d0'])
        mdl_func[p] = f
        xt = porid.loc[ord_ids]
        dif_lamb = float(np.nanmax(np.abs(f(xt.WT.values, xt.WC.values) - np.asarray(d['pred_mdl_test'], dtype=float))))
        dls = sorted(dc['dl_mdl'] for dc in cs.values())
        burn_idx = d['cfg']['burn']//10
        rh, ess = rhat_ess([dc['traza_cada10'][burn_idx:] for dc in cs.values()])
        fila.update(fallo_bms=False, cadena_mdl=cmin, dl_mdl=d['dl_mdl'], canon=d['canon_mdl'], modelo=d['modelo_mdl'],
                    n_param=len(usados), nodos=nodos(d['modelo_mdl']),
                    cadenas_misma_estructura=sum(dc['canon_mdl'] == d['canon_mdl'] for dc in cs.values()),
                    dl_rango=dls[-1] - dls[0], dl_segunda=dls[1] - dls[0], rhat_dl=rh, ess_dl=ess,
                    max_dif_lambdify=dif_lamb, **criterio_h3b(f))
    fila.update(no_finitos_mdl=nf, no_finitos_muestras=nfp, n_muestras=nmu)
    filas_bms.append(fila)
    yv = porid.loc[ord_ids, 'T'].values
    largo.append(pd.DataFrame(dict(modelo='BMS_MDL', particion=p, id=ord_ids, y=yv, pred=pm, sustituida=False)))
    largo.append(pd.DataFrame(dict(modelo='BMS_promedio', particion=p, id=ord_ids, y=yv, pred=ppost, sustituida=False)))
tb = pd.DataFrame(filas_bms); tcad = pd.DataFrame(filas_cad)
tb.to_csv(OUT/'bms_por_particion.csv', index=False); tcad.to_csv(OUT/'bms_cadenas.csv', index=False)

# ---------------------------------------------------------------- PySR (C-002)
filas_py = []
for p in PARTS:
    fch = DIR_PYSR/f'p{p:02d}.json'
    if not fch.exists():
        inf['avisos'].append(f'PySR sin resultado en la partición {p}: se usa la media (fallo)')
        a, nf = np.full(len(ids_test(p)), media_tr[p]), len(ids_test(p)); ids = ids_test(p); fila = dict(particion=p, fallo=True)
    else:
        d = json.loads(fch.read_text(encoding='utf-8')); ids = d['ids_test']
        assert sorted(ids) == sorted(ids_test(p)), p
        a, nf = a_finito(d['pred'], p)
        fila = dict(particion=p, fallo=False, ecuacion=d['ecuacion'], complejidad=int(d['complejidad']),
                    n_constantes=int(d['n_constantes']), nodos=nodos(d['ecuacion']), cpu_s=d.get('cpu_s'),
                    mse_c002=d['mse'])
    fila['no_finitos'] = nf; filas_py.append(fila)
    largo.append(pd.DataFrame(dict(modelo='PySR', particion=p, id=ids, y=porid.loc[ids, 'T'].values, pred=a, sustituida=False)))
tpy = pd.DataFrame(filas_py); tpy.to_csv(OUT/'pysr_por_particion.csv', index=False)
L = pd.concat(largo, ignore_index=True)
L.to_csv(OUT/'predicciones_todas.csv', index=False)


# ---------------------------------------------------------------- métricas por partición
def metricas(g):
    y, q = g.y.values, g.pred.values; e = y - q
    sse = np.sum(e**2); sst = np.sum((y - y.mean())**2)
    ry, rq = y >= UMBRAL, q >= UMBRAL
    rec = [np.mean(rq[ry] == ry[ry]) if ry.any() else np.nan, np.mean(rq[~ry] == ry[~ry]) if (~ry).any() else np.nan]

    def f1(pos_y, pos_q):
        tp = np.sum(pos_y & pos_q); fp = np.sum(~pos_y & pos_q); fn = np.sum(pos_y & ~pos_q)
        return 2*tp/(2*tp + fp + fn) if (2*tp + fp + fn) > 0 else np.nan
    return pd.Series(dict(mse=sse/len(y), rmse=np.sqrt(sse/len(y)), mae=np.mean(np.abs(e)),
                          r2=1 - sse/sst if sst > 0 else np.nan, error_clas=np.mean(ry != rq),
                          exactitud_equilibrada=np.nanmean(rec), f1_aceptable=f1(ry, rq), f1_rechazo=f1(~ry, ~rq),
                          n_test=len(y)))


M = L.groupby(['modelo', 'particion']).apply(metricas).reset_index()
M.to_csv(OUT/'metricas_por_particion.csv', index=False)
chk = M.merge(mse_c[['modelo', 'particion', 'mse']], on=['modelo', 'particion'], suffixes=('', '_c004'))
inf['check_mse_c004_max_dif'] = float(np.max(np.abs(chk.mse - chk.mse_c004)))
if len(tpy) and 'mse_c002' in tpy:
    cp = M[M.modelo == 'PySR'].merge(tpy[['particion', 'mse_c002']], on='particion')
    inf['check_mse_c002_max_dif'] = float(np.nanmax(np.abs(cp.mse - cp.mse_c002)))
W = M.pivot(index='particion', columns='modelo', values='mse').loc[PARTS]
W.to_csv(OUT/'mse_por_particion_ancho.csv')

# coste de cómputo (CPU·h; BMS: tiempo de reloj de procesos de un hilo ≈ CPU)
coste = mse_c.groupby('modelo').cpu_s.sum()/3600
coste['PySR'] = tpy.cpu_s.sum()/3600 if 'cpu_s' in tpy else np.nan
coste['BMS_MDL'] = coste['BMS_promedio'] = tcad.minutos.sum()/60 if len(tcad) else np.nan
res = M.groupby('modelo').agg(mse=('mse', 'mean'), mse_de=('mse', 'std'), rmse=('rmse', 'mean'), mae=('mae', 'mean'),
                              r2=('r2', 'mean'), error_clas=('error_clas', 'mean'),
                              exactitud_equilibrada=('exactitud_equilibrada', 'mean'),
                              f1_aceptable=('f1_aceptable', 'mean'), f1_rechazo=('f1_rechazo', 'mean')).reset_index()
res['cpu_h_total'] = res.modelo.map(coste)
nf_c = mse_c.groupby('modelo').no_finitos.sum()
res['no_finitos'] = res.modelo.map(lambda m: int(tb.no_finitos_mdl.sum()) if m == 'BMS_MDL' else
                                   int(tb.no_finitos_muestras.sum()) if m == 'BMS_promedio' else
                                   int(tpy.no_finitos.sum()) if m == 'PySR' else int(nf_c.get(m, 0)))
res = res.sort_values('mse'); res.to_csv(OUT/'resumen_modelos.csv', index=False)


# ---------------------------------------------------------------- comparaciones (§5)
def holm(ps):
    ps = np.asarray(ps, dtype=float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i)*ps[j])); adj[j] = run
    return adj


try:
    from baycomp.single import CorrelatedTTest
except Exception:
    CorrelatedTTest = None
filas_cmp = []
for b in BMS:
    for c in FAMILIA + EXPLORATORIOS + [x for x in BMS if x != b]:
        if c not in W:
            continue
        d = (W[b] - W[c]).values; dbar = d.mean(); sd = d.std(ddof=1)
        se = np.sqrt((1/n + RATIO)*sd**2); gl = n - 1
        t = dbar/se if se > 0 else np.nan; q95, q975 = stats.t.ppf(0.95, gl), stats.t.ppf(0.975, gl)
        post = stats.t(gl, loc=dbar, scale=se)
        fila = dict(bms=b, comparador=c, en_familia_holm=c in FAMILIA, mse_bms=W[b].mean(), mse_comp=W[c].mean(),
                    d_media=dbar, d_de=sd, ee_corregido=se, t=t, p_bilateral=2*stats.t.sf(abs(t), gl),
                    ic95_inf=dbar - q975*se, ic95_sup=dbar + q975*se,
                    lim_sup_unilateral95=dbar + q95*se, lim_inf_unilateral95=dbar - q95*se,
                    P_bms_mejor=post.cdf(-DELTA), P_equivalentes=post.cdf(DELTA) - post.cdf(-DELTA), P_bms_peor=post.sf(DELTA),
                    frac_particiones_bms_mejor=float(np.mean(d < 0)))
        if CorrelatedTTest is not None and n % 5 == 0:
            pb = CorrelatedTTest.probs(W[c].values, W[b].values, rope=DELTA, runs=n//5)   # diff = BMS − c
            fila['baycomp_max_dif'] = float(np.max(np.abs(np.array(pb) - [fila['P_bms_mejor'], fila['P_equivalentes'], fila['P_bms_peor']])))
        filas_cmp.append(fila)
cmp = pd.DataFrame(filas_cmp)
cmp['p_holm'] = np.nan
for b in BMS:
    msk = (cmp.bms == b) & cmp.en_familia_holm
    cmp.loc[msk, 'p_holm'] = holm(cmp.loc[msk, 'p_bilateral'].values)
cmp.to_csv(OUT/'comparaciones.csv', index=False)


def decision_h1(fila):
    if fila.lim_sup_unilateral95 < DELTA:
        return 'SE CUMPLE: no inferioridad del BMS frente a EBM (margen 0.03 kN²)'
    if fila.lim_inf_unilateral95 > DELTA:
        return 'NO SE CUMPLE: BMS inferior a EBM (límite inferior > 0.03 kN²)'
    return 'NO CONCLUYENTE'


for b, clave in (('BMS_MDL', 'H1'), ('BMS_promedio', 'H1_secundaria_promedio')):
    r = cmp[(cmp.bms == b) & (cmp.comparador == 'EBM')]
    if len(r):
        r = r.iloc[0]
        inf[clave] = dict(d_media=r.d_media, d_de=r.d_de, ee_corregido=r.ee_corregido, gl=n - 1,
                          lim_sup_unilateral95=r.lim_sup_unilateral95, lim_inf_unilateral95=r.lim_inf_unilateral95,
                          delta=DELTA, decision=decision_h1(r), n_particiones=n)

# ---------------------------------------------------------------- H2, H3a, H3b
ok = tb[~tb.fallo_bms]
inf['H2_descriptiva'] = dict(n_param_mdl=collections.Counter(int(x) for x in ok.n_param),
                             nodos_mdl=dict(mediana=float(ok.nodos.median()), min=int(ok.nodos.min()), max=int(ok.nodos.max())) if len(ok) else None,
                             frac_muestras_k_le_4=float(np.mean(np.array(muestras_k) <= 4)) if muestras_k else None,
                             pysr_complejidad=dict(mediana=float(tpy.complejidad.median()), min=int(tpy.complejidad.min()),
                                                   max=int(tpy.complejidad.max())) if 'complejidad' in tpy else None,
                             pysr_n_constantes=collections.Counter(int(x) for x in tpy.n_constantes.dropna()) if 'n_constantes' in tpy else None)
cumple = (tb.R2_ui >= 0.90) & (tb.p_T >= 0.01) & ~tb.fallo_bms   # fallo o NaN → no cumple
inf['H3b'] = dict(particiones_que_cumplen=int(cumple.sum()), n=n, fraccion=float(cumple.mean()), umbral=0.70,
                  decision='SE CUMPLE (índice único)' if cumple.mean() >= 0.70 else 'NO SE CUMPLE',
                  gamma_mediana=float(ok.gamma.median()), gamma_iqr=[float(ok.gamma.quantile(.25)), float(ok.gamma.quantile(.75))],
                  R2_ui_mediana=float(ok.R2_ui.median()), R2_ui_min=float(ok.R2_ui.min()),
                  p_T_mediana=float(ok.p_T.median()), n_celdas_min=int(ok.n_celdas.min()))
# H3a: MDL del análisis A, reajustado en el entrenamiento de cada partición
cadA = {x.stem: json.loads(x.read_text(encoding='utf-8')) for x in DIR_A.glob('D_cadena*.json') if not x.name.endswith('.progreso.json')}
h3a = []
if cadA:
    A = min(cadA.values(), key=lambda d: d['dl_mdl'])
    fA, flA, usA, thA = funcion_bms(A['modelo_mdl'], A['parametros_mdl']['d0'])
    inf['H3a_referencia'] = dict(modelo=A['modelo_mdl'], canon=A['canon_mdl'], dl=A['dl_mdl'])
    for p in PARTS:
        tr = entrenamiento(p); X = np.vstack([tr.WT.values, tr.WC.values])
        fb = tb.set_index('particion')
        fila = dict(particion=p, misma_estructura_canonica=bool(not fb.loc[p, 'fallo_bms'] and fb.loc[p].get('canon') == A['canon_mdl']))
        try:
            th, _ = curve_fit(lambda Xa, *q: np.asarray(flA(Xa[0], Xa[1], *q), dtype=float)*np.ones(Xa.shape[1]),
                              X, tr['T'].values, p0=thA, maxfev=20000)
            gA = fA(G_WT, G_WC, th)
            if p in mdl_func:
                dif = gA - mdl_func[p](G_WT, G_WC)
                fila.update(rmse_121=float(np.sqrt(np.mean(dif**2))) if np.all(np.isfinite(dif)) else np.nan, ajuste='ok')
            else:
                fila.update(rmse_121=np.nan, ajuste='ok (fallo BMS en la partición)')
        except Exception as e:
            fila.update(rmse_121=np.nan, ajuste=f'fallo: {type(e).__name__}')
        fila['equivalente'] = bool(np.isfinite(fila['rmse_121']) and fila['rmse_121'] < TOL_EQUIV)
        h3a.append(fila)
    th3a = pd.DataFrame(h3a); th3a.to_csv(OUT/'H3a_por_particion.csv', index=False)
    inf['H3a_descriptiva'] = dict(equivalentes=int(th3a.equivalente.sum()), misma_estructura_canonica=int(th3a.misma_estructura_canonica.sum()),
                                  n=n, ajustes_fallidos=int((~th3a.ajuste.str.startswith('ok')).sum()),
                                  rmse_121_mediana=float(th3a.rmse_121.median()))
# estabilidad estructural entre particiones
inf['estructuras_mdl_cv'] = collections.Counter(ok.canon).most_common()
inf['diagnosticos_bms'] = dict(
    particiones_fallidas=int(tb.fallo_bms.sum()), cadenas=len(tcad),
    cadenas_relanzadas=int(tcad.relanzada.sum()) if len(tcad) else 0,
    particiones_con_cadenas_concordantes=int((ok.cadenas_misma_estructura >= 2).sum()),
    rhat_dl_mediana=float(ok.rhat_dl.median()), ess_dl_mediana=float(ok.ess_dl.median()),
    aceptacion_swaps_media=float(tcad.aceptacion_swaps.mean()) if len(tcad) else None,
    timeouts_total=int(tcad.timeouts.sum()) if len(tcad) else 0, complejas_total=int(tcad.complejas.sum()) if len(tcad) else 0,
    excepciones_total=int(tcad.excepciones.sum()) if len(tcad) else 0,
    max_dif_lambdify=float(ok.max_dif_lambdify.max()) if len(ok) else None,
    minutos_por_cadena=dict(mediana=float(tcad.minutos.median()), max=float(tcad.minutos.max())) if len(tcad) else None)

# ---------------------------------------------------------------- informe
(OUT/'C-005_informe.json').write_text(json.dumps(inf, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o),
                                                 ensure_ascii=False), encoding='utf-8')
cols_cmp = ['bms', 'comparador', 'mse_bms', 'mse_comp', 'd_media', 'ee_corregido', 'lim_sup_unilateral95', 'p_bilateral', 'p_holm',
            'P_bms_mejor', 'P_equivalentes', 'P_bms_peor']
md = ['# C-005 — Análisis estadístico de la CV (informe automático)', '',
      f'Protocolo SHA {PROTOCOLO_SHA[:12]}…; particiones: {n}; prueba sintética: {PRUEBA}', '',
      '## H1 (BMS-MDL frente a EBM, δ = 0.03 kN²)', '', f"**{inf.get('H1', {}).get('decision')}**", '',
      '```json', json.dumps({k: inf.get(k) for k in ('H1', 'H1_secundaria_promedio')}, indent=1, default=float), '```', '',
      '## Resumen por modelo (media de 25 particiones; MSE en kN²)', '', a_md(res), '',
      '## Comparaciones (d = MSE_BMS − MSE_comparador; Holm en la familia de 8)', '', a_md(cmp[cols_cmp]), '',
      '## H3b (índice único)', '', '```json', json.dumps(inf['H3b'], indent=1, default=float), '```', '',
      '## BMS por partición', '',
      a_md(tb[[c for c in ['particion', 'n_cadenas_completas', 'cadena_mdl', 'dl_mdl', 'n_param', 'nodos', 'cadenas_misma_estructura',
                           'rhat_dl', 'gamma', 'R2_ui', 'p_T', 'no_finitos_mdl', 'canon'] if c in tb]]), '',
      '## PySR por partición', '', a_md(tpy[[c for c in ['particion', 'complejidad', 'n_constantes', 'ecuacion'] if c in tpy]]), '',
      '## Resto del informe', '', '```json',
      json.dumps({k: v for k, v in inf.items() if k not in ('H1', 'H1_secundaria_promedio', 'H3b')}, indent=1, default=str,
                 ensure_ascii=False), '```']
(OUT/'C-005_informe.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md[:12]))
print(a_md(res[['modelo', 'mse', 'mse_de', 'rmse', 'r2', 'cpu_h_total']]))
print(a_md(cmp[cmp.bms == 'BMS_MDL'][['comparador', 'd_media', 'lim_sup_unilateral95', 'p_holm', 'P_bms_mejor', 'P_equivalentes', 'P_bms_peor']]))
print(json.dumps({k: inf.get(k) for k in ('H3b', 'H3a_descriptiva', 'H2_descriptiva', 'diagnosticos_bms', 'avisos',
                                          'check_y_max_dif', 'check_mse_c004_max_dif', 'check_mse_c002_max_dif',
                                          'check_media_entrenamiento_max_dif', 'comparadores_fallidos')},
                 indent=1, default=str, ensure_ascii=False))
