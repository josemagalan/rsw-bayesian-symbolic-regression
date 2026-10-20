"""E-015 (EXPLORATORIO, post hoc; 2026-10-19): apuntalar el hallazgo A-024 (BMS y PySR convergen en (a − b/WT)·ln WC)
para la Discusión. Todo es posterior a ver los datos: descriptivo, sin contrastes con error controlado.
  1. Inventario de TODAS las ecuaciones simbólicas disponibles: PySR (C-002, 25; C-008, 77; E-014, 1) y MDL del BMS
     (C-003, 25; C-001, 8 de D; C-007, 17). Para cada una, en la rejilla nominal 11 × 11 (WT 0.04–0.24 s, WC 1.5–6.5 kA):
     - motivos sintácticos (sympy): logaritmo con WC dentro; forma en que entra WT (WT⁻¹, potencia 0–1, log, tanh/exp);
     - forma funcional, independiente de la sintaxis: R² de perfiles a + b·ln WC (frente a lineal y √WC) por fila de WT y
       de a + b/WT (frente a ln WT y lineal) por columna de WC;
     - distancia a la familia F1 = (a − b/WT)·ln WC: RMSE de la mejor aproximación de su superficie por F1 (proyección,
       sin datos) y los (a, b) de esa proyección.
     Referencias para esa distancia: las superficies de GAM y EBM ajustados en los mismos 25 entrenamientos de la CV, y las
     medias por celda de los datos.
  2. CV (25 particiones agrupadas, las de C-005) de formas cerradas de 2–4 parámetros: F1 y variantes y controles sin
     logaritmo o sin hipérbola. Formas sugeridas por los datos: error optimista.
  3. Estabilidad de (a, b) de F1: reajuste en las 25 particiones y bootstrap por celdas con todos los datos (1000);
     t₀ = b/a (tiempo en que F1 se anula).
  4. Triangulación con modelos no simbólicos: dependencia parcial (PDP) en WC y en WT de GAM, EBM, RF, GBM y TabPFN
     ajustados con todos los datos; R² de las formas ln WC y 1/WT frente a alternativas de igual número de parámetros.
Uso: python analisis/exploratorio/E-015_convergencia_formas.py [--sin-tabpfn] [--rapido]  (--rapido: 5 particiones; pruebas)
Salida: resultados/2026-10-19_E-015/ (o _pruebas/E-015_prueba con --rapido)."""
import importlib.util, json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, sympy as sp
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
from bms_comun import cargar_datos, ROOT
warnings.filterwarnings('ignore')

RAPIDO = '--rapido' in sys.argv; SIN_TABPFN = '--sin-tabpfn' in sys.argv
R = ROOT/'resultados'
OUT = R/('_pruebas/E-015_prueba' if RAPIDO else '2026-10-19_E-015'); OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('c008', ROOT/'analisis'/'confirmatorio'/'C-008_sensibilidad_comparadores.py')
c008 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c008)
df0 = cargar_datos(); id_cero = int(df0.loc[df0['T'] == 0, 'id'].iloc[0]); df0['WC_nom'] = 1.5 + 0.5*(df0['nivel'] - 1)
NP = 5 if RAPIDO else 25
GWT = np.arange(2, 13)*0.02; GWC = np.linspace(1.5, 6.5, 11); GT, GC = np.meshgrid(GWT, GWC, indexing='ij')
WT_s, WC_s = sp.symbols('WT WC')
F1 = lambda X, a, b: (a - b/X[0])*np.log(X[1])
t0 = time.time()


def evaluar_rejilla(ex):
    f = sp.lambdify([WT_s, WC_s], ex, 'numpy')
    with np.errstate(all='ignore'):
        z = np.asarray(f(GT, GC), dtype=complex)*np.ones(GT.shape)
    return np.where(np.abs(z.imag) < 1e-9, z.real, np.nan)


def r2_perfiles(Z, eje, transf):
    """R² medio de ajustar z = a + b·g(x) a cada perfil (filas: WT fijo, variando WC; o columnas)."""
    x = GWC if eje == 'WC' else GWT; g = transf(x); r2 = []
    perfiles = Z if eje == 'WC' else Z.T
    for z in perfiles:
        if not np.all(np.isfinite(z)) or np.var(z) < 1e-12:
            continue
        A = np.column_stack([np.ones_like(g), g]); b, *_ = np.linalg.lstsq(A, z, rcond=None)
        r2.append(1 - np.sum((z - A@b)**2)/np.sum((z - z.mean())**2))
    return float(np.mean(r2)) if r2 else np.nan


def proyeccion_F1(Z):
    if not np.all(np.isfinite(Z)):
        return dict(proj_rmse=np.nan, proj_a=np.nan, proj_b=np.nan)
    try:
        th, _ = curve_fit(F1, np.vstack([GT.ravel(), GC.ravel()]), Z.ravel(), p0=[4.0, 0.05], maxfev=20000)
        return dict(proj_rmse=float(np.sqrt(np.mean((F1(np.vstack([GT.ravel(), GC.ravel()]), *th) - Z.ravel())**2))),
                    proj_a=float(th[0]), proj_b=float(th[1]))
    except Exception:
        return dict(proj_rmse=np.nan, proj_a=np.nan, proj_b=np.nan)


def sintaxis(ex):
    logs = [a for a in ex.atoms(sp.log)]
    log_wc = any(WC_s in a.free_symbols for a in logs)
    pots = [p for p in ex.atoms(sp.Pow) if WT_s in p.base.free_symbols]
    wt_inv = any(p.exp.is_number and p.exp < 0 for p in pots)
    wt_pot = any(p.exp.is_number and 0 < p.exp < 1 for p in pots)
    wt_log = any(WT_s in a.free_symbols for a in logs)
    wt_sat = any(WT_s in a.free_symbols for a in list(ex.atoms(sp.tanh)) + list(ex.atoms(sp.exp)))
    return dict(log_WC=log_wc, WT_inversa=wt_inv, WT_potencia_0_1=wt_pot, WT_log=wt_log, WT_tanh_exp=wt_sat)


def describir(ex):
    Z = evaluar_rejilla(ex)
    d = dict(r2_WC_log=r2_perfiles(Z, 'WC', np.log), r2_WC_lineal=r2_perfiles(Z, 'WC', lambda x: x),
             r2_WC_raiz=r2_perfiles(Z, 'WC', np.sqrt), r2_WT_inversa=r2_perfiles(Z, 'WT', lambda x: 1/x),
             r2_WT_log=r2_perfiles(Z, 'WT', np.log), r2_WT_lineal=r2_perfiles(Z, 'WT', lambda x: x),
             no_finita=bool(~np.all(np.isfinite(Z))))
    d.update(proyeccion_F1(Z)); d.update(sintaxis(ex))
    return d


def bms_expr(d):
    ex = sp.sympify(d['modelo_mdl'])
    return ex.subs({sp.Symbol(k): v for k, v in d['parametros_mdl']['d0'].items()})


def min_dl(ficheros):
    cs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in ficheros]
    return min(cs, key=lambda d: d['dl_mdl']) if len(cs) >= 2 else None


# ---------------------------------------------------------------- 1. inventario y descriptores
filas = []
for f in sorted((R/'C-002_pysr_cv').glob('p??.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    filas.append(dict(metodo='PySR', fuente='C-002 (CV)', variante='principal', particion=d['p'], expr=d['ecuacion']))
for v in ['aleatoria', 'wc_nominal', 'sin_cero', 'extrapolacion']:
    for f in sorted((R/'C-008_sensibilidad_comparadores'/v/'pysr').glob('p??.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        filas.append(dict(metodo='PySR', fuente='C-008 (sensibilidad)', variante=v, particion=d['p'], expr=d['ecuacion']))
f14 = R/'2026-10-19_E-014'/'pysr'/'p99.json'
if f14.exists():
    filas.append(dict(metodo='PySR', fuente='E-014 (todos los datos)', variante='todos', particion=-1,
                      expr=json.loads(f14.read_text(encoding='utf-8'))['ecuacion']))
for p in range(25):
    d = min_dl(sorted((R/'C-003_bms_cv').glob(f'p{p:02d}_cadena?.json')))
    if d:
        filas.append(dict(metodo='BMS', fuente='C-003 (CV)', variante='principal', particion=p, expr=str(bms_expr(d))))
for f in sorted((R/'C-001_analisis_A').glob('D_cadena?.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    filas.append(dict(metodo='BMS', fuente='C-001 (todos los datos)', variante=f.stem, particion=-1, expr=str(bms_expr(d))))
for v in ['aleatoria', 'wc_nominal', 'np4', 'extrapolacion']:
    pref = ['WT_extremos', 'WC_extremos'] if v == 'extrapolacion' else [f'p{p:02d}' for p in range(5)]
    for i, pr in enumerate(pref):
        d = min_dl(sorted((R/'C-007_bms_sensibilidad'/v).glob(f'{pr}_cadena?.json')))
        if d:
            filas.append(dict(metodo='BMS', fuente='C-007 (sensibilidad)', variante=v, particion=i, expr=str(bms_expr(d))))
for fila in filas:
    try:
        fila.update(describir(sp.sympify(fila['expr'], locals={'WT': WT_s, 'WC': WC_s})))
    except Exception as e:
        fila['error'] = repr(e)[:100]
EQ = pd.DataFrame(filas); EQ.to_csv(OUT/'ecuaciones.csv', index=False)
print(f'{len(EQ)} ecuaciones descritas ({time.time() - t0:.0f} s)', flush=True)

# referencias para la distancia a F1: GAM y EBM en los entrenamientos de la CV; medias por celda de los datos
modelos, _ = c008.cargar_modelos_c004(); M = dict(modelos)
folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
Xg = lambda ef: np.column_stack([GT.ravel(), GC.ravel(), np.full(GT.size, ef)])
ref = []
for p in range(NP):
    ids = set(folds.query('repeticion == @p // 5 and particion == @p % 5').id); tr = df0[~df0.id.isin(ids)]
    cv = list(GroupKFold(5, shuffle=True, random_state=20261016 + p).split(tr, tr['T'], tr['celda']))
    for nombre in ['GAM', 'EBM']:
        q, _ = M[nombre](tr[['WT', 'WC', 'EF']].values, tr['T'].values, np.vstack([Xg(1.0), Xg(1.5)]), cv, 20261016 + p, tr)
        q = np.asarray(q, float); Z = (0.5*(q[:GT.size] + q[GT.size:])).reshape(GT.shape)
        ref.append(dict(metodo=nombre, particion=p, **proyeccion_F1(Z), r2_WC_log=r2_perfiles(Z, 'WC', np.log),
                        r2_WC_lineal=r2_perfiles(Z, 'WC', lambda x: x), r2_WT_inversa=r2_perfiles(Z, 'WT', lambda x: 1/x),
                        r2_WT_log=r2_perfiles(Z, 'WT', np.log)))
cel = df0.groupby(['cyc', 'nivel'])['T'].mean().unstack().values           # 11 × 11: filas WT, columnas nivel de WC
ref.append(dict(metodo='Datos (medias por celda)', particion=-1, **proyeccion_F1(cel), r2_WC_log=r2_perfiles(cel, 'WC', np.log),
                r2_WC_lineal=r2_perfiles(cel, 'WC', lambda x: x), r2_WT_inversa=r2_perfiles(cel, 'WT', lambda x: 1/x),
                r2_WT_log=r2_perfiles(cel, 'WT', np.log)))
REF = pd.DataFrame(ref); REF.to_csv(OUT/'referencias_distancia.csv', index=False)
print(f'Referencias ({time.time() - t0:.0f} s)', flush=True)

# resumen de motivos por método y fuente
def resumen(g):
    return pd.Series(dict(n=len(g), log_WC=g.log_WC.mean(), WT_inversa=g.WT_inversa.mean(),
                          WT_saturante_cualquiera=(g.WT_inversa | g.WT_potencia_0_1 | g.WT_log | g.WT_tanh_exp).mean(),
                          r2_WC_log_med=g.r2_WC_log.median(), r2_WC_lineal_med=g.r2_WC_lineal.median(),
                          r2_WT_inversa_med=g.r2_WT_inversa.median(), r2_WT_log_med=g.r2_WT_log.median(),
                          proj_rmse_med=g.proj_rmse.median(), proj_rmse_p90=g.proj_rmse.quantile(0.9),
                          frac_proj_menor_0p1=(g.proj_rmse < 0.1).mean(), frac_proj_menor_0p2=(g.proj_rmse < 0.2).mean()))
RES = EQ.groupby(['metodo', 'fuente']).apply(resumen).reset_index(); RES.to_csv(OUT/'resumen_motivos.csv', index=False)

# ---------------------------------------------------------------- 2. CV de formas cerradas
FORMAS = {
    'F1 (a−b/WT)·lnWC': (lambda X, a, b: (a - b/X[0])*np.log(X[1]), [4.0, 0.05]),
    'F2 (a−b/WT)·ln(WC+WT)': (lambda X, a, b: (a - b/X[0])*np.log(X[1] + X[0]), [3.9, 0.047]),
    'F3 a·WT^b·lnWC': (lambda X, a, b: a*X[0]**b*np.log(X[1]), [5.0, 0.17]),
    'F4 a·tanh(b·WT)·lnWC': (lambda X, a, b: a*np.tanh(b*X[0])*np.log(X[1]), [3.7, 21.0]),
    'F5 a·lnWC + b/WT + c': (lambda X, a, b, c: a*np.log(X[1]) + b/X[0] + c, [3.0, -0.05, 0.5]),
    'F6 a·lnWC + b·lnWT + c': (lambda X, a, b, c: a*np.log(X[1]) + b*np.log(X[0]) + c, [3.3, 1.0, 2.4]),
    'C1 (a−b/WT)·√WC (sin log)': (lambda X, a, b: (a - b/X[0])*np.sqrt(X[1]), [3.0, 0.03]),
    'C2 (a+b·WT)·lnWC (sin hipérbola)': (lambda X, a, b: (a + b*X[0])*np.log(X[1]), [2.5, 5.0]),
    'C3 a·WC + b/WT + c (lineal en WC)': (lambda X, a, b, c: a*X[1] + b/X[0] + c, [1.0, -0.05, 1.0]),
    'C4 a·(1−exp(−b·WC^c·WT^d)) (E-001)': (lambda X, a, b, c, d: a*(1 - np.exp(-b*X[1]**c*X[0]**d)), [7.3, 0.34, 1.38, 0.35]),
}
filas, par = [], []
for p in range(NP):
    ids = set(folds.query('repeticion == @p // 5 and particion == @p % 5').id)
    te, tr = df0[df0.id.isin(ids)], df0[~df0.id.isin(ids)]
    for nom, (f, p0) in FORMAS.items():
        try:
            th, _ = curve_fit(f, np.vstack([tr.WT, tr.WC]), tr['T'].values, p0=p0, maxfev=50000)
            mse = float(np.mean((te['T'].values - f(np.vstack([te.WT, te.WC]), *th))**2))
        except Exception:
            th, mse = [np.nan]*len(p0), np.nan
        filas.append(dict(forma=nom, particion=p, mse=mse, k=len(p0)))
        if nom.startswith('F1'):
            par.append(dict(particion=p, a=th[0], b=th[1], t0=th[1]/th[0]))
CVF = pd.DataFrame(filas); CVF.to_csv(OUT/'formas_cv_por_particion.csv', index=False)
W = CVF.pivot(index='particion', columns='forma', values='mse'); base = W['F1 (a−b/WT)·lnWC']
tab = []
for nom in FORMAS:
    d = W[nom] - base; n = len(d); se = np.sqrt((1/n + 0.25)*d.var(ddof=1)) if n > 1 else np.nan
    tab.append(dict(forma=nom, k=len(FORMAS[nom][1]), mse=W[nom].mean(), mse_de=W[nom].std(), d_vs_F1=d.mean(),
                    ic95_inf=d.mean() - stats.t.ppf(0.975, n - 1)*se if n > 1 else np.nan,
                    ic95_sup=d.mean() + stats.t.ppf(0.975, n - 1)*se if n > 1 else np.nan, frac_mejor_que_F1=float((d < 0).mean())))
CVT = pd.DataFrame(tab).sort_values('mse'); CVT.to_csv(OUT/'formas_cv_resumen.csv', index=False)
print(f'CV de formas ({time.time() - t0:.0f} s)', flush=True)

# ---------------------------------------------------------------- 3. estabilidad de (a, b)
PAR = pd.DataFrame(par); PAR.to_csv(OUT/'F1_parametros_por_particion.csv', index=False)
X = np.vstack([df0.WT, df0.WC]); y = df0['T'].values
th, _ = curve_fit(F1, X, y, p0=[4.0, 0.05]); res = y - F1(X, *th)
rng = np.random.default_rng(20261019); celdas = df0['celda'].values; uc = np.unique(celdas); boot = []
for _ in range(200 if RAPIDO else 1000):
    cs = rng.choice(uc, len(uc)); idx = np.concatenate([np.flatnonzero(celdas == c) for c in cs])
    try:
        boot.append(curve_fit(F1, X[:, idx], y[idx], p0=th)[0])
    except Exception:
        pass
boot = np.array(boot); t0s = boot[:, 1]/boot[:, 0]
PARG = dict(a=float(th[0]), b=float(th[1]), t0=float(th[1]/th[0]), mse_en_muestra=float(np.mean(res**2)),
            ic95_a=np.percentile(boot[:, 0], [2.5, 97.5]).tolist(), ic95_b=np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
            ic95_t0=np.percentile(t0s, [2.5, 97.5]).tolist(), n_boot=len(boot))

# ---------------------------------------------------------------- 4. PDP de modelos no simbólicos (todos los datos)
cvall = list(GroupKFold(5, shuffle=True, random_state=20261016).split(df0, y, df0['celda']))
Xall = df0[['WT', 'WC', 'EF']].values; pdp = []
nombres = ['GAM', 'EBM', 'RF', 'GBM'] + ([] if SIN_TABPFN else ['TabPFN_v2'])
for nombre in nombres:
    for var, rej, col in [('WC', GWC, 1), ('WT', GWT, 0)]:
        Xq = np.vstack([np.where(np.arange(3) == col, v, Xall) for v in rej])      # cada valor de rejilla × 242 filas
        q, _ = M[nombre](Xall, y, Xq, cvall, 20261016, df0)
        curva = np.asarray(q, float).reshape(len(rej), len(Xall)).mean(1)
        for v, c in zip(rej, curva):
            pdp.append(dict(metodo=nombre, variable=var, x=v, pdp=c))
PDP = pd.DataFrame(pdp); PDP.to_csv(OUT/'pdp.csv', index=False)
TRANSF = {'WC': {'ln': np.log, 'lineal': lambda x: x, 'raiz': np.sqrt}, 'WT': {'inversa': lambda x: 1/x, 'ln': np.log,
                                                                                 'lineal': lambda x: x, 'raiz': np.sqrt}}
r2p = []
for (m, v), g in PDP.groupby(['metodo', 'variable']):
    z = g.pdp.values; x = g.x.values
    fila = dict(metodo=m, variable=v)
    for nt, tf in TRANSF[v].items():
        A = np.column_stack([np.ones_like(x), tf(x)]); b, *_ = np.linalg.lstsq(A, z, rcond=None)
        fila[f'r2_{nt}'] = 1 - np.sum((z - A@b)**2)/np.sum((z - z.mean())**2)
    r2p.append(fila)
R2P = pd.DataFrame(r2p); R2P.to_csv(OUT/'pdp_r2.csv', index=False)
print(f'PDP ({time.time() - t0:.0f} s)', flush=True)

# ---------------------------------------------------------------- informe
def a_md(t, dec=3):
    fmt = lambda v: f'{v:.{dec}f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, t.columns)) + ' |', '|' + '---|'*len(t.columns)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in t.itertuples(index=False)])


refr = REF.groupby('metodo').agg(proj_rmse_med=('proj_rmse', 'median'), proj_rmse_min=('proj_rmse', 'min'),
                                 proj_rmse_max=('proj_rmse', 'max'), r2_WC_log=('r2_WC_log', 'median'),
                                 r2_WC_lineal=('r2_WC_lineal', 'median'), r2_WT_inversa=('r2_WT_inversa', 'median'),
                                 r2_WT_log=('r2_WT_log', 'median')).reset_index()
exactas = EQ[(EQ.metodo == 'PySR') & (EQ.proj_rmse < 0.02)][['fuente', 'variante', 'particion', 'expr', 'proj_a', 'proj_b']]
md = [f"# E-015 — Convergencia de formas entre BMS y PySR (exploratorio){' — PRUEBA RÁPIDA' if RAPIDO else ''}", '',
      'Rejilla nominal 11 × 11. F1 = (a − b/WT)·ln WC. "proj_rmse": RMSE (kN) de la mejor aproximación de cada superficie por F1. '
      'Ruido de referencia σ ≈ 0.26 kN (C-006).', '',
      '## 1. Motivos y forma funcional de todas las ecuaciones', '', a_md(RES), '',
      '### Referencias (GAM y EBM en los entrenamientos de la CV; medias por celda)', '', a_md(refr), '',
      '### Ecuaciones de PySR que son F1 a efectos prácticos (proj_rmse < 0.02 kN)', '', a_md(exactas), '',
      '## 2. CV de formas cerradas (25 particiones agrupadas; d = forma − F1, IC con la corrección de Nadeau y Bengio)', '',
      a_md(CVT), '',
      '## 3. Estabilidad de F1', '', f'Todos los datos: {json.dumps(PARG)}', '',
      f"Por partición: a {PAR.a.median():.3f} [{PAR.a.min():.3f}, {PAR.a.max():.3f}]; b {PAR.b.median():.4f} "
      f"[{PAR.b.min():.4f}, {PAR.b.max():.4f}]; t₀ {PAR.t0.median()*1000:.1f} ms [{PAR.t0.min()*1000:.1f}, {PAR.t0.max()*1000:.1f}].", '',
      '## 4. Dependencia parcial de modelos no simbólicos (todos los datos): R² de cada forma 1D', '', a_md(R2P), '',
      f'Tiempo total: {time.time() - t0:.0f} s.']
(OUT/'E-015_informe.md').write_text('\n'.join(md), encoding='utf-8')
(OUT/'entorno.json').write_text(json.dumps(dict(rapido=RAPIDO, sin_tabpfn=SIN_TABPFN, F1_todos=PARG)), encoding='utf-8')
print('\n'.join(md))
