"""E-006 (EXPLORATORIO, piloto de F2). Continuación de E-005 con datos SINTÉTICOS.

Preguntas (a raíz de E-005):
  1. Opción D de D-006: 2 variables (WT, WC) en UN solo conjunto de datos (EF fuera del modelo),
     prior nv2.np6. ¿Se vuelven identificables las verdades de 4-5 parámetros (T1, T2) al no
     duplicar parámetros por nivel de EF, como ocurría en la opción B?
  2. Verdad T4 con efecto REAL de EF (amplitud 6.8 kN a 1000 N y 7.3 kN a 1500 N): ¿lo detectan
     A (EF como variable) y B (EF como 2 conjuntos), y cuánto pierde D (que lo ignora)?
     Sirve para validar el contraste de H4 antes de aplicarlo a datos reales.
Mismo criterio que E-005 (DL del MDL vs DL de la verdad reajustada; RMSE funcional).
Uso (desde la raíz): python analisis/exploratorio/E-006_recuperacion_sintetica_D_EF.py [pasos=20000] [procesos=auto]
       [--prueba] [--solo=nombre1,nombre2] [--max-horas=H]
"""
import sys, os, json, time, hashlib, platform, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'analisis' / 'vendor' / 'machine-scientist'
PRIORS = ROOT / 'analisis' / 'priors'
DATA = ROOT / 'Dataset' / 'datos_corregido_junio16_2_puntos20-69-88.xlsx'
SHA = 'ae881203f8f6027daf1baa3bccc56aa0d9d528b8d6cd2aaa0a87af47fc02e99b'
OUT = ROOT / 'resultados' / 'E-006_recuperacion_sintetica_D_EF'

# ---------------- Verdades (formato de cadena del BMS oficial) ----------------
J = '(_a0_ + -((_a0_ * exp(-((_a1_ * (pow2(WC) * WT)))))))'
P = '(_a0_ + -((_a0_ * exp(-((_a1_ * ((WC ** _a2_) * (WT ** _a3_))))))))'
X2 = '((_a0_ + -((_a0_ * exp(-((_a1_ * ((WC ** _a2_) * (WT ** _a3_)))))))) + -((_a4_ * pow2((pow2(WC) * WT)))))'
PJ = {'_a0_': 7.0, '_a1_': 0.6}
PP = {'_a0_': 7.3, '_a1_': 0.342, '_a2_': 1.38, '_a3_': 0.345}
PX = dict(PP, _a4_=0.005)
# Para cada verdad: f(WT, WC, EF) y, por opción, (expresión, parámetros) o None si no es representable.
# En B los parámetros son por conjunto de datos: {'EF1000': {...}, 'EF1500': {...}}.
TRUTHS = {
    'T3_joule': dict(f=lambda WT, WC, EF: 7.0*(1-np.exp(-0.6*WC**2*WT)),
                     D=(J, PJ), B=(J, {'EF1000': PJ, 'EF1500': PJ})),
    'T1_potencias': dict(f=lambda WT, WC, EF: 7.3*(1-np.exp(-0.342*WC**1.38*WT**0.345)),
                         D=(P, PP), B=(P, {'EF1000': PP, 'EF1500': PP})),
    'T2_expulsion': dict(f=lambda WT, WC, EF: 7.3*(1-np.exp(-0.342*WC**1.38*WT**0.345)) - 0.005*(WC**2*WT)**2,
                         D=(X2, PX), B=(X2, {'EF1000': PX, 'EF1500': PX})),
    # Efecto de EF en la amplitud: a(EF) = 5.8 + 1.0*EF[kN] -> 6.8 (1000 N), 7.3 (1500 N)
    'T4_ef': dict(f=lambda WT, WC, EF: (5.8 + 1.0*EF)*(1-np.exp(-0.6*WC**2*WT)),
                  A=('(((_a0_ + (_a1_ * EF))) + -(((_a0_ + (_a1_ * EF)) * exp(-((_a2_ * (pow2(WC) * WT)))))))',
                     {'_a0_': 5.8, '_a1_': 1.0, '_a2_': 0.6}),
                  B=(J, {'EF1000': {'_a0_': 6.8, '_a1_': 0.6}, 'EF1500': {'_a0_': 7.3, '_a1_': 0.6}}),
                  D=None),
}
# Configuraciones: (verdad, opción)
PLAN = [('T3_joule', 'D'), ('T1_potencias', 'D'), ('T2_expulsion', 'D'),
        ('T4_ef', 'A'), ('T4_ef', 'B'), ('T4_ef', 'D')]
OPTIONS = {
    'A': dict(variables=['WT', 'WC', 'EF'], npar=3, prior='nv3.np3', multi=False),
    'B': dict(variables=['WT', 'WC'], npar=6, prior='nv2.np6', multi=True),   # EF = 2 conjuntos de datos
    'D': dict(variables=['WT', 'WC'], npar=6, prior='nv2.np6', multi=False),  # EF fuera del modelo
}
SIGMAS = [0.10, 0.28]
SEEDS = [1, 2, 3]
VERSION = 1
TS = [1] + [1.04**k for k in range(1, 20)]   # temperaturas por defecto del tutorial oficial


def load_design():
    # Se lee UNA vez en el proceso principal y se pasa a los procesos hijos: en Windows, muchos
    # procesos abriendo el .xlsx a la vez provocaron PermissionError (2026-10-15).
    for intento in range(5):
        try:
            raw = DATA.read_bytes()
            break
        except PermissionError:
            time.sleep(2 + 3*intento)
    else:
        raise PermissionError(f'No se puede leer {DATA}')
    assert hashlib.sha256(raw).hexdigest() == SHA, 'El fichero de datos ha cambiado'
    import io
    df = pd.read_excel(io.BytesIO(raw), header=0).iloc[:, 2:9]   # el contenido cuyo hash se ha verificado
    df.columns = ['id', 'cyc', 'WT', 'WC', 'EF', 'T', 'Q']
    df = df.dropna(subset=['id'])
    X = df[['WT', 'WC', 'EF']].astype(float).reset_index(drop=True)
    X['EF'] = X['EF'] / 1000.0        # kN
    return X                          # la columna T (TSLBC real) NO se usa


def run_one(cfg):
    """Ejecuta una configuración. Función de nivel superior (necesario para multiprocessing en Windows)."""
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(VENDOR)); sys.path.insert(0, str(VENDOR / 'Prior'))
    import mcmc                                   # noqa
    sys.path.insert(0, str(ROOT / 'analisis'))
    import bms_parches; bms_parches.aplicar()     # D-007: expresiones complejas -> inválidas
    from mcmc import Tree
    from parallel import Parallel
    from fit_prior import read_prior_par
    import random as pyrandom

    name = f"{cfg['truth']}_{cfg['option']}_s{cfg['sigma']:.2f}_seed{cfg['seed']}"
    out = Path(cfg['out'])          # se pasa en cfg: en Windows los procesos hijos reimportan el módulo
    final = out / f'{name}.json'
    if final.exists():
        try:
            if json.loads(final.read_text(encoding='utf-8')).get('version_script') == VERSION:
                return name, 'ya existía'
        except Exception:
            pass
    try:
        return _run_one(cfg, name, out, final)
    except BaseException as e:                    # una configuración fallida no detiene las demás
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        import traceback
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{name}_ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return name, f'ERROR: {e!r}'


def _run_one(cfg, name, out, final):
    out.mkdir(parents=True, exist_ok=True)
    import mcmc                                   # noqa
    from mcmc import Tree
    from parallel import Parallel
    from fit_prior import read_prior_par
    import random as pyrandom
    import bms_parches
    for k in bms_parches.CONTADOR:                # contador por configuración (el Pool reutiliza procesos)
        bms_parches.CONTADOR[k] = 0
    truth, opt = TRUTHS[cfg['truth']], OPTIONS[cfg['option']]
    X = pd.DataFrame(cfg['X'])     # diseño leído en el proceso principal
    rng = np.random.default_rng(1000 + cfg['seed'])
    f_true = truth['f'](X['WT'].values, X['WC'].values, X['EF'].values)
    y = f_true + rng.normal(0, cfg['sigma'], len(X))
    pyrandom.seed(cfg['seed']); np.random.seed(cfg['seed'])

    prior = read_prior_par(str(next(PRIORS.glob(f"*{opt['prior']}.*.dat"))))
    params = ['a%d' % i for i in range(opt['npar'])]
    if opt['multi']:
        keys = {1.0: 'EF1000', 1.5: 'EF1500'}
        x = {k: X.loc[X.EF == ef, opt['variables']].reset_index(drop=True) for ef, k in keys.items()}
        ys = {k: pd.Series(y[(X.EF == ef).values]) for ef, k in keys.items()}
    else:
        x, ys = X[opt['variables']], pd.Series(y)

    # DL de la verdad (si cabe en el presupuesto de parámetros)
    dl_truth = None
    spec = truth.get(cfg['option'])
    if spec is not None:
        expr, par = spec
        t = Tree(variables=opt['variables'], parameters=params, x=x, y=ys, prior_par=prior, from_string=expr)
        full = lambda d: {p: d.get(p, 1.0) for p in ['_%s_' % q for q in params]}
        t.set_par_values({k: full(par[k]) for k in x} if opt['multi'] else full(par))
        dl_true_par = float(t.E)
        t.fit_par = {}; t.get_sse(fit=True); t.get_bic(reset=True); t.get_energy(reset=True)
        dl_truth = min(dl_true_par, float(t.E))

    pms = Parallel(TS, variables=opt['variables'], parameters=params, x=x, y=ys, prior_par=prior)
    t0, best, best_tree, best_step, trace = time.time(), np.inf, None, 0, []
    for i in range(cfg['steps']):
        pms.mcmc_step(); pms.tree_swap()
        E = float(pms.t1.E)
        if np.isnan(E) or E == -np.inf:          # estado inconsistente del MCMC: no se oculta
            raise RuntimeError(f'Energía no válida ({E}) en el paso {i}; modelo {pms.t1}')
        if i % 10 == 0:
            trace.append(E)
        if E < best:
            best, best_step = E, i
            best_tree = Tree(variables=opt['variables'], parameters=params, x=x, y=ys,
                             prior_par=prior, from_string=str(pms.t1))
            best_tree.set_par_values(pms.t1.par_values)
        if i % 500 == 0:                          # progreso (diagnóstico de cadenas lentas o colgadas)
            (out / f'{name}.progreso.json').write_text(json.dumps(dict(
                paso=i, minutos=round((time.time()-t0)/60, 1), dl_mdl=best, paso_mdl=best_step,
                modelo_actual_T1=str(pms.t1), modelo_mdl=str(best_tree))), encoding='utf-8')
    dt = time.time() - t0

    # Distancia funcional en los puntos del diseño (frente a la verdad sin ruido)
    if opt['multi']:
        pred = np.empty(len(X))
        pdict = best_tree.predict(x)                     # dict por conjunto de datos
        for ef, k in {1.0: 'EF1000', 1.5: 'EF1500'}.items():
            pred[(X.EF == ef).values] = np.asarray(pdict[k], dtype=float)
    else:
        pred = np.asarray(best_tree.predict(x), dtype=float)
    rmse_func = float(np.sqrt(np.mean((pred - f_true)**2)))

    res = dict(cfg={k: v for k, v in cfg.items() if k != 'X'}, nombre=name, dl_mdl=best, dl_verdad=dl_truth,
               delta_dl=None if dl_truth is None else best - dl_truth,
               modelo_mdl=str(best_tree), parametros_mdl={k: {p: float(v) for p, v in d.items()}
                                                          for k, d in best_tree.par_values.items()},
               paso_mdl=best_step, rmse_funcional=rmse_func, sigma=cfg['sigma'],
               tiempo_s=dt, pasos_por_s=cfg['steps']/dt, traza_cada10=trace,
               version_script=VERSION, parches=dict(bms_parches.CONTADOR),
               plataforma=platform.platform(), python=platform.python_version())
    final.write_text(json.dumps(res, indent=1), encoding='utf-8')
    return name, f"DL_MDL={best:.2f} DL_verdad={dl_truth} rmse_f={rmse_func:.3f} {dt/60:.1f} min"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    steps = int(args[0]) if args else 20000
    nproc = int(args[1]) if len(args) > 1 else max(1, (os.cpu_count() or 2) - 1)
    configs = [dict(truth=t, option=o, sigma=sg, seed=sd, steps=steps)
               for (t, o) in PLAN for sg in SIGMAS for sd in SEEDS]
    if '--prueba' in sys.argv:
        configs = [dict(truth='T4_ef', option=o, sigma=0.28, seed=1, steps=steps) for o in ('A', 'B', 'D')]
    opts = dict(a.split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    if '--solo' in opts:                           # p. ej. --solo=T3_joule_D_s0.10_seed1,T4_ef_A_s0.10_seed1
        quiero = set(opts['--solo'].split(','))
        configs = [c for c in configs if _nombre(c) in quiero]
    max_horas = float(opts.get('--max-horas', 0)) or None
    out = OUT.parent / (OUT.name + '_prueba') if '--prueba' in sys.argv else OUT
    X = load_design()
    for c in configs:
        c['out'] = str(out)
        c['X'] = X.to_dict('list')
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(configs)} configuraciones, {steps} pasos, {nproc} procesos, cpu_count={os.cpu_count()}', flush=True)
    with open(out / 'lanzamiento.txt', 'a', encoding='utf-8') as fh:
        fh.write(
        f'{time.strftime("%Y-%m-%d %H:%M")} configs={len(configs)} pasos={steps} procesos={nproc} '
        f'cpu={os.cpu_count()} {platform.platform()} python={platform.python_version()}\n')
    _planificar(configs, nproc, max_horas)


def _nombre(c):
    return f"{c['truth']}_{c['option']}_s{c['sigma']:.2f}_seed{c['seed']}"


def _hecho(c):
    f = Path(c['out']) / f'{_nombre(c)}.json'
    try:
        return json.loads(f.read_text(encoding='utf-8')).get('version_script') == VERSION
    except Exception:
        return False


def _entrada_proceso(c):
    name, msg = run_one(c)
    print(name, msg, flush=True)


def _planificar(configs, nproc, max_horas):
    """Un proceso por configuración, con límite de tiempo: una cadena que supera max_horas se detiene
    y se registra como <nombre>_TIEMPO_AGOTADO.txt (con su último progreso). Añadido el 2026-10-15
    tras cadenas de >15 h; no cambia los cálculos de las cadenas que terminan."""
    import multiprocessing as mp
    pend = [c for c in configs if not _hecho(c)]
    print(f'{len(configs) - len(pend)} ya hechas; {len(pend)} por ejecutar; límite {max_horas} h por cadena', flush=True)
    act = {}
    while pend or act:
        while pend and len(act) < nproc:
            c = pend.pop(0)
            p = mp.Process(target=_entrada_proceso, args=(c,))
            p.start()
            act[p] = (c, time.time())
        time.sleep(10)
        for p, (c, t0) in list(act.items()):
            if not p.is_alive():
                p.join(); del act[p]
            elif max_horas and time.time() - t0 > max_horas * 3600:
                p.terminate(); p.join(); del act[p]
                out = Path(c['out']); name = _nombre(c)
                prog = out / f'{name}.progreso.json'
                ultimo = prog.read_text(encoding='utf-8') if prog.exists() else 'sin progreso registrado'
                (out / f'{name}_TIEMPO_AGOTADO.txt').write_text(
                    f'Detenida tras {max_horas} h ({time.strftime("%Y-%m-%d %H:%M")}).\nÚltimo progreso: {ultimo}\n',
                    encoding='utf-8')
                print(name, f'TIEMPO AGOTADO ({max_horas} h)', flush=True)

if __name__ == '__main__':
    main()
