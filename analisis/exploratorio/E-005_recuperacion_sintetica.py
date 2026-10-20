"""E-005 (EXPLORATORIO, piloto de F2). Recuperación de fórmulas conocidas con DATOS SINTÉTICOS.

No usa la TSLBC real: solo el diseño X real (WT, WC medida, EF) y respuestas sintéticas
y = f_verdad(X) + N(0, sigma^2). Objetivos:
  1. ¿Recupera el BMS la estructura verdadera con n = 242 y ruido realista? (RQ4, protocolo §8)
  2. ¿Cuántos pasos MCMC hacen falta? (traza de la longitud de descripción)
  3. Comparar las opciones A y B de D-006.
Criterio (Fajardo-Fontiveros et al. 2023): se compara la longitud de descripción (DL) del modelo
MDL encontrado con la DL de la fórmula verdadera (reajustada a los mismos datos):
  DL_MDL > DL_verdad  -> fallo de muestreo (la verdad era más plausible y no se encontró)
  DL_MDL <= DL_verdad -> el BMS no puede preferir la verdad con estos datos (o la ha encontrado)
y la distancia funcional RMSE(f_MDL, f_verdad) en los puntos del diseño.

Uso (desde la raíz del proyecto):
  python analisis/exploratorio/E-005_recuperacion_sintetica.py [pasos=20000] [procesos=auto] [--prueba]
Es reanudable: las configuraciones con JSON final ya escrito se saltan.
"""
import sys, os, json, time, hashlib, platform, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'analisis' / 'vendor' / 'machine-scientist'
PRIORS = ROOT / 'analisis' / 'priors'
DATA = ROOT / 'Dataset' / 'datos_corregido_junio16_2_puntos20-69-88.xlsx'
SHA = 'ae881203f8f6027daf1baa3bccc56aa0d9d528b8d6cd2aaa0a87af47fc02e99b'
OUT = ROOT / 'resultados' / 'E-005_recuperacion_sintetica'

# ---------------- Verdades (formato de cadena del BMS oficial) ----------------
TRUTHS = {
    # Saturación tipo Joule, 2 parámetros (representable en A y B)
    'T3_joule': dict(expr='(_a0_ + -((_a0_ * exp(-((_a1_ * (pow2(WC) * WT)))))))',
                     par={'_a0_': 7.0, '_a1_': 0.6},
                     f=lambda WT, WC: 7.0*(1-np.exp(-0.6*WC**2*WT))),
    # Saturación con exponentes libres (forma de E-001), 4 parámetros (solo representable en B)
    'T1_potencias': dict(expr='(_a0_ + -((_a0_ * exp(-((_a1_ * ((WC ** _a2_) * (WT ** _a3_))))))))',
                         par={'_a0_': 7.3, '_a1_': 0.342, '_a2_': 1.38, '_a3_': 0.345},
                         f=lambda WT, WC: 7.3*(1-np.exp(-0.342*WC**1.38*WT**0.345))),
    # T1 con caída por expulsión a alto aporte térmico, 5 parámetros (solo B)
    'T2_expulsion': dict(expr='((_a0_ + -((_a0_ * exp(-((_a1_ * ((WC ** _a2_) * (WT ** _a3_)))))))) + -((_a4_ * pow2((pow2(WC) * WT)))))',
                         par={'_a0_': 7.3, '_a1_': 0.342, '_a2_': 1.38, '_a3_': 0.345, '_a4_': 0.005},
                         f=lambda WT, WC: 7.3*(1-np.exp(-0.342*WC**1.38*WT**0.345)) - 0.005*(WC**2*WT)**2),
}
OPTIONS = {
    'A': dict(variables=['WT', 'WC', 'EF'], npar=3, prior='nv3.np3', multi=False),
    'B': dict(variables=['WT', 'WC'], npar=6, prior='nv2.np6', multi=True),   # EF = 2 conjuntos de datos
}
SIGMAS = [0.10, 0.28]
SEEDS = [1, 2, 3]
VERSION = 2   # v2: parche D-007 (bms_parches) y captura de errores por configuración
TS = [1] + [1.04**k for k in range(1, 20)]   # temperaturas por defecto del tutorial oficial


def load_design():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == SHA, 'El fichero de datos ha cambiado'
    df = pd.read_excel(DATA, header=0).iloc[:, 2:9]
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
    except Exception as e:                        # una configuración fallida no detiene las demás
        import traceback
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{name}_ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return name, f'ERROR: {e!r}'


def _run_one(cfg, name, out, final):
    import mcmc                                   # noqa
    from mcmc import Tree
    from parallel import Parallel
    from fit_prior import read_prior_par
    import random as pyrandom
    import bms_parches
    for k in bms_parches.CONTADOR:                # contador por configuración (el Pool reutiliza procesos)
        bms_parches.CONTADOR[k] = 0
    truth, opt = TRUTHS[cfg['truth']], OPTIONS[cfg['option']]
    X = load_design()
    rng = np.random.default_rng(1000 + cfg['seed'])
    f_true = truth['f'](X['WT'].values, X['WC'].values)
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
    if len(truth['par']) <= opt['npar']:
        t = Tree(variables=opt['variables'], parameters=params, x=x, y=ys, prior_par=prior,
                 from_string=truth['expr'])
        pv = {p: truth['par'].get(p, 1.0) for p in ['_%s_' % q for q in params]}
        t.set_par_values({k: dict(pv) for k in x} if opt['multi'] else pv)
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

    res = dict(cfg=cfg, nombre=name, dl_mdl=best, dl_verdad=dl_truth,
               delta_dl=None if dl_truth is None else best - dl_truth,
               modelo_mdl=str(best_tree), parametros_mdl={k: {p: float(v) for p, v in d.items()}
                                                          for k, d in best_tree.par_values.items()},
               paso_mdl=best_step, rmse_funcional=rmse_func, sigma=cfg['sigma'],
               tiempo_s=dt, pasos_por_s=cfg['steps']/dt, traza_cada10=trace,
               version_script=VERSION, parches=dict(bms_parches.CONTADOR),
               plataforma=platform.platform(), python=platform.python_version())
    out.mkdir(parents=True, exist_ok=True)
    final.write_text(json.dumps(res, indent=1), encoding='utf-8')
    return name, f"DL_MDL={best:.2f} DL_verdad={dl_truth} rmse_f={rmse_func:.3f} {dt/60:.1f} min"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    steps = int(args[0]) if args else 20000
    nproc = int(args[1]) if len(args) > 1 else max(1, (os.cpu_count() or 2) - 1)
    configs = [dict(truth=t, option=o, sigma=s, seed=sd, steps=steps)
               for t, o, s, sd in itertools.product(TRUTHS, OPTIONS, SIGMAS, SEEDS)
               if not (o == 'A' and len(TRUTHS[t]['par']) > OPTIONS['A']['npar'] and sd > 1)]
    # En A, T1/T2 no son representables: se ejecuta solo 1 semilla por sigma como referencia
    if '--prueba' in sys.argv:
        configs = [dict(truth='T3_joule', option=o, sigma=0.28, seed=1, steps=steps) for o in OPTIONS]
    out = OUT.parent / (OUT.name + '_prueba') if '--prueba' in sys.argv else OUT
    for c in configs:
        c['out'] = str(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(configs)} configuraciones, {steps} pasos, {nproc} procesos, cpu_count={os.cpu_count()}', flush=True)
    with open(out / 'lanzamiento.txt', 'a', encoding='utf-8') as fh:
        fh.write(
        f'{time.strftime("%Y-%m-%d %H:%M")} configs={len(configs)} pasos={steps} procesos={nproc} '
        f'cpu={os.cpu_count()} {platform.platform()} python={platform.python_version()}\n')
    if nproc == 1:
        for c in configs:
            print(*run_one(c), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            for name, msg in pool.imap_unordered(run_one, configs):
                print(name, msg, flush=True)


if __name__ == '__main__':
    main()
