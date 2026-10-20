"""Código común de los análisis confirmatorios (protocolo congelado, D-014).
- cargar_datos(): verifica el hash, define celdas (ciclos, nivel nominal de WC) y estratos.
- correr_cadena(cfg): una cadena del BMS oficial (con parches D-007/D-008 v3) en el proceso actual.
- planificar(cfgs, nproc, max_horas): un proceso por cadena, límite de seguridad y UNA relanzada (semilla + 10000).
"""
import hashlib, io, json, os, platform, sys, time, traceback, warnings
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT/'analisis'/'vendor'/'machine-scientist'
PRIORS = ROOT/'analisis'/'priors'
DATA = ROOT/'Dataset'/'datos_corregido_junio16_2_puntos20-69-88.xlsx'
SHA = 'ae881203f8f6027daf1baa3bccc56aa0d9d528b8d6cd2aaa0a87af47fc02e99b'
PROTOCOLO_SHA = 'deebe865618780ca9761c2eb329620076a569a2d84a5e058098a39cae2d23335'
TS = [1] + [1.04**k for k in range(1, 20)]


def cargar_datos():
    for intento in range(5):
        try:
            raw = DATA.read_bytes(); break
        except PermissionError:
            time.sleep(2 + 3*intento)
    else:
        raise PermissionError(DATA)
    assert hashlib.sha256(raw).hexdigest() == SHA, 'El fichero de datos ha cambiado'
    df = pd.read_excel(io.BytesIO(raw), header=0).iloc[:, 2:9]
    df.columns = ['id', 'cyc', 'WT', 'WC', 'EF', 'T', 'Q']
    df = df.dropna(subset=['id']).reset_index(drop=True)
    df['id'] = df['id'].astype(int); df['cyc'] = df['cyc'].astype(int)
    df['EF'] = df['EF'].astype(float)/1000.0                     # kN
    df['nivel'] = df.groupby(['cyc', 'EF'])['WC'].rank(method='first').astype(int)   # 1..11
    df['celda'] = df['cyc']*100 + df['nivel']
    cm = df.groupby('celda')['T'].transform('mean')
    df['estrato'] = pd.qcut(cm.rank(method='first'), 5, labels=False)
    return df


def _importar_bms():
    warnings.filterwarnings('ignore')
    for p in (str(VENDOR), str(VENDOR/'Prior'), str(ROOT/'analisis')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import mcmc  # noqa
    import bms_parches
    bms_parches.aplicar()
    from mcmc import Tree
    from parallel import Parallel
    from fit_prior import read_prior_par
    return mcmc, bms_parches, Tree, Parallel, read_prior_par


def _a_x(datos, variables, multi):
    """datos: dict con listas (columnas). multi -> dict por nivel de EF."""
    df = pd.DataFrame(datos)
    if multi:
        return ({'EF1000': df.loc[df.EF == 1.0, variables].reset_index(drop=True),
                 'EF1500': df.loc[df.EF == 1.5, variables].reset_index(drop=True)},
                {'EF1000': df.loc[df.EF == 1.0, 'T'].reset_index(drop=True),
                 'EF1500': df.loc[df.EF == 1.5, 'T'].reset_index(drop=True)})
    return df[variables].reset_index(drop=True), df['T'].reset_index(drop=True)


def _predecir(tree, xdict_or_df, multi):
    p = tree.predict(xdict_or_df)
    if multi:
        return {k: np.asarray(v, dtype=float).tolist() for k, v in p.items()}
    return np.asarray(p, dtype=float).tolist()


def correr_cadena(cfg):
    """cfg: nombre, out, seed, steps, variables, multi, prior, npar, train (dict columnas incl. T y EF),
    test (dict o None), burn, thin. Escribe <out>/<nombre>.json y <nombre>.progreso.json."""
    mcmc, bp, Tree, Parallel, read_prior_par = _importar_bms()
    import random as pyrandom
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    nombre = cfg['nombre']
    pyrandom.seed(cfg['seed']); np.random.seed(cfg['seed'])
    prior = read_prior_par(str(next(PRIORS.glob(f"*{cfg['prior']}.*.dat"))))
    params = ['a%d' % i for i in range(cfg['npar'])]
    x, y = _a_x(cfg['train'], cfg['variables'], cfg['multi'])
    xt = _a_x(cfg['test'], cfg['variables'], cfg['multi'])[0] if cfg.get('test') else None
    pms = Parallel(TS, variables=cfg['variables'], parameters=params, x=x, y=y, prior_par=prior)
    t0 = time.time(); best = np.inf; best_tree = None; best_step = -1
    trace, muestras, swaps_ok, swaps = [], [], 0, 0
    for i in range(cfg['steps']):
        pms.mcmc_step()
        a, _ = pms.tree_swap(); swaps += 1; swaps_ok += a is not None
        E = float(pms.t1.E)
        if np.isnan(E) or E == -np.inf:
            raise RuntimeError(f'Energía no válida ({E}) en el paso {i}; modelo {pms.t1}')
        if i % 10 == 0:
            trace.append(E)
        if E < best:
            best, best_step = E, i
            best_tree = Tree(variables=cfg['variables'], parameters=params, x=x, y=y, prior_par=prior,
                             from_string=str(pms.t1))
            best_tree.set_par_values(pms.t1.par_values)
        if i >= cfg['burn'] and (i - cfg['burn']) % cfg['thin'] == 0:
            m = dict(paso=i, E=E, expr=str(pms.t1), canon=pms.t1.canonical())
            if xt is not None:
                try:
                    m['pred'] = _predecir(pms.t1, xt, cfg['multi'])
                except BaseException:
                    m['pred'] = None
            muestras.append(m)
        if i % 500 == 0:
            (out/f'{nombre}.progreso.json').write_text(json.dumps(dict(
                paso=i, minutos=round((time.time()-t0)/60, 1), dl_mdl=best, paso_mdl=best_step,
                modelo_actual=str(pms.t1), modelo_mdl=str(best_tree))), encoding='utf-8')
    res = dict(cfg={k: v for k, v in cfg.items() if k not in ('train', 'test')},
               dl_mdl=best, paso_mdl=best_step, modelo_mdl=str(best_tree),
               canon_mdl=best_tree.canonical(),
               parametros_mdl={k: {p: float(v) for p, v in d.items()} for k, d in best_tree.par_values.items()},
               pred_mdl_test=_predecir(best_tree, xt, cfg['multi']) if xt is not None else None,
               pred_mdl_train=_predecir(best_tree, x, cfg['multi']),
               muestras=muestras, traza_cada10=trace, aceptacion_swaps=swaps_ok/max(swaps, 1),
               parches=dict(bp.CONTADOR), expresiones_timeout=list(bp.REGISTRO_TIMEOUT),
               tiempo_s=time.time()-t0, plataforma=platform.platform(), python=platform.python_version(),
               protocolo_sha=PROTOCOLO_SHA)
    (out/f'{nombre}.json').write_text(json.dumps(res, default=float), encoding='utf-8')
    return nombre, f"DL_MDL={best:.2f} paso={best_step} {res['tiempo_s']/60:.1f} min"


def _entrada(cfg):
    try:
        n, msg = correr_cadena(cfg)
        print(n, msg, flush=True)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
        (out/f"{cfg['nombre']}_ERROR.txt").write_text(traceback.format_exc(), encoding='utf-8')
        print(cfg['nombre'], f'ERROR: {e!r}', flush=True)


def hecho(cfg):
    return (Path(cfg['out'])/f"{cfg['nombre']}.json").exists()


def planificar(cfgs, nproc, max_horas=12.0, relanzar=True):
    """Un proceso por cadena. Si una cadena agota max_horas o falla, se relanza UNA vez con semilla + 10000
    (nombre con sufijo _r). Se registra todo en <out>/registro_ejecucion.txt."""
    import multiprocessing as mp
    pend = [c for c in cfgs if not hecho(c)]
    print(f'{len(cfgs)-len(pend)} ya hechas; {len(pend)} por ejecutar; {nproc} procesos; límite {max_horas} h', flush=True)
    act = {}

    def reg(c, txt):
        with open(Path(c['out'])/'registro_ejecucion.txt', 'a', encoding='utf-8') as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {c['nombre']} {txt}\n")

    def relanza(c, motivo):
        if relanzar and not c['nombre'].endswith('_r'):
            c2 = dict(c, nombre=c['nombre']+'_r', seed=c['seed']+10000)
            reg(c, f'{motivo} -> relanzada como {c2["nombre"]} (semilla {c2["seed"]})')
            if not hecho(c2):
                pend.append(c2)
        else:
            reg(c, f'{motivo} (sin más relanzadas)')

    while pend or act:
        while pend and len(act) < nproc:
            c = pend.pop(0)
            Path(c['out']).mkdir(parents=True, exist_ok=True)
            p = mp.Process(target=_entrada, args=(c,)); p.start(); act[p] = (c, time.time())
            reg(c, f'inicio (semilla {c["seed"]})')
        time.sleep(10)
        for p, (c, t0) in list(act.items()):
            if not p.is_alive():
                p.join(); del act[p]
                if hecho(c):
                    reg(c, 'terminada')
                else:
                    relanza(c, 'ERROR')
            elif max_horas and time.time() - t0 > max_horas*3600:
                p.terminate(); p.join(); del act[p]
                prog = Path(c['out'])/f"{c['nombre']}.progreso.json"
                (Path(c['out'])/f"{c['nombre']}_TIEMPO_AGOTADO.txt").write_text(
                    f'Detenida tras {max_horas} h.\nÚltimo progreso: {prog.read_text(encoding="utf-8") if prog.exists() else "-"}\n',
                    encoding='utf-8')
                print(c['nombre'], f'TIEMPO AGOTADO ({max_horas} h)', flush=True)
                relanza(c, f'TIEMPO AGOTADO ({max_horas} h)')
