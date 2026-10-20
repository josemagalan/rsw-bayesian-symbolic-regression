"""C-002 (CONFIRMATORIO, protocolo §3, comparador 7): PySR en las 25 particiones agrupadas.
Entradas WT, WC. Operadores binarios {+, −, ×, ÷, ^}; unarios {exp, log, sqrt, tanh}; constraints {'^': (-1, 1)};
anidamiento prohibido de exp/log/tanh dentro de sí mismos; maxsize 25; niterations 200; populations 20;
model_selection 'best'; random_state 20261016 + partición; deterministic=True (serie).
Un proceso por partición (cada uno arranca su Julia). Reanudable.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-002_pysr_cv.py [procesos=6]
     ... --prueba-sintetica   (respuesta sintética, 2 particiones, 5 iteraciones; salida en resultados\\_pruebas)"""
import json, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA


def correr(cfg):
    out = Path(cfg['out']); f = out/f"p{cfg['p']:02d}.json"
    if f.exists():
        return cfg['p'], 'ya existía'
    try:
        import sympy as sp
        from pysr import PySRRegressor
        Xtr, ytr, Xte, yte = (np.array(cfg[k]) for k in ('Xtr', 'ytr', 'Xte', 'yte'))
        t0 = time.process_time(); tw = time.time()
        m = PySRRegressor(niterations=cfg['niter'], populations=20, maxsize=25,
                          binary_operators=['+', '-', '*', '/', '^'], unary_operators=['exp', 'log', 'sqrt', 'tanh'],
                          constraints={'^': (-1, 1)},
                          nested_constraints={'exp': {'exp': 0}, 'log': {'log': 0}, 'tanh': {'tanh': 0}},
                          model_selection='best', random_state=20261016 + cfg['p'], deterministic=True,
                          parallelism='serial', progress=False, verbosity=0,
                          output_directory=str(out/'pysr_tmp'), run_id=f"p{cfg['p']:02d}")
        m.fit(Xtr, ytr, variable_names=['WT', 'WC'])
        yp = np.asarray(m.predict(Xte), dtype=float); nf = int((~np.isfinite(yp)).sum())
        yp = np.where(np.isfinite(yp), yp, ytr.mean())
        ex = m.sympy(); nconst = len([a for a in ex.atoms(sp.Number) if not a.is_Integer])
        best = m.get_best()
        res = dict(p=cfg['p'], repeticion=cfg['rep'], k=cfg['k'], mse=float(np.mean((yp-yte)**2)), no_finitos=nf,
                   ecuacion=str(ex), complejidad=int(best['complexity']), n_constantes=nconst,
                   pred=yp.tolist(), y=yte.tolist(), ids_test=cfg['ids_test'],
                   frontera=m.equations_[['complexity', 'loss', 'equation']].astype(str).to_dict('records'),
                   cpu_s=time.process_time()-t0, reloj_s=time.time()-tw, python=platform.python_version(),
                   protocolo_sha=PROTOCOLO_SHA)
        f.write_text(json.dumps(res), encoding='utf-8')
        return cfg['p'], f"MSE={res['mse']:.4f} {res['ecuacion'][:70]}"
    except Exception as e:
        import traceback
        (out/f"p{cfg['p']:02d}_ERROR.txt").write_text(traceback.format_exc(), encoding='utf-8')
        return cfg['p'], f'ERROR {e!r}'


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    prueba = '--prueba-sintetica' in sys.argv
    nproc = int(args[0]) if args else 6
    df = cargar_datos()
    folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
    if prueba:
        rng = np.random.default_rng(7); df['T'] = 7.0*(1-np.exp(-0.6*df['WC']**2*df['WT'])) + rng.normal(0, 0.28, len(df))
        out = ROOT/'resultados'/'_pruebas'/'C-002_sintetica'
    else:
        out = ROOT/'resultados'/'C-002_pysr_cv'
    out.mkdir(parents=True, exist_ok=True)
    cfgs = []
    for rep in range(5):
        for k in range(5):
            p = 5*rep + k
            if prueba and p >= 2:
                continue
            ids_te = set(folds.query('repeticion == @rep and particion == @k')['id'])
            te = df[df.id.isin(ids_te)]; tr = df[~df.id.isin(ids_te)]
            cfgs.append(dict(p=p, rep=rep, k=k, out=str(out), niter=5 if prueba else 200,
                             Xtr=tr[['WT', 'WC']].values.tolist(), ytr=tr['T'].tolist(),
                             Xte=te[['WT', 'WC']].values.tolist(), yte=te['T'].tolist(), ids_test=te['id'].tolist()))
    from multiprocessing import Pool
    print(f'{len(cfgs)} particiones, {nproc} procesos', flush=True)
    with Pool(nproc, maxtasksperchild=1) as pool:
        for p, msg in pool.imap_unordered(correr, cfgs):
            print(f'p{p:02d}', msg, flush=True)
