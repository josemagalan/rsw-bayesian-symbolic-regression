"""C-008 (protocolo §9 con D-015): sensibilidades de los comparadores 1-6, 8 y 9 (C-004) y de PySR (comparador 7, C-002).
Variantes (salida en resultados/C-008_sensibilidad_comparadores/<variante>/):
  - aleatoria:     25 particiones SIN agrupar (folds_5x5_aleatorio.csv, C-000b). La CV interna sigue al esquema externo:
                   KFold(5) sin agrupar, barajada, semilla 20261016 + partición.
  - wc_nominal:    25 particiones agrupadas; WC nominal = 1.5 + 0.5·(nivel - 1) kA en lugar de la medida.
  - sin_cero:      25 particiones agrupadas sin el punto de TSLBC = 0 (fuera del entrenamiento y de la prueba; se reajusta todo).
  - extrapolacion: WT_extremos (p = 0) y WC_extremos (p = 1) de particiones_extrapolacion.csv (C-000b); CV interna agrupada
                   por celda, semilla 20261016 + 1000 + p.
  Todo lo demás es idéntico al análisis principal: las funciones de los comparadores y la lista MODELOS se ejecutan desde
  el código fuente de C-004 (no se copian), y PySR usa la función correr() de C-002 (mismos hiperparámetros;
  random_state 20261016 + p). La CV interna agrupada es la de C-004 (GroupKFold 5 por celda, barajada).
  Otras sensibilidades de §9: comparadores sin EF -> C-004 --sin-ef; recorte de predicciones -> posproceso en el análisis.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-008_sensibilidad_comparadores.py [procesos_pysr=6]
     ... --solo-comparadores | --solo-pysr
     ... --prueba-sintetica   (respuesta sintética, 2 particiones por variante, PySR 5 iteraciones; salida en _pruebas)
     ... --sin-tabpfn / --sin-pysr   (solo pruebas en entornos sin TabPFN o sin Julia; NO en la ejecución confirmatoria)
Reanudable: los pares (modelo, partición) ya hechos sin fallo y los p??.json de PySR existentes no se repiten."""
import ast, hashlib, importlib.util, json, sys, time, platform, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA
warnings.filterwarnings('ignore')

AQUI = Path(__file__).resolve().parent
PROTOCOLO_ANEXO_SHA = 'a9789d2ac4ce61263f1567e68d6b96cff02667f8f39f69e213879a930f207854'   # protocolo + Anexo A (D-015)
VARIANTES = ['aleatoria', 'wc_nominal', 'sin_cero', 'extrapolacion']


def cargar_modelos_c004():
    """Ejecuta del fuente de C-004 solo los import, las funciones m_* y la asignación MODELOS (nunca su bucle)."""
    f = AQUI/'C-004_comparadores.py'
    arbol = ast.parse(f.read_text(encoding='utf-8'))
    nodos = [n for n in arbol.body
             if isinstance(n, (ast.Import, ast.ImportFrom))
             or (isinstance(n, ast.FunctionDef) and n.name.startswith('m_'))
             or (isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'MODELOS' for t in n.targets))]
    ns = {'__file__': str(f), '__name__': 'c004_modelos'}
    exec(compile(ast.Module(body=nodos, type_ignores=[]), str(f), 'exec'), ns)
    return ns['MODELOS'], hashlib.sha256(f.read_bytes()).hexdigest()


def correr_pysr(cfg):
    spec = importlib.util.spec_from_file_location('c002_pysr', AQUI/'C-002_pysr_cv.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return cfg['variante'], mod.correr(cfg)


def cv_interna(tr, semilla, agrupada):
    X = np.zeros((len(tr), 1))
    if agrupada:
        return list(GroupKFold(5, shuffle=True, random_state=semilla).split(X, groups=tr['celda'].values))
    return list(KFold(5, shuffle=True, random_state=semilla).split(X))


def preparar(variante, df0, id_cero, DD, prueba):
    """Devuelve el conjunto de datos de la variante y la lista de divisiones (p, rep, k, división, ids de prueba, semilla)."""
    df = df0.copy()
    if variante == 'wc_nominal':
        df['WC'] = 1.5 + 0.5*(df['nivel'] - 1)
    if variante == 'sin_cero':
        df = df[df['id'] != id_cero].reset_index(drop=True)
    divs = []
    if variante == 'extrapolacion':
        ext = pd.read_csv(DD/'particiones_extrapolacion.csv')
        for s, div in enumerate(['WT_extremos', 'WC_extremos']):
            ids = set(ext.query('division == @div and conjunto == "prueba"')['id'])
            divs.append(dict(p=s, rep=-1, k=s, division=div, ids=ids, semilla=20261016 + 1000 + s))
    else:
        fichero = 'folds_5x5_aleatorio.csv' if variante == 'aleatoria' else 'folds_5x5_agrupado.csv'
        folds = pd.read_csv((DD if variante == 'aleatoria' else ROOT/'analisis'/'datos_derivados')/fichero)
        for rep in range(5):
            for k in range(5):
                p = 5*rep + k
                if prueba and p >= 2:
                    continue
                ids = set(folds.query('repeticion == @rep and particion == @k')['id'])
                divs.append(dict(p=p, rep=rep, k=k, division='', ids=ids, semilla=20261016 + p))
    return df, divs


def comparadores(variante, df, divs, modelos, out, sha_c004):
    out.mkdir(parents=True, exist_ok=True)
    agrupada = variante != 'aleatoria'
    filas, preds, hechas = [], [], set()
    fp = out/'mse_por_particion.csv'
    if fp.exists():
        prev = pd.read_csv(fp).fillna({'fallo': ''})
        ok = prev[prev['fallo'] == '']             # las filas con fallo técnico se repiten al relanzar
        filas = ok.to_dict('records'); hechas = {(r['modelo'], r['particion']) for r in filas}
        preds = pd.read_csv(out/'predicciones.csv').merge(ok[['modelo', 'particion']], on=['modelo', 'particion']).to_dict('records')
    VARS = ['WT', 'WC', 'EF']
    for d in divs:
        te = df[df.id.isin(d['ids'])]; tr = df[~df.id.isin(d['ids'])]
        Xtr, ytr, Xte, yte = tr[VARS].values, tr['T'].values, te[VARS].values, te['T'].values
        cv = cv_interna(tr, d['semilla'], agrupada)
        for nombre, fn in modelos:
            if (nombre, d['p']) in hechas:
                continue
            t0 = time.process_time(); tw = time.time()
            try:
                yp, info = fn(Xtr, ytr, Xte, cv, d['semilla'], tr); fallo = ''
            except Exception as e:
                yp, info, fallo = np.full(len(Xte), np.nan), {}, repr(e)
            yp = np.asarray(yp, dtype=float); nf = int((~np.isfinite(yp)).sum())
            yp = np.where(np.isfinite(yp), yp, ytr.mean())
            mse = float(np.mean((yp - yte)**2))
            filas.append(dict(variante=variante, modelo=nombre, particion=d['p'], repeticion=d['rep'], k=d['k'],
                              division=d['division'], mse=mse, n_test=len(yte), n_train=len(ytr), no_finitos=nf,
                              fallo=fallo, cpu_s=round(time.process_time()-t0, 2), reloj_s=round(time.time()-tw, 2),
                              info=json.dumps(info, default=str)))
            preds += [dict(modelo=nombre, particion=d['p'], id=int(i), y=float(a), pred=float(b), media_train=float(ytr.mean()))
                      for i, a, b in zip(te['id'], yte, yp)]
            print(f"{variante:13s} p{d['p']:02d} {nombre:18s} MSE={mse:.4f} no_finitos={nf} {fallo[:60]}", flush=True)
        pd.DataFrame(filas).to_csv(fp, index=False); pd.DataFrame(preds).to_csv(out/'predicciones.csv', index=False)
    (out/'entorno.json').write_text(json.dumps(dict(
        python=platform.python_version(), plataforma=platform.platform(), protocolo_sha=PROTOCOLO_SHA,
        protocolo_anexo_sha=PROTOCOLO_ANEXO_SHA, sha_C004=sha_c004, variante=variante, variables=VARS,
        cv_interna='GroupKFold(celda)' if agrupada else 'KFold', modelos=[m[0] for m in modelos])), encoding='utf-8')
    return pd.DataFrame(filas)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    prueba = '--prueba-sintetica' in sys.argv
    hacer_comp = '--solo-pysr' not in sys.argv
    hacer_pysr = '--solo-comparadores' not in sys.argv and '--sin-pysr' not in sys.argv
    nproc = int(args[0]) if args else 6
    actual = hashlib.sha256((ROOT/'00_gestion'/'PROTOCOLO_ANALISIS.md').read_bytes()).hexdigest()
    if actual != PROTOCOLO_ANEXO_SHA:
        print(f'AVISO: el protocolo ha cambiado (SHA {actual[:12]}…); se registra el esperado.', flush=True)
    df0 = cargar_datos()
    id_cero = df0.loc[df0['T'] == 0, 'id'].tolist()
    assert len(id_cero) == 1, id_cero
    id_cero = id_cero[0]
    if prueba:
        rng = np.random.default_rng(7)
        df0['T'] = 7.0*(1-np.exp(-0.6*df0['WC']**2*df0['WT'])) + rng.normal(0, 0.28, len(df0))
        base = ROOT/'resultados'/'_pruebas'/'C-008_sintetica'
        DD = ROOT/'resultados'/'_pruebas'/'datos_derivados'
    else:
        base = ROOT/'resultados'/'C-008_sensibilidad_comparadores'
        DD = ROOT/'analisis'/'datos_derivados'
    datos = {v: preparar(v, df0, id_cero, DD, prueba) for v in VARIANTES}
    for v, (df, divs) in datos.items():
        print(f'{v}: n = {len(df)}, {len(divs)} divisiones, n_test = {[len(df[df.id.isin(d["ids"])]) for d in divs][:5]}…', flush=True)

    if hacer_comp:
        modelos, sha_c004 = cargar_modelos_c004()
        if '--sin-tabpfn' in sys.argv:
            modelos = [m for m in modelos if m[0] != 'TabPFN_v2']
        res = [comparadores(v, df, divs, modelos, base/v, sha_c004) for v, (df, divs) in datos.items()]
        r = pd.concat(res).groupby(['variante', 'modelo'])['mse'].agg(['mean', 'std', 'count'])
        print(r.to_string(), flush=True)

    if hacer_pysr:
        cfgs = []
        for v, (df, divs) in datos.items():
            out = base/v/'pysr'; out.mkdir(parents=True, exist_ok=True)
            for d in divs:
                te = df[df.id.isin(d['ids'])]; tr = df[~df.id.isin(d['ids'])]
                cfgs.append(dict(variante=v, division=d['division'], p=d['p'], rep=d['rep'], k=d['k'], out=str(out),
                                 niter=5 if prueba else 200,
                                 Xtr=tr[['WT', 'WC']].values.tolist(), ytr=tr['T'].tolist(),
                                 Xte=te[['WT', 'WC']].values.tolist(), yte=te['T'].tolist(), ids_test=te['id'].tolist()))
        from multiprocessing import Pool
        print(f'PySR: {len(cfgs)} ajustes, {nproc} procesos', flush=True)
        with Pool(nproc, maxtasksperchild=1) as pool:
            for v, (p, msg) in pool.imap_unordered(correr_pysr, cfgs):
                print(f'PySR {v:13s} p{p:02d}', msg, flush=True)
