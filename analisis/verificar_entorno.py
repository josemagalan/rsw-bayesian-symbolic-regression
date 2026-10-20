"""Comprueba que el entorno ampliado (protocolo §12) funciona antes de congelar el protocolo.
Uso: .\\.venv\\Scripts\\python.exe analisis\\verificar_entorno.py
Hace ajustes mínimos con datos sintéticos (no usa los datos reales)."""
import importlib, platform, time, sys
import numpy as np
print('Python', platform.python_version(), platform.platform())
rng = np.random.default_rng(0)
X = np.column_stack([rng.uniform(0.04, 0.24, 120), rng.uniform(1.5, 6.5, 120), rng.choice([1.0, 1.5], 120)])
y = 7*(1-np.exp(-0.6*X[:, 1]**2*X[:, 0])) + rng.normal(0, 0.2, 120)
ok = True
def paso(nombre, fn):
    global ok
    t = time.time()
    try:
        r = fn(); print(f'[OK]  {nombre:10s} {time.time()-t:6.1f} s  {r}')
    except Exception as e:
        ok = False; print(f'[ERR] {nombre:10s} {e!r}')
for m in ['numpy', 'scipy', 'sklearn', 'pandas', 'sympy', 'interpret', 'pygam', 'pysr', 'tabpfn', 'baycomp', 'torch']:
    try:
        mod = importlib.import_module(m); print(f'      {m:10s} {getattr(mod, "__version__", "?")}')
    except Exception as e:
        ok = False; print(f'[ERR] import {m}: {e!r}')
def gam():
    from pygam import LinearGAM, te, l
    g = LinearGAM(te(0, 1) + l(2)).gridsearch(X, y, progress=False); return f'R2={g.statistics_["pseudo_r2"]["explained_deviance"]:.3f}'
def pysr():
    from pysr import PySRRegressor
    m = PySRRegressor(niterations=5, populations=4, binary_operators=['+', '-', '*', '/', '^'],
                      unary_operators=['exp', 'log', 'sqrt', 'tanh'], constraints={'^': (-1, 1)},
                      maxsize=25, random_state=0, deterministic=True, parallelism='serial', progress=False, verbosity=0)
    m.fit(X[:, :2], y, variable_names=['WT', 'WC']); return str(m.sympy())[:60]
def tabpfn():
    from tabpfn import TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion
        r = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    except Exception as e:
        print('      (aviso) no se pudo seleccionar V2 con create_default_for_version:', repr(e)); r = TabPFNRegressor()
    r.fit(X[:90], y[:90]); p = r.predict(X[90:]); return f'MSE={np.mean((p-y[90:])**2):.3f}'
def bayc():
    import baycomp
    a, b = rng.normal(0.10, 0.02, 25), rng.normal(0.11, 0.02, 25)
    return str(baycomp.two_on_single(a, b, rope=0.03, runs=5))
paso('GAM', gam); paso('PySR', pysr); paso('TabPFN v2', tabpfn); paso('baycomp', bayc)
print('\nTODO CORRECTO' if ok else '\nHAY ERRORES (pegar la salida a Claude)')
sys.exit(0 if ok else 1)
