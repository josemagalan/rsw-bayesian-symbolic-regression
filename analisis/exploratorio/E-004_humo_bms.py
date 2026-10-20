"""E-004 (TÉCNICO, 2026-10-14). Prueba de humo del BMS oficial con DATOS SINTÉTICOS.
No usa la TSLBC real: genera y = 7(1 - exp(-0.34 WC^1.4 WT^0.35)) + ruido N(0, 0.28) sobre el diseño 11x11x2.
Mide la velocidad (pasos MCMC/s) y comprueba que el código corre en este entorno.
Uso (desde la raíz): python analisis/exploratorio/E-004_humo_bms.py [pasos=500] [semilla=1]
OJO: no usar nombres de variables reservados por sympy (I, E, S, N, O, Q, pi...).
"""
import sys, time, platform, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'analisis' / 'vendor' / 'machine-scientist'
sys.path.insert(0, str(VENDOR)); sys.path.insert(0, str(VENDOR / 'Prior'))
from mcmc import *                     # noqa: E402,F401 (API oficial)
sys.path.insert(0, str(ROOT / 'analisis'))
import bms_parches; bms_parches.aplicar()   # D-007
from parallel import Parallel          # noqa: E402
from fit_prior import read_prior_par   # noqa: E402
import random as pyrandom              # después de 'from mcmc import *' (que importa random())

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
pyrandom.seed(seed); np.random.seed(seed)

prior_file = next((ROOT / 'analisis' / 'priors').glob('*nv3.np3.*.dat'))
prior = read_prior_par(str(prior_file))

cyc = np.repeat(np.arange(2, 13), 22)
WC = np.tile(np.repeat(np.linspace(1.5, 6.5, 11), 2), 11)
EF = np.tile([1.0, 1.5], 121)
WT = cyc * 0.02
x = pd.DataFrame({'WT': WT, 'WC': WC, 'EF': EF})
rng = np.random.default_rng(seed)
y = pd.Series(7*(1-np.exp(-0.34*WC**1.4*WT**0.35)) + rng.normal(0, 0.28, len(x)))

Ts = [1] + [1.04**k for k in range(1, 20)]
pms = Parallel(Ts, variables=['WT', 'WC', 'EF'], parameters=['a%d' % i for i in range(3)],
               x=x, y=y, prior_par=prior)
t0, best, best_model, trace = time.time(), np.inf, None, []
for i in range(steps):
    pms.mcmc_step(); pms.tree_swap()
    trace.append(pms.t1.E)
    if pms.t1.E < best:
        best, best_model = pms.t1.E, str(pms.t1)
dt = time.time() - t0

out = ROOT / 'resultados' / f'{time.strftime("%Y-%m-%d")}_E-004_{platform.system()}'
out.mkdir(parents=True, exist_ok=True)
pd.Series(trace, name='E').to_csv(out / f'traza_seed{seed}.csv', index_label='paso')
msg = (f'plataforma={platform.platform()} python={platform.python_version()}\n'
       f'prior={prior_file.name}\npasos={steps} semilla={seed} tiempo={dt:.1f}s '
       f'({steps/dt:.1f} pasos/s)\nMDL={best:.3f}\nmodelo={best_model}\n')
(out / f'resumen_seed{seed}.txt').write_text(msg, encoding='utf-8')
print(msg)
