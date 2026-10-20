"""E-012 (EXPLORATORIO; 2026-10-18). Control del prior: ¿aparecen los motivos 1/WT y log por el prior del BMS o por los datos?
Muestrea el prior nv2.np6 SIN datos (con datos vacíos el BMS pone BIC = 0 y la energía es solo la del prior), con la misma
configuración que C-003 (2 variables, 6 parámetros, 20 temperaturas), pero con 12 000 pasos (quemado 2000) para caber en la VM
(sin datos no hay ajuste y el prior mezcla rápido). 8 cadenas, semillas 300001–300008. Muestras de t1 una de cada 10 tras el quemado. Compara la frecuencia de los motivos con la del posterior (muestras de
C-003) y con los MDL (C-001 D, C-003).
Uso: python analisis/exploratorio/E-012_prior_motivos.py correr <cadena> [<cadena> ...]   |   python ... analizar"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from bms_comun import _importar_bms, PRIORS, TS, ROOT

OUT = ROOT/'resultados'/'2026-10-18_E-012'; OUT.mkdir(parents=True, exist_ok=True)
PASOS, QUEMADO, CADA = 12000, 2000, 10


def cadena(c):
    import random
    mcmc, bp, Tree, Parallel, read_prior_par = _importar_bms()
    random.seed(300000 + c); np.random.seed(300000 + c)
    prior = read_prior_par(str(next(PRIORS.glob('*nv2.np6.*.dat'))))
    x = pd.DataFrame({'WT': pd.Series(dtype=float), 'WC': pd.Series(dtype=float)}); y = pd.Series(dtype=float)
    pms = Parallel(TS, variables=['WT', 'WC'], parameters=['a%d' % i for i in range(6)], x=x, y=y, prior_par=prior)
    m, t0 = [], time.time()
    for i in range(PASOS):
        pms.mcmc_step(); pms.tree_swap()
        if i >= QUEMADO and (i - QUEMADO) % CADA == 0:
            try:
                can = pms.t1.canonical()
            except BaseException:
                can = None
            m.append(dict(paso=i, E=float(pms.t1.E), expr=str(pms.t1), canon=can))
    (OUT/f'prior_cadena{c}.json').write_text(json.dumps(dict(cadena=c, semilla=300000 + c, muestras=m, parches=dict(bp.CONTADOR),
                                                             minutos=(time.time() - t0)/60)), encoding='utf-8')
    return c


def motivos(s):
    s = s.replace(' ', '')
    return dict(inv_WT=bool(re.search(r'/WT(?![\w*])|/\(WT[+\-]', s)), log=('log(' in s),
                log_WC=bool(re.search(r'log\([^()]*WC', s)), ambas=('WT' in s and 'WC' in s),
                n_param=len(set(re.findall(r'c\d+', s))))


if __name__ == '__main__':
    if sys.argv[1] == 'correr':
        import multiprocessing as mp
        ids = [int(a) for a in sys.argv[2:]] or list(range(1, 9))
        with mp.Pool(min(2, len(ids))) as pool:
            for c in pool.imap_unordered(cadena, ids):
                print('cadena', c, 'hecha', flush=True)
    else:
        filas = []
        for f in sorted(OUT.glob('prior_cadena*.json')):
            d = json.loads(f.read_text(encoding='utf-8'))
            filas += [dict(fuente='prior (sin datos)', canon=m['canon']) for m in d['muestras'] if m['canon']]
        for f in sorted((ROOT/'resultados'/'C-003_bms_cv').glob('p??_cadena?.json')):
            d = json.loads(f.read_text(encoding='utf-8'))
            filas += [dict(fuente='posterior CV (muestras C-003)', canon=m['canon']) for m in d['muestras']]
            filas.append(dict(fuente='MDL de cada cadena CV (C-003)', canon=d['canon_mdl']))
        for f in sorted((ROOT/'resultados'/'C-001_analisis_A').glob('D_cadena?.json')):
            d = json.loads(f.read_text(encoding='utf-8'))
            filas += [dict(fuente='posterior A (muestras C-001 D)', canon=m['canon']) for m in d['muestras']]
        b = pd.read_csv(ROOT/'resultados'/'C-005_analisis_cv'/'bms_por_particion.csv')
        filas += [dict(fuente='MDL por partición (C-005)', canon=c) for c in b.canon]
        D = pd.DataFrame(filas); D = pd.concat([D, pd.DataFrame([motivos(s) for s in D.canon])], axis=1)
        D.to_csv(OUT/'motivos_por_expresion.csv', index=False)
        res = []
        for fu, g in D.groupby('fuente', sort=False):
            gb = g[g.ambas]
            res.append(dict(fuente=fu, n=len(g), frac_con_WT_y_WC=g.ambas.mean(),
                            inv_WT=g.inv_WT.mean(), log=g.log.mean(), log_WC=g.log_WC.mean(),
                            inv_WT_si_ambas=gb.inv_WT.mean(), log_si_ambas=gb.log.mean(), log_WC_si_ambas=gb.log_WC.mean(),
                            ambos_motivos_si_ambas=(gb.inv_WT & gb.log).mean(), n_param_mediana=g.n_param.median()))
        R = pd.DataFrame(res); R.to_csv(OUT/'resumen.csv', index=False)
        pd.set_option('display.width', 250); print(R.round(3).to_string(index=False))
        print(D[D.fuente == 'prior (sin datos)'].canon.value_counts().head(15).to_string())
