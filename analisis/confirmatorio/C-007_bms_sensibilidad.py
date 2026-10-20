"""C-007 (protocolo §9 con D-015): sensibilidades del BMS (configuración D, 20 000 pasos, 4 cadenas por partición).
  - aleatoria:  particiones SIN agrupar, repetición 0 (5 particiones), prior nv2.np6.
  - wc_nominal: particiones agrupadas, repetición 0, WC nominal (1.5 + 0.5·(nivel − 1) kA), prior nv2.np6.
  - np4:        particiones agrupadas, repetición 0, prior nv2.np4 (4 parámetros).
    Semillas: 150000 + 100·partición + cadena (D-015: las de C-003 + 50000).
  - extrapolacion: 2 divisiones (WT_extremos, WC_extremos), prior nv2.np6; semillas 160000 + 100·división + cadena.
  Total 68 cadenas. Límite de 12 h y una relanzada (semilla + 10000), como C-003. Reanudable.
  Requiere C-000b (particiones de sensibilidad).
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-007_bms_sensibilidad.py [procesos=20]
     ... --prueba-sintetica [pasos]   (respuesta sintética; 1 partición por variante, 2 cadenas; salida en _pruebas)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from bms_comun import cargar_datos, planificar, ROOT

PROTOCOLO_ANEXO_SHA = 'a9789d2ac4ce61263f1567e68d6b96cff02667f8f39f69e213879a930f207854'   # protocolo + Anexo A (D-015)
VARIANTES = {'aleatoria': dict(folds='folds_5x5_aleatorio.csv', prior='nv2.np6', npar=6, wc='WC'),
             'wc_nominal': dict(folds='folds_5x5_agrupado.csv', prior='nv2.np6', npar=6, wc='WC_nom'),
             'np4': dict(folds='folds_5x5_agrupado.csv', prior='nv2.np4', npar=4, wc='WC')}

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    prueba = '--prueba-sintetica' in sys.argv
    nproc = int(args[0]) if args else 20
    pasos = int(args[1]) if (prueba and len(args) > 1) else (200 if prueba else 20000)
    ncad = 2 if prueba else 4
    df = cargar_datos()
    df['WC_nom'] = 1.5 + 0.5*(df['nivel'] - 1)
    if prueba:
        rng = np.random.default_rng(7)
        df['T'] = 7.0*(1-np.exp(-0.6*df['WC']**2*df['WT'])) + rng.normal(0, 0.28, len(df))
        base = ROOT/'resultados'/'_pruebas'/'C-007_sintetica'
    else:
        base = ROOT/'resultados'/'C-007_bms_sensibilidad'
    DD = ROOT/'resultados'/'_pruebas'/'datos_derivados' if prueba else ROOT/'analisis'/'datos_derivados'
    comun = dict(steps=pasos, variables=['WT', 'WC'], multi=False, burn=pasos//2, thin=max(1, pasos//200),
                 protocolo_anexo_sha=PROTOCOLO_ANEXO_SHA)

    def cfg(nombre, out, seed, te, tr, wc, prior, npar, **extra):
        cols = lambda d: d[['WT', wc, 'EF', 'T']].rename(columns={wc: 'WC'}).to_dict('list')
        return dict(comun, nombre=nombre, out=str(out), seed=seed, prior=prior, npar=npar, train=cols(tr), test=cols(te),
                    ids_test=te['id'].tolist(), ids_train=tr['id'].tolist(), wc=wc, **extra)
    cfgs = []
    for v, o in VARIANTES.items():
        folds = pd.read_csv((ROOT/'analisis'/'datos_derivados' if o['folds'] == 'folds_5x5_agrupado.csv' else DD)/o['folds'])
        for k in range(1 if prueba else 5):
            p = k   # repetición 0 → partición = k
            ids_te = set(folds.query('repeticion == 0 and particion == @k')['id'])
            te, tr = df[df.id.isin(ids_te)], df[~df.id.isin(ids_te)]
            for c in range(1, ncad + 1):
                cfgs.append(cfg(f'p{p:02d}_cadena{c}', base/v, 150000 + 100*p + c, te, tr, o['wc'], o['prior'], o['npar'],
                                variante=v, particion=p, repeticion=0, k=k))
    ext = pd.read_csv(DD/'particiones_extrapolacion.csv')
    for s, div in enumerate(['WT_extremos', 'WC_extremos']):
        ids_te = set(ext.query('division == @div and conjunto == "prueba"')['id'])
        te, tr = df[df.id.isin(ids_te)], df[~df.id.isin(ids_te)]
        for c in range(1, ncad + 1):
            cfgs.append(cfg(f'{div}_cadena{c}', base/'extrapolacion', 160000 + 100*s + c, te, tr, 'WC', 'nv2.np6', 6,
                            variante='extrapolacion', division=div))
    planificar(cfgs, nproc, max_horas=12.0)
