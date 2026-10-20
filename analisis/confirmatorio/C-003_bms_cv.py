"""C-003 (CONFIRMATORIO, protocolo §2B, §3, §7): BMS (configuración D) en las 25 particiones agrupadas.
4 cadenas por partición; semillas 100000 + 100·partición + cadena (partición = 5·repetición + k);
20 000 pasos; muestras desde el paso 10 000 cada 100 con predicciones de prueba; un proceso por cadena;
límite de seguridad 12 h y una relanzada. Reanudable.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-003_bms_cv.py [procesos=20]
     ... --prueba-sintetica [pasos]   (respuesta sintética, 2 particiones, salida en resultados\\_pruebas)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from bms_comun import cargar_datos, planificar, ROOT

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    prueba = '--prueba-sintetica' in sys.argv
    nproc = int(args[0]) if args else 20
    pasos = int(args[1]) if (prueba and len(args) > 1) else (200 if prueba else 20000)
    df = cargar_datos()
    folds = pd.read_csv(ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv')
    if prueba:
        rng = np.random.default_rng(7)
        df['T'] = 7.0*(1-np.exp(-0.6*df['WC']**2*df['WT'])) + rng.normal(0, 0.28, len(df))
        out = ROOT/'resultados'/'_pruebas'/'C-003_sintetica'
    else:
        out = ROOT/'resultados'/'C-003_bms_cv'
    cfgs = []
    for rep in range(5):
        for k in range(5):
            p = 5*rep + k
            if prueba and p >= 2:
                continue
            ids_te = set(folds.query('repeticion == @rep and particion == @k')['id'])
            te = df[df.id.isin(ids_te)]; tr = df[~df.id.isin(ids_te)]
            for c in range(1, 5 if not prueba else 3):
                cfgs.append(dict(nombre=f'p{p:02d}_cadena{c}', out=str(out), seed=100000 + 100*p + c, steps=pasos,
                                 variables=['WT', 'WC'], multi=False, prior='nv2.np6', npar=6,
                                 train=tr[['WT', 'WC', 'EF', 'T']].to_dict('list'),
                                 test=te[['WT', 'WC', 'EF', 'T']].to_dict('list'),
                                 burn=pasos//2, thin=max(1, pasos//200),
                                 particion=p, repeticion=rep, k=k, ids_test=te['id'].tolist(), ids_train=tr['id'].tolist()))
    planificar(cfgs, nproc, max_horas=12.0)
