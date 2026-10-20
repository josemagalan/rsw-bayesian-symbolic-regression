"""C-001 (CONFIRMATORIO, protocolo §2A): BMS con TODOS los datos.
  - Configuración D (WT, WC; un conjunto; prior nv2.np6): 8 cadenas, semillas 1001-1008.
  - Configuración B (WT, WC; EF como 2 conjuntos; nv2.np6): 8 cadenas, semillas 2001-2008 (solo H4b).
20 000 pasos; muestras MCMC desde el paso 10 000 cada 100; un proceso por cadena; límite de seguridad 12 h
y una relanzada (semilla + 10000). Reanudable.
Uso:
  .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-001_analisis_A.py [procesos=16]
  ... --prueba-sintetica [pasos]   (respuesta sintética, salida en resultados\\_pruebas; NO usa la TSLBC real)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from bms_comun import cargar_datos, planificar, ROOT

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    prueba = '--prueba-sintetica' in sys.argv
    nproc = int(args[0]) if args else 16
    pasos = int(args[1]) if (prueba and len(args) > 1) else (300 if prueba else 20000)
    df = cargar_datos()
    if prueba:   # respuesta sintética (Joule): no se mira la TSLBC real
        rng = np.random.default_rng(7)
        df['T'] = 7.0*(1-np.exp(-0.6*df['WC']**2*df['WT'])) + rng.normal(0, 0.28, len(df))
        out = ROOT/'resultados'/'_pruebas'/'C-001_sintetica'
    else:
        out = ROOT/'resultados'/'C-001_analisis_A'
    train = df[['WT', 'WC', 'EF', 'T']].to_dict('list')
    cfgs = []
    for conf, multi, base in (('D', False, 1000), ('B', True, 2000)):
        for c in range(1, 9 if not prueba else 3):
            cfgs.append(dict(nombre=f'{conf}_cadena{c}', out=str(out), seed=base+c, steps=pasos,
                             variables=['WT', 'WC'], multi=multi, prior='nv2.np6', npar=6,
                             train=train, test=None, burn=pasos//2, thin=max(1, pasos//200)))
    planificar(cfgs, nproc, max_horas=12.0)
