"""C-000b (protocolo §9 + D-015): genera UNA VEZ las particiones de las sensibilidades. No sobrescribe.
  - folds_5x5_aleatorio.csv: 5-fold × 5 repeticiones SIN agrupar (las gemelas pueden separarse), estratificadas por
    quintiles de la TSLBC media de la celda (como las agrupadas), semillas 20261016 + repetición.
  - particiones_extrapolacion.csv: 'WT_extremos' (prueba = 2 y 12 ciclos) y 'WC_extremos' (prueba = niveles 1 y 11).
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-000b_particiones_sensibilidad.py   [--prueba: escribe en resultados\\_pruebas]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from bms_comun import cargar_datos, ROOT

D = ROOT/'resultados'/'_pruebas'/'datos_derivados' if '--prueba' in sys.argv else ROOT/'analisis'/'datos_derivados'
D.mkdir(parents=True, exist_ok=True)
df = cargar_datos()
dst = D/'folds_5x5_aleatorio.csv'
if dst.exists():
    print(f'Ya existe {dst}: no se regenera.')
else:
    filas = []
    for rep in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=20261016 + rep)
        for k, (_, te) in enumerate(skf.split(df, df['estrato'])):
            filas += [dict(id=int(df.loc[i, 'id']), celda=int(df.loc[i, 'celda']), repeticion=rep, particion=k) for i in te]
    f = pd.DataFrame(filas)
    assert (f.groupby('repeticion')['id'].nunique() == len(df)).all()
    f.sort_values(['repeticion', 'particion', 'id']).to_csv(dst, index=False)
    sep = (f.groupby(['repeticion', 'celda'])['particion'].nunique() > 1).mean()
    print(f'Escrito {dst}; fracción de celdas con las gemelas separadas: {sep:.2f}')
dst = D/'particiones_extrapolacion.csv'
if dst.exists():
    print(f'Ya existe {dst}: no se regenera.')
else:
    filas = []
    for div, te in (('WT_extremos', df.cyc.isin([2, 12])), ('WC_extremos', df.nivel.isin([1, 11]))):
        filas += [dict(id=int(i), division=div, conjunto='prueba' if t else 'entrenamiento') for i, t in zip(df.id, te)]
    f = pd.DataFrame(filas); f.to_csv(dst, index=False)
    print(f'Escrito {dst}'); print(f.groupby(['division', 'conjunto']).size())
