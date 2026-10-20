"""C-000 (CONFIRMATORIO, protocolo §2B): genera UNA VEZ las 25 particiones (5-fold × 5 repeticiones)
agrupadas por celda y estratificadas por quintiles de la TSLBC media de la celda. No sobrescribe.
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-000_particiones.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from bms_comun import cargar_datos, ROOT

dst = ROOT/'analisis'/'datos_derivados'/'folds_5x5_agrupado.csv'
if dst.exists():
    sys.exit(f'Ya existe {dst}: no se regenera (protocolo: se generan una sola vez).')
df = cargar_datos()
filas = []
for rep in range(5):
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=20261016 + rep)
    for k, (_, te) in enumerate(skf.split(df, df['estrato'], df['celda'])):
        for i in te:
            filas.append(dict(id=int(df.loc[i, 'id']), celda=int(df.loc[i, 'celda']), repeticion=rep, particion=k))
f = pd.DataFrame(filas)
# comprobaciones: cada id aparece una vez por repetición; las gemelas van juntas
assert (f.groupby('repeticion')['id'].nunique() == len(df)).all()
assert (f.groupby(['repeticion', 'celda'])['particion'].nunique() == 1).all()
dst.parent.mkdir(parents=True, exist_ok=True)
f.sort_values(['repeticion', 'particion', 'id']).to_csv(dst, index=False)
print(f'Escrito {dst}'); print(f.groupby(['repeticion', 'particion']).size().unstack())
