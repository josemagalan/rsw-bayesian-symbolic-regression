"""E-019 (exploratorio, descriptivo): resumen homogéneo de tiempos de pared ya guardados.

No reajusta ningún modelo. Lee las salidas de C-002 (PySR), C-003 (BMS) y C-004 (comparadores)
y resume el tiempo de PARED por ajuste (una partición) y total sobre las 25 particiones.
Motivo: C-005 mezcla tiempo de pared (BMS) con time.process_time() del proceso padre (comparadores),
que no cuenta los procesos trabajadores de joblib (GBM con n_jobs=-1). Ver D-016.
"""
import json, platform, sys
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[2]
OUT = R/'resultados'/'2026-10-19_E-019'; OUT.mkdir(parents=True, exist_ok=True)

filas = []
c4 = pd.read_csv(R/'resultados'/'C-004_comparadores'/'mse_por_particion.csv').fillna({'fallo': ''})
for r in c4.itertuples():
    filas.append(dict(modelo=r.modelo, particion=r.particion, unidad='ajuste', reloj_s=r.reloj_s, cpu_proceso_s=r.cpu_s))
for f in sorted((R/'resultados'/'C-002_pysr_cv').glob('p[0-9][0-9].json')):
    d = json.load(open(f)); filas.append(dict(modelo='PySR', particion=int(f.stem[1:]), unidad='ajuste',
                                              reloj_s=d.get('reloj_s'), cpu_proceso_s=d.get('cpu_s')))
for f in sorted((R/'resultados'/'C-003_bms_cv').glob('p[0-9][0-9]_cadena[0-9].json')):
    d = json.load(open(f)); filas.append(dict(modelo='BMS', particion=int(f.stem[1:3]), unidad='cadena',
                                              reloj_s=d.get('tiempo_s'), cpu_proceso_s=None))
T = pd.DataFrame(filas); T.to_csv(OUT/'tiempos_por_ajuste.csv', index=False)

porpart = T.groupby(['modelo', 'particion']).agg(reloj_s=('reloj_s', 'sum'), n=('reloj_s', 'size')).reset_index()
res = porpart.groupby('modelo').agg(particiones=('particion', 'nunique'), unidades=('n', 'sum'),
                                    reloj_mediana_por_particion_s=('reloj_s', 'median'),
                                    reloj_total_h=('reloj_s', lambda x: x.sum()/3600)).reset_index()
cpu = T.groupby('modelo').cpu_proceso_s.sum(min_count=1)/3600
res['cpu_proceso_padre_total_h'] = res.modelo.map(cpu)
res = res.sort_values('reloj_total_h', ascending=False)
res.to_csv(OUT/'resumen_tiempos.csv', index=False)
print(res.to_string(index=False))
with open(OUT/'entorno.txt', 'w') as fh:
    fh.write(f'python {sys.version}\nplataforma {platform.platform()}\npandas {pd.__version__}\n')
