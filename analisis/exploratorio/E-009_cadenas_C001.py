"""E-009 (EXPLORATORIO, post hoc sobre C-001): descripción de los MDL de TODAS las cadenas de D (no solo el global):
ajuste en muestra, criterio de índice único y comportamiento en los límites. No cambia ningún resultado confirmatorio."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, sympy as sp
from bms_comun import cargar_datos, ROOT
D = ROOT/'resultados'/'C-001_analisis_A'
df = cargar_datos(); y = df['T'].values
WT_s, WC_s = sp.symbols('WT WC')
WTn = np.arange(2, 13)*0.02; WCn = np.linspace(1.5, 6.5, 11); GT, GC = np.meshgrid(WTn, WCn, indexing='ij')
out = []
for f in sorted(D.glob('*_cadena*.json')):
    if f.name.endswith('progreso.json'): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    for ds, pv in d['parametros_mdl'].items():
        ex = sp.sympify(d['modelo_mdl'], locals={'WT': WT_s, 'WC': WC_s}).subs({sp.Symbol(k): v for k, v in pv.items()})
        fl = sp.lambdify((WT_s, WC_s), ex, 'numpy')
        F = lambda t, c: np.asarray(fl(t, c), dtype=float)*np.ones_like(np.asarray(t, dtype=float))
        with np.errstate(all='ignore'):
            if ds == 'd0':
                mse = float(np.mean((F(df.WT.values, df.WC.values)-y)**2))
            else:
                m = (df.EF == (1.0 if ds == 'EF1000' else 1.5)).values
                mse = float(np.mean((F(df.WT.values[m], df.WC.values[m])-y[m])**2))
            h = 1e-4
            dT = np.ravel((F(GT*np.exp(h), GC)-F(GT*np.exp(-h), GC))/(2*h)); dC = np.ravel((F(GT, GC*np.exp(h))-F(GT, GC*np.exp(-h)))/(2*h))
            g = np.sum(dT*dC)/np.sum(dC**2); r2 = 1-np.sum((dT-g*dC)**2)/np.sum(dT**2)
            lim = [float(F(np.array([0.005]), np.array([4.0]))[0]), float(F(np.array([0.12]), np.array([0.3]))[0]),
                   float(F(np.array([0.5]), np.array([4.0]))[0]), float(F(np.array([0.12]), np.array([9.0]))[0])]
        out.append(dict(cadena=f.stem, conjunto=ds, dl=round(d['dl_mdl'], 2), mse_muestra=round(mse, 4), gamma=round(float(g), 3),
                        R2_ui=round(float(r2), 3), f_WT0005=round(lim[0], 2), f_WC03=round(lim[1], 2), f_WT05=round(lim[2], 2), f_WC9=round(lim[3], 2),
                        expr=str(sp.nsimplify(ex, tolerance=1e-3, rational=False))[:110]))
        if ds == 'EF1000' or ds == 'd0':
            pass
for r in sorted(out, key=lambda r: (r['cadena'][0], r['dl'])):
    print(r)
dst = ROOT/'resultados'/'2026-10-17_E-009'; dst.mkdir(parents=True, exist_ok=True)
(dst/'resultados.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
