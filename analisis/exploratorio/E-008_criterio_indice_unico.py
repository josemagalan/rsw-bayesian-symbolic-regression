"""E-008 (EXPLORATORIO, calibración de H3b sin datos reales). 2026-10-16.
Criterio de "índice único": ¿depende f(WT, WC) solo de z = log WC + γ·log WT?
Si f = g(z): ∂f/∂lnWT = γ·∂f/∂lnWC en todos los puntos. Con las 121 celdas nominales del diseño:
  γ̂ = Σ(∂T·∂C)/Σ(∂C²);  R²_ui = 1 − Σ(∂T − γ̂·∂C)² / Σ(∂T²);  peso de WT: p_T = Σ∂T² / (Σ∂T² + Σ∂C²).
Controles positivos (índice único exacto) y negativos (no lo son), más los modelos MDL de los pilotos E-005/E-006.
Uso: python analisis/exploratorio/E-008_criterio_indice_unico.py
"""
import json, glob, re
from pathlib import Path
import numpy as np, sympy as sp

ROOT = Path(__file__).resolve().parents[2]
WT = np.arange(2, 13) * 0.02
WC = np.linspace(1.5, 6.5, 11)            # niveles nominales
G_WT, G_WC = np.meshgrid(WT, WC, indexing='ij')


def criterio(f, h=1e-4):
    lt, lc = np.log(G_WT), np.log(G_WC)
    dT = (f(np.exp(lt + h), G_WC) - f(np.exp(lt - h), G_WC)) / (2*h)
    dC = (f(G_WT, np.exp(lc + h)) - f(G_WT, np.exp(lc - h))) / (2*h)
    dT, dC = np.ravel(dT), np.ravel(dC)
    ok = np.isfinite(dT) & np.isfinite(dC)
    dT, dC = dT[ok], dC[ok]
    sC, sT = np.sum(dC**2), np.sum(dT**2)
    g = np.sum(dT*dC) / sC
    r2 = 1 - np.sum((dT - g*dC)**2) / sT if sT > 0 else np.nan
    return dict(gamma=g, R2_ui=r2, peso_WT=sT/(sT+sC), n=int(ok.sum()))


def eq6(wt, wc):   # Martín et al. 2016, ec. (6), WT en ciclos, EF = 1.25 kN
    c = wt/0.02; ef = 1.25
    return (-1.93914 + 0.34904*c - 0.02266*c**2 + 1.95763*wc - 0.14341*wc**2 - 0.36273*ef
            + 0.01260*c*wc + 0.03211*c*ef + 0.01422*wc*ef)

controles = {
    '+ T1 potencias (γ=0.25)': lambda t, c: 7.3*(1-np.exp(-0.342*c**1.38*t**0.345)),
    '+ T3 Joule (γ=0.5)': lambda t, c: 7.0*(1-np.exp(-0.6*c**2*t)),
    '+ tanh(WC²√WT)': lambda t, c: 7.0*np.tanh((c**2*np.sqrt(t)/7)**0.6),
    '~ T2 potencias+expulsión': lambda t, c: 7.3*(1-np.exp(-0.342*c**1.38*t**0.345)) - 0.005*(c**2*t)**2,
    '− producto no potencial': lambda t, c: 7*np.tanh(c/3)*(1-np.exp(-t/0.08)),
    '− aditivo': lambda t, c: 3.5*np.tanh(c/3) + 3.5*(1-np.exp(-t/0.08)),
    '− cuadrático ec.(6) 2016': eq6,
    '− solo WC (degenerado)': lambda t, c: 7*np.tanh(c/3) + 0*t,
}
print(f"{'función':32s} {'γ̂':>7s} {'R²_ui':>8s} {'peso_WT':>8s}")
res = {}
for k, f in controles.items():
    r = criterio(f); res[k] = r
    print(f"{k:32s} {r['gamma']:7.3f} {r['R2_ui']:8.4f} {r['peso_WT']:8.3f}")

# Modelos MDL de los pilotos (opción D: un solo conjunto; opción B: se usa EF1000)
print('\nModelos MDL de los pilotos (opciones B y D):')
WTs, WCs = sp.symbols('WT WC')
filas = []
for fjs in sorted(glob.glob(str(ROOT/'resultados'/'E-00[56]*'/'*.json'))):
    if fjs.endswith('progreso.json'):
        continue
    d = json.load(open(fjs, encoding='utf-8'))
    if d['cfg']['option'] not in ('B', 'D'):
        continue
    pv = d['parametros_mdl']; pv = pv.get('d0') or pv.get('EF1000')
    ex = sp.sympify(d['modelo_mdl'], locals={'WT': WTs, 'WC': WCs})
    ex = ex.subs({sp.Symbol(p): v for p, v in pv.items()})
    fl = sp.lambdify((WTs, WCs), ex, 'numpy')
    f = lambda t, c, fl=fl: np.real_if_close(np.asarray(fl(t, c), dtype=complex)) * np.ones_like(t)
    with np.errstate(all='ignore'):
        r = criterio(lambda t, c: np.asarray(f(t, c), dtype=float))
    filas.append((d['nombre'], r))
    print(f"  {d['nombre']:30s} γ̂={r['gamma']:7.3f} R²_ui={r['R2_ui']:8.4f} peso_WT={r['peso_WT']:.3f}  {d['modelo_mdl'][:60]}")
out = ROOT/'resultados'/'2026-10-16_E-008'
out.mkdir(parents=True, exist_ok=True)
(out/'resultados.json').write_text(json.dumps(dict(controles=res, pilotos={n: r for n, r in filas}), indent=1, default=float), encoding='utf-8')
