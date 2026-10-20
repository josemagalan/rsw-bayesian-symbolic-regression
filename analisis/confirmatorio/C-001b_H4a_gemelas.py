"""C-001b (CONFIRMATORIO, protocolo §6 H4a): contraste sin modelo del efecto de EF con las 121 parejas gemelas.
d = TSLBC(1500 N) − TSLBC(1000 N) por celda. Test t pareado y Wilcoxon (bilaterales, α = 0.05).
Descriptivo: dependencia de d con WT y WC (Spearman; medias por nivel).
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-001b_H4a_gemelas.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA

df = cargar_datos()
p = df.pivot_table(index='celda', columns='EF', values='T')
g = df.groupby('celda')[['WT', 'WC', 'cyc', 'nivel']].mean()
d = (p[1.5] - p[1.0]).rename('d'); tab = g.join(d)
assert len(tab) == 121 and tab['d'].notna().all()
t = stats.ttest_1samp(tab['d'], 0.0)
w = stats.wilcoxon(tab['d'], zero_method='wilcox')
ci = stats.t.interval(0.95, len(tab)-1, loc=tab['d'].mean(), scale=stats.sem(tab['d']))
res = dict(
    n_celdas=len(tab), media_d=float(tab['d'].mean()), mediana_d=float(tab['d'].median()),
    dt_d=float(tab['d'].std(ddof=1)), ic95_media=[float(ci[0]), float(ci[1])],
    t=float(t.statistic), p_t=float(t.pvalue), W=float(w.statistic), p_wilcoxon=float(w.pvalue),
    H4a_significativo=bool(t.pvalue < 0.05 or w.pvalue < 0.05),
    descriptivo=dict(
        spearman_d_WT=[float(x) for x in stats.spearmanr(tab['WT'], tab['d'])],
        spearman_d_WC=[float(x) for x in stats.spearmanr(tab['WC'], tab['d'])],
        media_d_por_ciclos=tab.groupby('cyc')['d'].mean().round(3).to_dict(),
        media_d_por_nivel_WC=tab.groupby('nivel')['d'].mean().round(3).to_dict()),
    var_d_mitad=float(tab['d'].var(ddof=1)/2), protocolo_sha=PROTOCOLO_SHA)
out = ROOT/'resultados'/'C-001_analisis_A'; out.mkdir(parents=True, exist_ok=True)
(out/'H4a_gemelas.json').write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
tab.to_csv(out/'H4a_diferencias_gemelas.csv')
print(json.dumps({k: v for k, v in res.items() if k != 'descriptivo'}, indent=1))
