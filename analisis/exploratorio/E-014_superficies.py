"""E-014 (EXPLORATORIO, propuesta de JMG 2026-10-19): superficies TSLBC(WT, WC) de todos los métodos, ajustados con los
242 puntos, en una rejilla que excede el dominio ensayado (WT 0.02–0.30 s; WC 1.0–7.5 kA) para ver cómo extrapolan.
Es una ilustración, no una medida de error (los errores están en C-005, C-009 y E-013).
Métodos (ajustados con todos los datos):
  - comparadores 2–6, 8 y 9 con las funciones de C-004 (cargadas de su fuente vía C-008; CV interna agrupada por celda,
    semilla 20261016); PySR con correr() de C-002 (random_state 20261016 + 99);
  - BMS MDL del análisis A (cadena de D con menor DL, C-001), con sus parámetros guardados;
  - BMS, mediana de los 25 MDL de la CV (C-003, cadena de menor DL de cada partición, con sus parámetros guardados): muestra
    cuánto coinciden las estructuras al extrapolar. NO es el promedio del posterior: las muestras MCMC no guardan sus
    parámetros y reajustarlos no reproduce su energía (prueba en la VM: con 50 arranques, solo el 30 % de las muestras de A
    recuperan su DL registrada; la mediana de la diferencia es +23), así que ese promedio no se puede reconstruir fielmente;
  - forma compacta (a − b/WT)·ln(WC + WT) (E-010).
EF: media de las predicciones con 1000 y 1500 N (sin efecto detectable, A-004). Valores fuera de [0, 1.2·máx(y)] se
marcan como "fuera de rango" (textura) en la figura.
Salida: resultados/2026-10-19_E-014/ (superficies.csv, resumen.csv, Fig_superficies.pdf/png, pysr/).
Uso: .\\.venv\\Scripts\\python.exe analisis\\exploratorio\\E-014_superficies.py   [--sin-tabpfn] [--sin-pysr] (solo pruebas)"""
import importlib.util, json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, sympy as sp
from sklearn.model_selection import GroupKFold
from scipy.optimize import curve_fit
from bms_comun import cargar_datos, ROOT
warnings.filterwarnings('ignore')

CONF = ROOT/'analisis'/'confirmatorio'
OUT = ROOT/'resultados'/('_pruebas/E-014_prueba' if '--prueba' in sys.argv else '2026-10-19_E-014')
OUT.mkdir(parents=True, exist_ok=True)


def modulo(nombre, fichero):
    spec = importlib.util.spec_from_file_location(nombre, CONF/fichero)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


c008 = modulo('c008', 'C-008_sensibilidad_comparadores.py')
df = cargar_datos(); y = df['T'].values; YMAX = 1.2*y.max()
GWT = np.round(np.arange(0.02, 0.3001, 0.005), 4); GWC = np.round(np.arange(1.0, 7.5001, 0.1), 3)
TT, CC = np.meshgrid(GWT, GWC, indexing='ij')
DOM = dict(WT=(0.04, 0.24), WC=(df.WC.min(), df.WC.max()))
SUP = {}


def rejilla_ef(ef):
    return np.column_stack([TT.ravel(), CC.ravel(), np.full(TT.size, ef)])


def media_ef(fn):
    """fn(X con columnas WT, WC, EF) -> predicción; media de EF = 1.0 y 1.5 kN."""
    return 0.5*(np.asarray(fn(rejilla_ef(1.0)), float) + np.asarray(fn(rejilla_ef(1.5)), float)).reshape(TT.shape)


if __name__ == '__main__':
    t0 = time.time()
    # ---- comparadores (C-004)
    modelos, _ = c008.cargar_modelos_c004()
    quitar = {'media'} | ({'TabPFN_v2'} if '--sin-tabpfn' in sys.argv else set())
    Xtr = df[['WT', 'WC', 'EF']].values
    cv = list(GroupKFold(5, shuffle=True, random_state=20261016).split(Xtr, y, df['celda'].values))
    for nombre, fn in modelos:
        if nombre in quitar:
            continue
        # se ajusta una vez y se predice la rejilla de los dos niveles de EF en la misma llamada
        Xg = np.vstack([rejilla_ef(1.0), rejilla_ef(1.5)])
        pred, info = fn(Xtr, y, Xg, cv, 20261016, df)
        pred = np.asarray(pred, float); m = TT.size
        SUP[nombre] = (0.5*(pred[:m] + pred[m:])).reshape(TT.shape)
        print(f'{nombre}: {info}', flush=True)
    # ---- PySR (C-002)
    if '--sin-pysr' not in sys.argv:
        c002 = modulo('c002', 'C-002_pysr_cv.py')
        (OUT/'pysr').mkdir(exist_ok=True)
        Xg = np.column_stack([TT.ravel(), CC.ravel()])
        cfg = dict(p=99, rep=-1, k=-1, out=str(OUT/'pysr'), niter=200, Xtr=df[['WT', 'WC']].values.tolist(), ytr=y.tolist(),
                   Xte=Xg.tolist(), yte=[0.0]*len(Xg), ids_test=list(range(len(Xg))))
        print('PySR:', c002.correr(cfg), flush=True)
        r = json.loads((OUT/'pysr'/'p99.json').read_text(encoding='utf-8'))
        SUP['PySR'] = np.array(r['pred'], float).reshape(TT.shape)
    # ---- BMS: MDL de A y promedio del posterior de A
    cad = [json.loads(Path(f).read_text(encoding='utf-8')) for f in sorted((ROOT/'resultados'/'C-001_analisis_A').glob('D_cadena?.json'))]
    mA = min(cad, key=lambda d: d['dl_mdl'])
    WT_s, WC_s = sp.symbols('WT WC'); ex = sp.sympify(mA['modelo_mdl'])
    us = sorted(str(s) for s in ex.free_symbols if str(s) not in ('WT', 'WC'))
    fA = sp.lambdify([WT_s, WC_s] + [sp.Symbol(u) for u in us], ex, 'numpy')
    with np.errstate(all='ignore'):
        SUP['BMS_MDL'] = np.asarray(fA(TT, CC, *[mA['parametros_mdl']['d0'][u] for u in us]), float)*np.ones(TT.shape)
    capas = []
    for p in range(25):
        cs = [json.loads(Path(f).read_text(encoding='utf-8'))
              for f in sorted((ROOT/'resultados'/'C-003_bms_cv').glob(f'p{p:02d}_cadena?.json'))]
        if len(cs) < 2:
            continue
        d = min(cs, key=lambda d: d['dl_mdl']); e = sp.sympify(d['modelo_mdl'])
        u = sorted(str(s_) for s_ in e.free_symbols if str(s_) not in ('WT', 'WC'))
        g = sp.lambdify([WT_s, WC_s] + [sp.Symbol(x) for x in u], e, 'numpy')
        with np.errstate(all='ignore'):
            z = np.asarray(g(TT, CC, *[d['parametros_mdl']['d0'][x] for x in u]), dtype=complex)*np.ones(TT.shape)
        capas.append(np.where(np.abs(z.imag) < 1e-12, z.real, np.nan))
    capas = np.array(capas)
    SUP['BMS_MDL_CV_mediana'] = np.nanmedian(capas, axis=0)
    # dispersión entre los 25 MDL (rango intercuartílico), para el resumen
    RIC_CV = np.nanpercentile(capas, 75, axis=0) - np.nanpercentile(capas, 25, axis=0)
    print(f'MDL de la CV: {len(capas)} superficies', flush=True)
    # ---- forma compacta (E-010)
    fC = lambda X, a, b: (a - b/X[0])*np.log(X[1] + X[0])
    thC, _ = curve_fit(fC, np.vstack([df.WT, df.WC]), y, p0=[3.9, 0.05])
    with np.errstate(all='ignore'):
        SUP['compacta'] = fC(np.vstack([TT.ravel(), CC.ravel()]), *thC).reshape(TT.shape)

    # ---- tablas
    filas = []
    dentro = (TT >= DOM['WT'][0]) & (TT <= DOM['WT'][1]) & (CC >= DOM['WC'][0]) & (CC <= DOM['WC'][1])
    for m, Z in SUP.items():
        fuera_rango = (~np.isfinite(Z)) | (Z < 0) | (Z > YMAX)
        filas.append(dict(metodo=m, min=float(np.nanmin(Z)), max=float(np.nanmax(Z)),
                          frac_fuera_de_rango_dentro=float(fuera_rango[dentro].mean()),
                          frac_fuera_de_rango_fuera=float(fuera_rango[~dentro].mean()),
                          frac_aceptable_dentro=float((Z[dentro] >= 5.93).mean())))
    filas.append(dict(metodo='RIC entre los 25 MDL de la CV', min=float(np.nanmin(RIC_CV)), max=float(np.nanmax(RIC_CV)),
                      ric_mediano_dentro=float(np.nanmedian(RIC_CV[dentro])), ric_mediano_fuera=float(np.nanmedian(RIC_CV[~dentro]))))
    pd.DataFrame(filas).to_csv(OUT/'resumen.csv', index=False)
    pd.DataFrame({'WT': TT.ravel(), 'WC': CC.ravel(), **{m: Z.ravel() for m, Z in SUP.items()}}).to_csv(OUT/'superficies.csv', index=False)
    (OUT/'entorno.json').write_text(json.dumps(dict(forma_compacta=list(map(float, thC)), metodos=list(SUP), segundos=time.time()-t0,
                                                    n_mdl_cv=len(capas))), encoding='utf-8')
    print(pd.DataFrame(filas).round(3).to_string(index=False), flush=True)

    # ---- figura
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    TINTA, TINTA2, GRIS_CLARO = '#0b0b0b', '#52514e', '#c9c8c2'
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'axes.edgecolor': GRIS_CLARO, 'xtick.color': TINTA2, 'ytick.color': TINTA2, 'axes.labelcolor': TINTA2,
                         'pdf.fonttype': 42, 'savefig.dpi': 300, 'hatch.color': GRIS_CLARO, 'hatch.linewidth': 0.6})
    cmap = LinearSegmentedColormap.from_list('azul', ['#e8f1fd', '#b7d3f6', '#6da7ec', '#2a78d6', '#1c5cab', '#0d366b'])
    VMAX = 7.5
    paneles = [('datos', 'Data (cell means)'), ('BMS_MDL', 'BMS MDL (all data)'), ('BMS_MDL_CV_mediana', 'BMS CV-MDL median'),
               ('compacta', 'Compact (2 par.)'), ('PySR', 'PySR'), ('cuadratico_EN', 'Quadratic'),
               ('GAM', 'GAM'), ('EBM', 'EBM'), ('GBM', 'GBM'), ('TabPFN_v2', 'TabPFN v2')]
    paneles = [p for p in paneles if p[0] == 'datos' or p[0] in SUP]
    nc = 5; nr = int(np.ceil(len(paneles)/nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.2, 1.75*nr + 0.5), sharex=True, sharey=True)
    ext_x, ext_y = (GWC[0] - 0.05, GWC[-1] + 0.05), (GWT[0] - 0.0025, GWT[-1] + 0.0025)
    for ax, (clave, titulo) in zip(axs.ravel(), paneles):
        if clave == 'datos':
            cel = df.groupby('celda').agg(WT=('WT', 'mean'), WC=('WC', 'mean'), T=('T', 'mean'))
            im = ax.scatter(cel.WC, cel.WT, c=cel['T'], cmap=cmap, vmin=0, vmax=VMAX, marker='s', s=11, lw=0, zorder=2)
        else:
            Z = SUP[clave].copy(); Z[(~np.isfinite(Z)) | (Z < 0) | (Z > YMAX)] = np.nan
            # fuera del dominio, el mismo color atenuado; dentro, a pleno
            ax.pcolormesh(GWC, GWT, Z, cmap=cmap, vmin=0, vmax=VMAX, shading='nearest', alpha=0.45, zorder=1, rasterized=True)
            Zd = np.where(dentro, Z, np.nan)
            malo = np.isnan(Z)
            if malo.any():      # fuera de rango: textura solo donde la predicción es absurda
                ax.contourf(GWC, GWT, malo.astype(float), levels=[0.5, 1.5], colors='none', hatches=['/////'], zorder=2)
            im = ax.pcolormesh(GWC, GWT, Zd, cmap=cmap, vmin=0, vmax=VMAX, shading='nearest', zorder=1.5, rasterized=True)
            try:
                cs = ax.contour(GWC, GWT, np.where(np.isnan(Z), -1, Z), levels=[5.93], colors=TINTA, linewidths=1.0, zorder=3)
            except Exception:
                pass
        ax.add_patch(Rectangle((DOM['WC'][0], DOM['WT'][0]), DOM['WC'][1] - DOM['WC'][0], DOM['WT'][1] - DOM['WT'][0],
                               fill=False, ec=TINTA2, lw=0.8, ls=(0, (3, 2)), zorder=4))
        ax.set_xlim(*ext_x); ax.set_ylim(*ext_y); ax.set_title(titulo, loc='left', fontsize=7.2)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axs.ravel()[len(paneles):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('WC (kA)')
    for ax in axs[:, 0]:
        ax.set_ylabel('WT (s)')
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.015, extend='max'); cb.set_label('Predicted TSLBC (kN)'); cb.outline.set_visible(False)
    fig.text(0.01, -0.01, 'Dashed box: tested domain; outside it colours are lighter (extrapolation). Black line: 5.93 kN acceptance '
             'contour.\nHatched: prediction outside [0, 1.2·max(y)]. Each method fitted to all 242 joints; EF averaged.',
             fontsize=6.3, color=TINTA2, ha='left', va='top')
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'Fig_superficies.{ext}', bbox_inches='tight')
    print(f'Hecho en {time.time() - t0:.0f} s', flush=True)
