"""C-010 (CONFIRMATORIO, protocolo §7, parches D-008 v3): reevaluación fuera de línea de las expresiones que agotaron
el límite de 5 s de sympify durante las cadenas del BMS. Pregunta (§7): ¿la DL de alguna competía con el MDL?
Fuentes: C-001 (análisis A) y C-003 (CV); C-007 (sensibilidad) como complemento. Se leen los `expresiones_timeout`.

Dos tipos de registro (se distinguen por el formato de la cadena de texto, ver mcmc.py del BMS):
  - CON espacios = str(árbol): lo sympifican get_sse y bms_parches._tiene_imaginaria. Si agota el límite, la expresión
    se RECHAZÓ en la cadena (SSE = ∞ o "compleja"): su DL no llegó a verse. Son las que importan para §7.
  - SIN espacios = str(árbol) sin espacios: lo sympifica canonical(). Si agota el límite, canonical() usa la forma
    completa y la energía se calculó con normalidad: solo afecta a la etiqueta canónica, no a la DL.
Se agrupan por expresión única (sin espacios) dentro de cada cadena.

Procedimiento para cada expresión única:
  1. DL sin sympy: la cadena se interpreta con `ast` (la sintaxis del BMS es Python válido) y se evalúa con numpy;
     el árbol de operaciones da los recuentos del prior (`**` con exponente literal 2/3 = pow2/pow3, menos unario = '-').
     Ajuste con curve_fit como el BMS (maxfev 10 000) desde todos los parámetros a 1 (como el BMS) y desde 1000 arranques
     aleatorios (signo ±, módulo 10^U(-1.5, 1.5); semilla 20261019 + índice); se queda la menor DL (criterio favorable a
     la expresión) y se registra la mejor tras 20, 200 y 1000 arranques para ver la convergencia. (En la prueba con 20
     arranques, el MDL de C-001 reajustado desde cero no llegaba a su DL registrada; con 200 sí.) DL = E = BIC/2 + E_prior con la fórmula del BMS: BIC = (k − n)·ln n + n·(ln 2π + ln SSE + 1), k = 1 + nº
     de parámetros distintos. Así no hace falta ningún límite de tiempo.
  2. Literal del protocolo: sympify de la cadena registrada con límite de 10 min de CPU (proceso nuevo); se anota si
     termina y cuánto tarda.
  3. Control del procedimiento: el MDL de cada partición afectada se evalúa igual en sus parámetros guardados y debe
     reproducir la DL registrada; también se reajusta desde cero.
  Cota sin ajuste: con la E_prior de la expresión, se calcula el MSE de entrenamiento que necesitaría para igualar la DL
  del MDL (despejando la fórmula del BIC) y se compara con el ruido estimado en C-006 (σ² ≈ 0.066 kN²). Si hace falta un
  MSE por debajo del ruido, competir exigiría sobreajustar el ruido.
Se compara con la DL del MDL de la partición (mínimo entre sus cadenas completas): "compite" si ΔDL < 0. Si alguna
compite, se informa del MSE de prueba que habría tenido (descriptivo; no se sustituye nada).
Uso: .\\.venv\\Scripts\\python.exe analisis\\confirmatorio\\C-010_reevaluar_timeouts.py [procesos=6]
     ... --sin-sympify   (omite el paso 2; solo para pruebas)   ... --prueba (salida en resultados\\_pruebas\\C-010_prueba)
Aviso: alguna expresión parecida ocupó ~20 GB de RAM en sympify (E-006); con muchos procesos puede agotarse la memoria."""
import ast, collections, glob, json, re, sys, time, platform, traceback, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.special import factorial
from bms_comun import cargar_datos, ROOT, PROTOCOLO_SHA, PRIORS, _importar_bms

PRUEBA = '--prueba' in sys.argv
SIN_SYMPIFY = '--sin-sympify' in sys.argv
OUT = ROOT/'resultados'/('_pruebas/C-010_prueba' if PRUEBA else 'C-010_timeouts')
LIMITE_SYMPIFY_S = 600.0
N_ARRANQUES = 1000
SIGMA2_RUIDO = 0.066          # kN², ruido estimado en C-006 (Gasser 0.255² ≈ gemelas 0.261²)
CORTES = (20, 200, 1000)      # se registra la mejor DL tras 20, 200 y 1000 arranques (convergencia)
UNARIOS = {'sin', 'cos', 'tan', 'exp', 'log', 'sinh', 'cosh', 'tanh', 'abs', 'sqrt', 'fac'}
BINARIOS = {ast.Add: '+', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'}
NS = dict(sin=np.sin, cos=np.cos, tan=np.tan, exp=np.exp, log=np.log, sinh=np.sinh, cosh=np.cosh, tanh=np.tanh,
          abs=np.abs, sqrt=np.sqrt, fac=factorial)


def leer_prior(nombre):
    """Idéntico a fit_prior.read_prior_par del BMS (primera línea: nombres; última: valores; se salta el primer campo),
    con el mismo fichero que elige bms_comun (next(glob))."""
    lineas = next(PRIORS.glob(f'*{nombre}.*.dat')).read_text().splitlines()
    return dict(zip(lineas[0].strip().split()[1:], [float(v) for v in lineas[-1].strip().split()[1:]]))


def recuentos(expr):
    arbol = ast.parse(expr.replace(' ', ''), mode='eval').body
    nops, hojas = collections.Counter(), set()

    def rec(n):
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Pow) and isinstance(n.right, ast.Constant) and n.right.value in (2, 3):
                nops[f'pow{n.right.value}'] += 1; rec(n.left); return
            if type(n.op) not in BINARIOS:
                raise ValueError(f'operador no BMS: {ast.dump(n.op)}')
            nops[BINARIOS[type(n.op)]] += 1; rec(n.left); rec(n.right)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            nops['-'] += 1; rec(n.operand)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in UNARIOS and len(n.args) == 1:
            nops[n.func.id] += 1; rec(n.args[0])
        elif isinstance(n, ast.Name):
            hojas.add(n.id)
        else:
            raise ValueError(f'nodo no BMS: {ast.dump(n)[:80]}')
    rec(arbol)
    return nops, sorted(h for h in hojas if h.startswith('_')), sorted(h for h in hojas if not h.startswith('_'))


def energia_prior(nops, prior):
    return sum(prior.get(f'Nopi_{o}', 0.0)*k + prior.get(f'Nopi2_{o}', 0.0)*k**2 for o, k in nops.items())


def compilar(expr, variables, params):
    return eval(f"lambda {', '.join(variables + params)}: {expr.replace(' ', '')}", dict(NS))


def sse_de(f, X, y, theta):
    with np.errstate(all='ignore'):
        try:
            q = np.asarray(f(*X, *theta), dtype=float)*np.ones(len(y))
        except Exception:
            return np.inf
        e = np.square(y - q)
    return float(np.sum(e)) if not np.isnan(e).any() else np.inf


def dl_desde(sse, n, k, ep):
    if not np.isfinite(sse) or sse <= 0:
        return np.inf
    return ((k - n)*np.log(n) + n*(np.log(2*np.pi) + np.log(sse) + 1))/2 + ep


def evaluar(expr, variables_cfg, X_df, y, prior, semilla, theta_fijo=None):
    nops, params, vars_usadas = recuentos(expr)
    vs = [v for v in variables_cfg if v in vars_usadas]
    f = compilar(expr, vs, params)
    X = [X_df[v].values for v in vs]; n = len(y); k = 1 + len(params); ep = energia_prior(nops, prior)
    if theta_fijo is not None:
        th = [theta_fijo[p.strip('_')] if p.strip('_') in theta_fijo else theta_fijo.get(p, 1.0) for p in params]
        return dict(dl=dl_desde(sse_de(f, X, y, th), n, k, ep), ep=ep, n_parametros=len(params), theta=th)
    rng = np.random.default_rng(semilla)
    arranques = [np.ones(len(params))] + [rng.choice([-1, 1], len(params))*10**rng.uniform(-1.5, 1.5, len(params))
                                          for _ in range(N_ARRANQUES)]
    mejor = dict(dl=np.inf, theta=None); hitos = {}
    feval = lambda x, *th: f(*x, *th)
    for i, p0 in enumerate(arranques):
        th = p0
        if params:
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore')
                try:
                    th = curve_fit(feval, X, y, p0=list(p0), maxfev=10000)[0]
                except Exception:
                    th = p0                       # como el BMS: si el ajuste falla, se quedan los valores de partida
        dl = dl_desde(sse_de(f, X, y, th), n, k, ep)
        if dl < mejor['dl']:
            mejor = dict(dl=dl, theta=[float(v) for v in th])
        if i in CORTES:
            hitos[f'dl_{i}_arranques'] = mejor['dl']
    dl_unos = dl_desde(sse_de(f, X, y, np.ones(len(params))), n, k, ep) if params else mejor['dl']
    return dict(mejor, **hitos, dl_unos=dl_unos, ep=ep, n_parametros=len(params), nops=dict(nops), vs=vs)


def evaluar_tarea(a):
    """Envoltorio para el Pool (sin funciones compiladas en el resultado)."""
    clave, expr, variables, tr, prior, semilla = a
    t0 = time.process_time()
    try:
        r = evaluar(expr, variables, tr, tr['T'].values, prior, semilla)
        r['estado'] = 'ok' if np.isfinite(r['dl']) else 'no_ajustable'
    except Exception:
        r = dict(estado='error', traza=traceback.format_exc()[-300:])
    r['segundos_cpu_ajuste'] = time.process_time() - t0
    return clave, r


def sympify_10min(cadena):
    """Paso 2 (literal del protocolo), en un proceso nuevo."""
    t0 = time.process_time()
    try:
        mcmc, bp, *_ = _importar_bms()
        bp.TIEMPO_SYMPIFY_S = LIMITE_SYMPIFY_S
        bp.sympify_con_limite(cadena)
        estado = 'termina'
    except BaseException as e:
        estado = 'timeout_600s' if type(e).__name__ == 'TiempoSympify' else f'error: {e!r}'[:120]
    return cadena, estado, time.process_time() - t0


def recoger(df):
    fuentes = [('C-001', 'C-001_analisis_A/*_cadena*.json'), ('C-003', 'C-003_bms_cv/p??_cadena*.json'),
               ('C-007', 'C-007_bms_sensibilidad/*/*_cadena*.json')]
    grupos = {}
    for fuente, patron in fuentes:
        for f in sorted(glob.glob(str(ROOT/'resultados'/patron))):
            if f.endswith('.progreso.json'):
                continue
            p = Path(f); m = re.match(r'(.+)_cadena(\d+)(_r)?\.json$', p.name)
            g = grupos.setdefault((fuente, p.parent.name, m.group(1)), {})
            if int(m.group(2)) not in g or m.group(3):
                g[int(m.group(2))] = json.loads(p.read_text(encoding='utf-8'))
    return grupos


def datos(cfg, df):
    d = df.copy(); d['WC_nom'] = 1.5 + 0.5*(d['nivel'] - 1)
    wc = cfg.get('wc', 'WC'); d['WC'] = d[wc]
    tr = d[d.id.isin(cfg['ids_train'])] if cfg.get('ids_train') else d          # C-001: todos los datos
    te = d.set_index('id').loc[cfg['ids_test']] if cfg.get('ids_test') else None
    return tr, te


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    nproc = int(args[0]) if args else 6
    OUT.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    grupos = recoger(df)
    filas, controles, cadenas_sympify, idx = [], [], [], 0
    trabajos, contexto = [], {}
    for (fuente, carpeta, part), cs in sorted(grupos.items()):
        if not any(d.get('expresiones_timeout') for d in cs.values()):
            continue
        dl_part = min(d['dl_mdl'] for d in cs.values()); cmin = min(cs, key=lambda c: cs[c]['dl_mdl']); dm = cs[cmin]
        assert not dm['cfg']['multi'], 'C-010 solo implementa la configuración D (un conjunto de datos)'
        prior = leer_prior(dm['cfg']['prior'])
        tr, te = datos(dm['cfg'], df); y = tr['T'].values
        # 3. control: el MDL de la partición
        c_fijo = evaluar(dm['modelo_mdl'], dm['cfg']['variables'], tr, y, prior, 0, theta_fijo=dm['parametros_mdl']['d0'])
        c_ajuste = evaluar(dm['modelo_mdl'], dm['cfg']['variables'], tr, y, prior, 20261019)
        controles.append(dict(fuente=fuente, carpeta=carpeta, particion=part, cadena_mdl=cmin, dl_registrada=dm['dl_mdl'],
                              dl_parametros_guardados=c_fijo['dl'], dl_reajuste_20=c_ajuste.get('dl_20_arranques'),
                              dl_reajuste_200=c_ajuste.get('dl_200_arranques'), dl_reajuste=c_ajuste['dl'], mdl=dm['canon_mdl']))
        for c, d in sorted(cs.items()):
            regs = d.get('expresiones_timeout', [])
            assert d['cfg']['prior'] == dm['cfg']['prior'] and d['cfg'].get('ids_train') == dm['cfg'].get('ids_train')
            unicas = list(dict.fromkeys(e.replace(' ', '') for e in regs))
            for e in unicas:
                formas = [r for r in regs if r.replace(' ', '') == e]
                cadenas_sympify += formas
                fila = dict(fuente=fuente, carpeta=carpeta, particion=part, cadena=c, indice=idx,
                            rechazada_en_cadena=any(' ' in r for r in formas), formas_registradas=len(formas),
                            dl_mdl_cadena=d['dl_mdl'], dl_mdl_particion=dl_part, mdl_particion=dm['canon_mdl'], expr=e,
                            n_train=len(tr), mse_train_mdl=float(np.mean((np.array(dm['pred_mdl_train'], float) - y)**2)))
                trabajos.append((idx, e, d['cfg']['variables'], tr, prior, 20261019 + idx))
                contexto[idx] = (tr, te, dm)
                filas.append(fila); idx += 1
    from multiprocessing import Pool
    print(f'Ajuste de {len(trabajos)} expresiones únicas ({N_ARRANQUES} arranques), {nproc} procesos', flush=True)
    res = {}
    with Pool(nproc) as pool:
        for clave, r in pool.imap_unordered(evaluar_tarea, trabajos):
            res[clave] = r; print(f"  #{clave:02d} {r['estado']} DL={r.get('dl', float('nan')):.2f} {r['segundos_cpu_ajuste']:.0f} s", flush=True)
    for fila in filas:
        r = res[fila['indice']]; tr, te, dm = contexto[fila['indice']]
        fila.update({k: v for k, v in r.items() if k not in ('theta', 'vs', 'nops')})
        fila['theta'] = json.dumps(r.get('theta')); fila['nops'] = json.dumps(r.get('nops'))
        if r['estado'] == 'ok':
            fila['delta_dl'] = r['dl'] - fila['dl_mdl_particion']
            if fila['delta_dl'] < 0 and te is not None:
                f = compilar(fila['expr'], r['vs'], recuentos(fila['expr'])[1])
                q = np.asarray(f(*[te[v].values for v in r['vs']], *r['theta']), float)*np.ones(len(te))
                fila['mse_test'] = float(np.mean((q - te['T'].values)**2))
                fila['mse_test_mdl'] = float(np.mean((np.array(dm['pred_mdl_test']) - te['T'].values)**2))
    T = pd.DataFrame(filas); C = pd.DataFrame(controles)
    T['compite'] = T.get('delta_dl', pd.Series(np.nan, index=T.index)) < 0
    # MSE de entrenamiento necesario para igualar la DL del MDL: ln SSE = (2(DL_MDL − EP) − (k − n)·ln n)/n − ln 2π − 1
    n_tr = T['n_train']; k = 1 + T['n_parametros']
    ln_sse = (2*(T['dl_mdl_particion'] - T['ep']) - (k - n_tr)*np.log(n_tr))/n_tr - np.log(2*np.pi) - 1
    T['mse_train_necesario'] = np.exp(ln_sse)/n_tr
    T['por_debajo_del_ruido'] = T['mse_train_necesario'] < SIGMA2_RUIDO*(n_tr - k)/n_tr   # E[SSE/n] = σ²(n − k)/n

    # 2. sympify con 10 min (literal del protocolo)
    if not SIN_SYMPIFY and cadenas_sympify:
        from multiprocessing import Pool
        unicas = list(dict.fromkeys(cadenas_sympify))
        print(f'sympify con {LIMITE_SYMPIFY_S:.0f} s: {len(unicas)} cadenas, {nproc} procesos', flush=True)
        est = {}
        with Pool(nproc, maxtasksperchild=1) as pool:
            for cad, e, s in pool.imap_unordered(sympify_10min, unicas):
                est[cad] = (e, s); print(f'  {e:14s} {s:7.1f} s  {cad[:70]}', flush=True)
        T['sympify_10min'] = [', '.join(f"{'str' if ' ' in r else 'canónica'}: {est[r][0]} ({est[r][1]:.0f} s)"
                                        for r in dict.fromkeys(cadenas_sympify) if r.replace(' ', '') == e)
                              for e in T.expr]

    T.to_csv(OUT/'reevaluacion.csv', index=False); C.to_csv(OUT/'control_mdl.csv', index=False)
    (OUT/'entorno.json').write_text(json.dumps(dict(python=platform.python_version(), plataforma=platform.platform(),
                                                    protocolo_sha=PROTOCOLO_SHA, n_arranques=N_ARRANQUES,
                                                    limite_sympify_s=None if SIN_SYMPIFY else LIMITE_SYMPIFY_S)), encoding='utf-8')
    fmt = lambda v: f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    tabla = lambda t, cols: '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|'*len(cols)] +
                                      ['| ' + ' | '.join(fmt(getattr(r, c)) for c in cols) + ' |' for r in t.itertuples()])
    cols = ['fuente', 'carpeta', 'particion', 'cadena', 'rechazada_en_cadena', 'estado', 'dl', 'dl_mdl_particion', 'delta_dl',
            'ep', 'mse_train_necesario', 'mse_train_mdl', 'n_parametros', 'dl_20_arranques', 'dl_200_arranques'] + (['sympify_10min'] if 'sympify_10min' in T else [])
    md = [f"# C-010 — Reevaluación de las expresiones que agotaron el límite de sympify (§7){' — PRUEBA' if PRUEBA else ''}", '',
          f"- Expresiones únicas: {len(T)}; rechazadas en la cadena (forma con espacios): {int(T.rechazada_en_cadena.sum())}; "
          f"solo canonical(): {int((~T.rechazada_en_cadena).sum())}.",
          f"- Estados: {T.estado.value_counts().to_dict()}.",
          f"- Compiten con el MDL de su partición (ΔDL < 0): {int(T.compite.sum())}. ΔDL mínima: {T.delta_dl.min():.2f}; "
          f"entre las rechazadas: {T[T.rechazada_en_cadena].delta_dl.min():.2f}.",
          f"- Expresiones que, para igualar al MDL, necesitarían un MSE de entrenamiento por debajo del ruido "
          f"(σ² = {SIGMA2_RUIDO} kN², C-006): {int(T.por_debajo_del_ruido.sum())} de {len(T)} "
          f"(máximo necesario: {T.mse_train_necesario.max():.4f}).", '',
          '## Control del procedimiento (MDL de cada partición afectada)', '',
          tabla(C, ['fuente', 'carpeta', 'particion', 'cadena_mdl', 'dl_registrada', 'dl_parametros_guardados', 'dl_reajuste_20',
                    'dl_reajuste_200', 'dl_reajuste']), '',
          '## Expresiones (ordenadas por ΔDL)', '', tabla(T.sort_values('delta_dl'), cols)]
    (OUT/'C-010_informe.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))
