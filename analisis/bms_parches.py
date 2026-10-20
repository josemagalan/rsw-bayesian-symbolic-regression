"""Parches mínimos al BMS oficial, aplicados en tiempo de ejecución (el código de vendor/ NO se modifica).

D-007 (2026-10-14). Problema: sympy simplifica algunas expresiones a formas con la unidad imaginaria
(p. ej. log(-(WT/WT)) -> I*pi, (-1)**WT -> exp(I*pi*WT)). Su evaluación da valores complejos, la
longitud de descripción (energía) resulta compleja y el MCMC falla (float(EB) en tree_swap) o compara
energías complejas. Además, scipy imprime "TypeError: Cannot cast array data from complex128..."
al intentar ajustar esas expresiones.

Solución: una expresión que no es real sobre los datos no puede describir una magnitud real, así que
se trata como inválida (SSE = inf -> energía = inf -> el movimiento se rechaza), igual que el código
original hace con las expresiones que no se pueden evaluar.

D-008 (2026-10-15). Problema: algunas cadenas no terminaban (>14 h). Volcado con py-spy de los 3 procesos:
los tres estaban dentro de sympify (mcmc.get_sse, Tree(from_string) y bms_parches._tiene_imaginaria), en la
evaluación automática de funciones hiperbólicas/exponenciales (_eval_is_real -> Mod -> gcd de polinomios /
expand_multinomial), que para ciertas expresiones anidadas explota combinatoriamente. Uno de ellos ocupaba 19.5 GB.
Solución: límite de tiempo (TIEMPO_SYMPIFY_S) a cada sympify que hace el BMS. Si se supera: en canonical()
el propio código original recurre a la forma completa (except desnudo); en get_sse la expresión se trata como
inválida (SSE = inf, el movimiento se rechaza). Se cuenta en CONTADOR['timeout_sympify'].
Mecanismo: hilo vigilante que lanza una excepción asíncrona (PyThreadState_SetAsyncExc) en el hilo principal;
funciona porque sympy es Python puro. En expresiones normales (ms) no interviene.

D-008 v3 (2026-10-16, exigida por el protocolo congelado §7): límite por tiempo de CPU del proceso;
conjunto de fallos que nunca se expulsa; registro de las expresiones que agotan el límite (REGISTRO_TIMEOUT)
para reevaluarlas fuera de línea. Cada cadena debe correr en un proceso nuevo (cachés limpias).

Uso: importar este módulo DESPUÉS de importar mcmc:
    import mcmc; import bms_parches; bms_parches.aplicar()
"""
import ctypes
import gc
import threading
import time
from collections import OrderedDict
import numpy as np
import sympy

TIEMPO_SYMPIFY_S = 5.0


class TiempoSympify(BaseException):
    """BaseException para que los 'except Exception' internos de sympy no la oculten."""


def _con_limite(fn, *args, **kwargs):
    tid = threading.get_ident()
    terminado = threading.Event()
    lock = threading.Lock()

    def vigilante():
        # v3 (2026-10-16): el límite cuenta tiempo de CPU del proceso (time.process_time), no de reloj,
        # para que no dependa de la carga de la máquina. Se repite cada TIEMPO_SYMPIFY_S de CPU por si
        # la excepción se captura y el cálculo sigue.
        inicio = time.process_time()
        while not terminado.wait(0.2):
            if time.process_time() - inicio >= TIEMPO_SYMPIFY_S:
                with lock:
                    if not terminado.is_set():
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid),
                                                                   ctypes.py_object(TiempoSympify))
                inicio = time.process_time()
    h = threading.Thread(target=vigilante, daemon=True)
    gc_activo = gc.isenabled()
    gc.disable()        # evita que la excepción asíncrona caiga dentro de un finalizador del recolector
    h.start()
    try:
        return fn(*args, **kwargs)
    finally:
        with lock:
            terminado.set()
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)  # anula una pendiente
        if gc_activo:
            gc.enable()


_CACHE_SYMPIFY = OrderedDict()
_CACHE_MAX = 50000
_FALLO = object()
_FALLOS = set()          # v3: expresiones que agotaron el límite; nunca se expulsan
REGISTRO_TIMEOUT = []    # v3: expresiones que agotaron el límite (para reevaluarlas fuera de línea)


def sympify_con_limite(*args, **kwargs):
    """sympify con límite de tiempo y CACHÉ por cadena de entrada (v2, 2026-10-15).
    La caché hace que el resultado (éxito o tiempo agotado) de una misma expresión sea siempre el mismo
    dentro de una cadena. Sin ella, una expresión aceptada podía agotar el tiempo al recalcularse tras la
    aceptación (get_bic(reset=True, fit=False)), dejando bic = inf con E finita; el movimiento siguiente
    daba dE = -inf y la cadena quedaba con E = -inf (observado en E-006, T1_potencias_D_s0.10_seed2).
    Los objetos de sympy son inmutables, así que reutilizar el resultado no cambia nada."""
    try:
        clave = (args, tuple(sorted(kwargs.items())))
        hash(clave)
    except TypeError:
        clave = None
    if clave is not None and clave in _FALLOS:
        raise TiempoSympify()
    if clave is not None and clave in _CACHE_SYMPIFY:
        r = _CACHE_SYMPIFY[clave]
        _CACHE_SYMPIFY.move_to_end(clave)
        if r is _FALLO:
            raise TiempoSympify()
        return r
    try:
        r = _con_limite(sympy.sympify, *args, **kwargs)
    except TiempoSympify:
        CONTADOR['timeout_sympify'] += 1
        if clave is not None:
            _FALLOS.add(clave)
        if args and isinstance(args[0], str) and len(REGISTRO_TIMEOUT) < 2000:
            REGISTRO_TIMEOUT.append(args[0])
        raise
    if clave is not None:
        _guardar(clave, r)
    return r


def _guardar(clave, valor):
    _CACHE_SYMPIFY[clave] = valor
    if len(_CACHE_SYMPIFY) > _CACHE_MAX:
        _CACHE_SYMPIFY.popitem(last=False)

CONTADOR = {'expresion_compleja': 0, 'sse_compleja': 0, 'excepcion': 0, 'timeout_sympify': 0}
_CACHE_I = {}
_aplicado = False


def _tiene_imaginaria(expr_str):
    v = _CACHE_I.get(expr_str)
    if v is None:
        try:
            v = bool(sympify_con_limite(expr_str).has(sympy.I))
        except TiempoSympify:
            v = True        # no se puede ni interpretar en un tiempo razonable: se rechaza
        except Exception:
            v = False
        _CACHE_I[expr_str] = v
    return v


def aplicar():
    global _aplicado
    if _aplicado:
        return
    import mcmc
    mcmc.sympify = sympify_con_limite         # D-008: todos los sympify de mcmc.py con límite de tiempo
    original = mcmc.Tree.get_sse

    def get_sse_robusto(self, fit=True, verbose=False):
        if _tiene_imaginaria(str(self)):
            CONTADOR['expresion_compleja'] += 1
            self.sse = {ds: np.inf for ds in self.x}
            return self.sse
        try:
            sse = original(self, fit=fit, verbose=verbose)
        except (Exception, TiempoSympify):
            CONTADOR['excepcion'] += 1
            sse = {ds: np.inf for ds in self.x}
        if isinstance(sse, dict):
            for ds, v in list(sse.items()):
                if np.iscomplexobj(v):
                    if np.imag(v) != 0:
                        CONTADOR['sse_compleja'] += 1
                        sse[ds] = np.inf
                    else:
                        sse[ds] = float(np.real(v))
        self.sse = sse
        return sse

    mcmc.Tree.get_sse = get_sse_robusto
    _aplicado = True
