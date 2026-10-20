# PROTOCOLO DE ANÁLISIS (pre-registro interno)

> **Estado: CONGELADO — versión 2, 2026-10-16** (D-014; SHA-256 de este fichero en `DECISIONES.md`). Incorpora la revisión adversaria (`00_gestion/revision_adversaria_protocolo_v1.md`) y las calibraciones E-007 (potencia) y E-008 (criterio de H3b).
> Decisiones de H1 tomadas por JMG (D-013). Entorno ampliado instalado y verificado (`analisis/verificar_entorno.py`: todo correcto) y lock fijado.
> Al congelar se anotan la fecha y el SHA-256 de este fichero en `DECISIONES.md`. Desde ese momento, cualquier cambio es una **desviación** (D-xxx con motivo). Todo lo hecho antes, o fuera de este protocolo, es exploratorio.

## 0. Advertencias previas (contaminación e identificabilidad)

Los datos reales ya se han mirado en estos análisis exploratorios, y eso condiciona lo que se puede confirmar:
- **E-001:** formas cerradas con los 242 puntos. De ahí salen los exponentes ≈ 1.38 / 0.345 (hipótesis de "dosis" WC²·√WT, A-008), un exponente de EF ≈ −0.03 y los MSE de un cuadrático y un GBM.
- **E-002:** MSE de EBM, GBM y RF (CV aleatoria).
- **E-007:** MSE por partición del cuadrático, RF, GBM y el modelo de E-001 con las particiones agrupadas, usados solo para calcular la potencia.
- **Consecuencias:**
  - una γ ≈ 0.25 en H3b o un efecto nulo de EF en H4 serán *consistentes* con lo explorado, **no confirmaciones independientes**;
  - el BMS no se ha ejecutado nunca con los datos reales.
- **Identificabilidad** (E-005/E-006): con n = 242 y σ ≥ 0.10 kN solo se identifican estructuras de ≈ 2 parámetros. Con verdades más complejas, el BMS da sustitutos compactos y precisos, que conservan la combinación de variables pero no "la ley". La interpretación física se limita a lo estable (H3b) y se presenta como hipótesis.

## 1. Datos y variables

- Datos: 242 puntos, SHA-256 `ae881203…02e99b`, sin exclusiones en el análisis principal.
- **Celda:** (ciclos, nivel nominal de corriente). El nivel nominal es el rango de la WC medida dentro de cada (ciclos, EF) y va de 1 a 11. Hay 121 celdas, cada una con dos uniones "gemelas" (EF = 1000 y 1500 N); la WC medida de las gemelas coincide en mediana (diferencia máxima 0.27 kA).
- Respuesta: TSLBC (kN).
- **Predictores:**
  - BMS y PySR usan WT (s) y WC medida (kA), en unidades físicas.
  - Los demás comparadores usan WT, WC y EF (kN).
  - WC medida en todo el análisis principal (D-002, cerrada); WC nominal como sensibilidad.
- El error se informa siempre en **kN²**.

## 2. Diseño: dos análisis (D-004)

**A. Explicativo (todos los datos).** Busca la forma funcional, el posterior sobre estructuras y H4.
- BMS configuración D (WT, WC; un conjunto; prior `nv2.np6`): 8 cadenas, semillas 1001–1008.
- BMS configuración B (WT, WC; EF como 2 conjuntos; `nv2.np6`): 8 cadenas, semillas 2001–2008. Solo para H4.
- Sus resultados **no** estiman el error de predicción.

**B. Predictivo (CV).**
- **5-fold repetida 5 veces = 25 particiones, agrupada por celda** (las dos gemelas siempre van juntas al entrenamiento o a la prueba). Se usa `StratifiedGroupKFold`, estratificada por quintiles de la TSLBC media de la celda, con semillas 20261016 + repetición.
- **Motivo de agrupar** (E-007): con particiones aleatorias, la gemela de un punto de prueba suele estar en el entrenamiento y actúa casi como una réplica, lo que da errores optimistas. Con particiones agrupadas, el MSE de RF pasó de ≈ 0.081 a 0.097 kN².
- Particiones guardadas en `analisis/datos_derivados/folds_5x5_agrupado.csv` y generadas una sola vez.
- **CV interna para los hiperparámetros:** 5 particiones agrupadas por celda dentro del entrenamiento, semilla fija, métrica MSE.
- El BMS repite en cada partición la búsqueda de estructura, con 4 cadenas y semillas 100000 + 100·partición + cadena.

## 3. Modelos

**Principal: BMS** (Guimerà et al. 2020; commit `9defb7a`; parches D-007 y D-008 v3, ver §7), configuración D. Dos predicciones:
- **MDL** (decide H1, D-010): la expresión de mínima DL entre las cadenas, con sus parámetros ajustados.
- **Promedio de muestras MCMC (aproximación al promedio del posterior)**, secundaria: media de las predicciones de las expresiones de t1 en los pasos 10 000–20 000 (una de cada 100), juntando las cadenas. No es un posterior exacto, porque las cadenas no se mezclan entre sí; se informan diagnósticos (§7).

**Comparadores** (D-009; conjunto, versiones y rejillas fijados ahora):
1. Media del entrenamiento.
2. Cuadrático con elastic net (Martín et al. 2016): grado 2 sobre (WT, WC, EF) estandarizados; `l1_ratio` ∈ {0.1, 0.5, 0.9, 1}, α por camino de 50 valores.
3. GAM (`pygam`): `te(WT, WC) + l(EF)`; λ por búsqueda en rejilla (`gridsearch`) dentro del entrenamiento.
4. Random forest: 500 árboles; `max_features` ∈ {1, 2, 3} × `min_samples_leaf` ∈ {1, 3, 5}.
5. GBM: `n_estimators` ∈ {100, 300, 600} × `max_depth` ∈ {2, 3, 4} × `learning_rate` ∈ {0.03, 0.1}.
6. EBM (`interpret`, versión del lock): parámetros por defecto.
7. **PySR** (versión fijada):
   - entradas WT y WC;
   - operadores binarios {+, −, ×, ÷, ^}; unarios {exp, log, sqrt, tanh};
   - `constraints = {"^": (-1, 1)}` y anidamiento prohibido de exp/log/tanh dentro de sí mismos;
   - `maxsize` = 25, `niterations` = 200, `populations` = 20;
   - `model_selection = "best"`, semilla fija, `deterministic = True` (ejecución en serie).
8. **TabPFN v2** (Hollmann et al. 2025, *Nature*, doi 10.1038/s41586-024-08328-6): `TabPFNRegressor` con `ModelVersion.V2` y la configuración por defecto, entradas WT, WC y EF, paquete `tabpfn` en la versión del lock. Se elige v2, y no la versión 3.5 que el paquete usa por defecto, porque es la versión revisada por pares y sus pesos tienen licencia abierta (Prior Labs License = Apache 2.0 + atribución, sin registro). Las versiones ≥ 2.5 tienen licencia no comercial y piden registrarse.
9. Modelo de saturación de E-001, reajustado en cada partición. **Solo exploratorio.**

**Reglas comunes:**
- **Predicciones no finitas:** se sustituyen por la media del entrenamiento y se cuentan.
- Sensibilidad (§9): recortar *todas* las predicciones a [0, 1.2·máx(y_entrenamiento)].
- Si un comparador no se puede ejecutar, es una desviación: se registra y nunca se sustituye en silencio.
- Se informa del coste de cómputo (CPU·h) de cada método.

## 4. Métricas

- **Principal:** MSE fuera de muestra (kN²), media sobre las 25 particiones.
- **Secundarias:** RMSE, MAE y R²; dispersión entre particiones.
- **Clasificación (descriptiva):** umbral de 5.93 kN sobre la predicción; error, exactitud equilibrada y F1.
- **Complejidad:** parámetros distintos y nodos (BMS, PySR).

## 5. Comparación estadística

- **Test t corregido** (Nadeau y Bengio 2003; extensión a k-fold repetido: Bouckaert y Frank 2004, *por verificar*):
  - d_j = MSE_BMS,j − MSE_c,j, para j = 1…25;
  - t = d̄ / √[(1/25 + n_te/n_tr)·s_d²], con s_d² de denominador 24 (ddof = 1) y 24 gl;
  - n_te/n_tr = 0.25 (48.4/193.6).
- **Test t bayesiano correlacionado** (Benavoli et al. 2017; `baycomp`; ρ = n_te/n = 0.2; prior por defecto del paquete; **ROPE = ±δ = ±0.03 kN²**): P(BMS mejor), P(equivalentes) y P(BMS peor), frente a cada comparador.
- **Contrastes secundarios de superioridad** (bilaterales, frente a cada comparador): Holm.
- **Multiplicidad:** H1 es el único contraste con tasa de error controlada (α = 0.05). H4a es un contraste propio, también con α = 0.05. Lo demás son reglas de decisión o descripciones. No se hace ninguna afirmación conjunta.

## 6. Hipótesis

- **H1 (no inferioridad predictiva del BMS-MDL frente a EBM)** (D-013): referencia **EBM**, fijada de antemano (es el comparador interpretable de Santos et al. 2022); margen **δ = 0.03 kN²**, que equivale a ΔRMSE ≈ 0.05 kN con RMSE ≈ 0.32 kN, ≈ 0.8 % del umbral de aceptación de 5.93 kN: sin relevancia práctica. El resto de comparadores, TabPFN y PySR incluidos, se informan con el test bayesiano y con los contrastes secundarios (Holm). El promedio de muestras MCMC se contrasta igual frente a EBM, como secundario.
  - Con d_j = MSE_BMS-MDL,j − MSE_EBM,j, se cumple si el límite superior unilateral al 95 % de E[d] es < δ = 0.03: d̄ + t₀.₉₅,₂₄·√[(1/25 + 0.25)·s_d²] < 0.03.
  - Si no se cumple, el resultado es **no concluyente**. Solo se dirá "inferior" si el límite inferior unilateral supera δ.
  - **Potencia** (E-007, con s_d de pares de comparadores entre 0.020 y 0.039 kN² y verdad ΔMSE = 0): con δ = 0.0127 es de 13–30 %; con 0.03, de 39–86 %; con 0.05, de 75–100 %. Más repeticiones apenas ayudan, porque el término n_te/n_tr = 0.25 domina la corrección.
- **H2 (parsimonia) → descriptiva.** En los pilotos, 46 de 47 MDL tenían ≤ 4 parámetros: el umbral sería trivial. Se informan los parámetros, los nodos y la frecuencia de k ≤ 4 en las muestras MCMC.
- **H3a (estabilidad estructural) → descriptiva.** Frecuencia, en las 25 particiones, de modelos MDL *funcionalmente equivalentes* al del análisis A: reajustado en el entrenamiento, el RMSE entre sus predicciones en las 121 celdas es < 0.05 kN. Se informa sin umbral de decisión.
- **H3b (índice único, confirmatoria):**
  - Para el MDL de cada partición, f(WT, WC), se calculan derivadas numéricas en log en las 121 celdas nominales (niveles nominales de WC 1.5…6.5).
  - Se calculan γ̂ = Σ(∂_T·∂_C)/Σ(∂_C²), R²_ui = 1 − Σ(∂_T − γ̂·∂_C)²/Σ∂_T² y p_T = Σ∂_T²/(Σ∂_T² + Σ∂_C²).
  - **Se cumple si en ≥ 70 % de las particiones R²_ui ≥ 0.90 y p_T ≥ 0.01** (WT no puede ser irrelevante).
  - **Calibración** (E-008, sin datos reales):
    - controles positivos (T1, T3, tanh(WC²√WT)): R²_ui = 1;
    - todos los MDL sustitutos de los pilotos: R²_ui ≥ 0.944;
    - controles negativos: producto no potencial 0.60, aditivo 0.81 y cuadrático de la ec. (6) de 2016 0.16;
    - en los pilotos, γ̂ recupera el valor verdadero con un error ≤ 0.05.
  - Se informan la mediana y el rango intercuartílico de γ̂. Su interpretación física (γ ≈ 0.25 ↔ WC²·√WT) es **consistencia**, no confirmación (§0).
- **H4 (efecto de EF):**
  - **H4a (sin modelo; contraste principal de H4):** con las 121 diferencias gemelas d = y₁₅₀₀ − y₁₀₀₀, test t pareado y Wilcoxon (α = 0.05, bilateral). Se informa la dependencia de d frente a (WT, WC) de forma descriptiva.
  - **H4b (BMS):** Δ = DL_D − DL_B con la mínima DL de 8 cadenas por configuración. Tres desenlaces:
    - Δ > 20 → el modelado separado por EF compensa;
    - |Δ| ≤ 20 → no concluyente;
    - Δ < −20 → no compensa.
    - Calibración de E-006 con una cadena: sin efecto, Δ ∈ [−14, +9]; con un efecto de 0.5 kN, Δ ≥ 34.
  - **Consecuencia si H4a es significativo o H4b > 20:** el modelo D está mal especificado para EF. Se presenta el MDL de B como modelo explicativo, junto con el de D, y se añade el BMS-B a la CV como **análisis secundario**. H1 sigue decidiéndose con D, como estaba registrado.

## 7. Configuración, diagnósticos y fallos del BMS

- 20 000 pasos por cadena; 20 temperaturas (1.04^k).
- **Tope de seguridad de 12 h de reloj por cadena.** Una cadena cortada se relanza **una vez** con semilla + 10 000. Si vuelve a cortarse, se usa su mejor modelo hasta el corte y se declara.
- Una partición necesita **≥ 2 cadenas completas**. Si no las tiene, cuenta como fallo del BMS (predicción = media del entrenamiento) y **nunca se descarta**.
- Una energía −∞ o NaN aborta la cadena (`*_ERROR.txt`), que se trata como cortada.
- **Parches D-008 v3** (a implementar y probar antes de la ejecución confirmatoria):
  - el límite de `sympify` cuenta tiempo de CPU del proceso (`time.process_time`), no tiempo de reloj;
  - las expresiones que agotan el límite van a un conjunto que nunca se vacía y se registran;
  - cada cadena corre en un proceso nuevo (cachés limpias);
  - fuera de línea, las expresiones que agotaron el límite se reevalúan con un límite de 10 min, para comprobar que su DL no competía con el MDL.
- **Diagnósticos:**
  - por cadena: DL mínima, paso del MDL y estructura;
  - concordancia entre cadenas;
  - R̂ y ESS de la traza de DL (se espera mala mezcla entre cadenas y se informa);
  - aceptación de intercambios entre temperaturas;
  - intervenciones de los parches.
- **Posterior sobre estructuras** (análisis A): las 10 estructuras canónicas más frecuentes en las muestras MCMC, con su DL.

## 8. Estimación del ruido (pre-especificada)

- **Estimadores sin modelo:**
  - por diferencias en la rejilla (Rice/Gasser), a lo largo de WT y de WC, para cada nivel de EF;
  - var(d)/2 de las gemelas, que es una cota superior del ruido si EF influye.
- IC al 95 % por bootstrap (por celdas).
- Se informa la posición de σ̂ respecto a los pilotos (σ = 0.10 y 0.28) y, por tanto, qué complejidad estructural es defendible (§0).

## 9. Análisis de sensibilidad (secundarios)

- Particiones aleatorias, sin agrupar (comparabilidad con los trabajos previos).
- Sin el punto de TSLBC = 0.
- WC nominal en lugar de la medida.
- Comparadores sin EF (mismas entradas que el BMS).
- Predicciones recortadas (§3).
- BMS con prior `nv2.np4`.
- **Extrapolación:** se entrena sin los niveles extremos de WT (2 y 12 ciclos) y se predicen esos niveles. Lo mismo con WC (niveles 1 y 11). Se aplica a todos los métodos.

## 10. Informe adicional obligatorio

- **Incertidumbre de los parámetros del MDL (análisis A):** bootstrap de residuos con la estructura fija, 1000 réplicas, IC al 95 %.
- Diagnóstico de residuos: heterocedasticidad y puntos influyentes.
- **Comprobaciones físicas del MDL:**
  - f ≥ 0 en el dominio;
  - f → 0 cuando WT → 0 o WC → 0;
  - saturación;
  - comportamiento fuera del dominio, como advertencia.
- Todas las cadenas, en material suplementario.

## 11. Orden de ejecución (todo fijado aquí; el orden no afecta a la validez)

1. Implementar y probar D-008 v3 con datos sintéticos.
2. Generar las particiones.
3. Análisis A (BMS D y B, 8 + 8 cadenas).
4. PySR en las 25 particiones.
5. BMS en las 25 particiones (100 cadenas).
6. Comparadores 1–6, 8 y 9.
7. Ruido, tests, hipótesis, sensibilidad e informe.

## 12. Reproducibilidad

- **Entorno de referencia:** Windows de JMG, Python 3.14.5.
  - Lock en `analisis/requirements-lock-windows.txt` (72 paquetes; entre otros numpy 2.5.4, scipy 1.16.3, scikit-learn 1.9.1, interpret-core 0.7.8, pygam 0.12.0, pysr 2.7.0, tabpfn 9.2.0 con modelo v2, torch 2.14.1, baycomp 1.0.3).
  - Julia 1.11.9, con su entorno en `analisis/julia_Manifest.toml` y `julia_Project.toml`.
  - BMS `9defb7a`.
  - pygam bajó scipy de 1.18.1 a 1.16.3. E-005 y E-006 (exploratorios) se ejecutaron con 1.18.1.
- Los resultados confirmatorios se generan solo en ese entorno.
- Cada ejecución se guarda en `resultados/C-xxx_<nombre>/` con configuración, semillas, versiones, hash de los datos, salidas y trazas.
- Paquete final en Zenodo, sujeto a Q-005.

---

## Anexo A — Desviaciones registradas tras la congelación

### D-015 (2026-10-18): alcance reducido de la sensibilidad del BMS (§9)
Motivo: coste. El BMS cuesta ≈ 3 CPU·h por cadena (304 CPU·h en C-003). Las siete sensibilidades completas con BMS supondrían ≈ 900 CPU·h. Se aprueba después de ver C-005; afecta solo a análisis secundarios, no a H1–H4.
- **Sin cambios (completo, 25 particiones):** comparadores 1–6, 8 y 9 y PySR en todas las sensibilidades; recorte de predicciones (posproceso); comparadores sin EF.
- **BMS, particiones aleatorias, WC nominal y prior `nv2.np4`:** solo la repetición 0 (5 particiones) × 4 cadenas, con la misma configuración y semillas + 50000. Se compara por pares con las mismas 5 particiones del análisis principal. Total: 60 cadenas.
- **BMS, sin el punto de TSLBC = 0:** sin reajuste. Se reevalúan las predicciones ya obtenidas excluyendo ese punto de los conjuntos de prueba. Los comparadores y PySR sí se reajustan.
- **Extrapolación (WT 2 y 12 ciclos; WC niveles 1 y 11):** completa para todos los métodos (2 divisiones; BMS 4 cadenas en cada una = 8 cadenas).
