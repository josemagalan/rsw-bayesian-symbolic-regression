# Bayesian symbolic regression of resistance spot weld strength — analysis code

Code to reproduce the analyses of the paper *Bayesian symbolic regression of RSW joint strength*
(manuscript in preparation; citation and DOI will be added on publication).

The study applies the Bayesian Machine Scientist (BMS; Guimerà et al., 2020) to the tensile-shear load-bearing
capacity (TSLBC, kN) of 242 resistance spot welds in AISI 304 sheet (0.8 mm) produced under a full factorial
design of welding time (WT, s), welding current (WC, kA) and electrode force (EF, N), and compares it with PySR
and with non-symbolic baselines (mean, quadratic elastic net, GAM, random forest, gradient boosting, EBM, TabPFN v2)
under one grouped cross-validation protocol.

> **Language note.** Code comments, console messages and the analysis protocol are in Spanish (the working
> language of the project). This README and the paper are in English.

## What is (and is not) in this repository

| Included | Not included |
|---|---|
| All analysis scripts, confirmatory (`C-…`) and exploratory (`E-…`) | Experimental data (see [`Dataset/README.md`](Dataset/README.md)) |
| Frozen cross-validation partitions (`analisis/datos_derivados/`: joint ids and folds only, no responses) | Result files (archived separately; see the paper) |
| Frozen analysis protocol (`00_gestion/`, in Spanish) | BMS source code and priors (fetched by `analisis/obtener_bms.py`) |
| Locked environment (pip lock, Julia manifest) | Fig. 4 (hand-designed workflow schematic) and Figs. 1–2 (photographs) |

## Confirmatory vs exploratory analyses

The analysis protocol was frozen on 2026-10-16 before any confirmatory analysis
(`00_gestion/PROTOCOLO_ANALISIS_congelado_2026-10-16.md`, SHA-256 `deebe865…2d23335`). Its Annex A
(reduced BMS sensitivity analyses, decision D-015, 2026-10-18) is in `00_gestion/PROTOCOLO_ANALISIS.md`
(SHA-256 `a9789d2a…f207854`). Scripts record these hashes in their outputs.

- `analisis/confirmatorio/C-*.py`: analyses specified in the protocol.
- `analisis/exploratorio/E-*.py`: everything else — pilots on synthetic data, calibrations done before
  freezing the protocol (which looked at the real data; see protocol §0) and post hoc analyses.
  They are all published, including those that do not appear in the paper.

The protocol refers to internal project documents (decision log, lab notebook) that are not part of this repository.

## Setup

Reference environment: Windows, Python 3.14.5, Julia 1.11.9 (installed by PySR through `juliacall`).
Some exploratory analyses were run on Linux (noted in the paper).

```bash
git clone https://github.com/josemagalan/rsw-bayesian-symbolic-regression.git
cd rsw-bayesian-symbolic-regression
python -m venv .venv
# Windows: .venv\Scripts\python.exe   Linux/macOS: .venv/bin/python
python -m pip install -r analisis/requirements-lock-windows.txt   # exact versions (or requirements.txt)
python analisis/obtener_bms.py      # clones the BMS at commit 9defb7a and extracts the priors (needs git)
python analisis/verificar_entorno.py   # quick check of GAM, PySR, TabPFN and baycomp on synthetic data
```

- The BMS is used unmodified; two run-time patches are applied by `analisis/bms_parches.py` (complex
  expressions and a 5 s time limit on `sympify`; decisions D-007 and D-008, described in the paper's SM).
- TabPFN downloads the v2 model weights on first use; depending on the TabPFN version this may require
  accepting the model licence on Hugging Face.
- Figure assembly (`articulo/componer_figuras*.py`) needs `pdflatex` and `pdftoppm`.
- Then copy the data file into `Dataset/` (see [`Dataset/README.md`](Dataset/README.md)).

## Running

Run every script **from the repository root**: `python analisis/<folder>/<script>.py [args]`.
Each script checks the data hash and writes to `resultados/<ID>/` or `resultados/<date>_<ID>/`, together with its
configuration, seeds and environment. Usage and inputs are given in each script's docstring.
Most confirmatory scripts accept `--prueba-sintetica` (C-010: `--prueba`), which replaces the response with synthetic
data and writes to `resultados/_pruebas/`; use it to test the pipeline without the real data.

Confirmatory pipeline (approximate wall times in the reference environment):

| Step | Script | Notes |
|---|---|---|
| 0 | `C-000_particiones.py`, `C-000b_particiones_sensibilidad.py` | Partitions (already in `datos_derivados/`; the scripts refuse to overwrite them) |
| 1 | `C-001b_H4a_gemelas.py` | H4a, paired test on 121 twin joints (seconds) |
| 2 | `C-001_analisis_A.py [procs]` → `C-001_analisis.py` | Analysis A: BMS on all data, 16 chains (≈ 4–6 h) |
| 3 | `C-004_comparadores.py` (also `--sin-ef`) | Baselines on the 25 grouped partitions (≈ 30–60 min) |
| 4 | `C-002_pysr_cv.py [procs]` | PySR on the 25 partitions (≈ 1–3 h) |
| 5 | `C-003_bms_cv.py [procs]` | BMS on the 25 partitions, 100 chains (≈ 17–20 h) |
| 6 | `C-005_analisis_cv.py` | Analysis B: cross-validation statistics (H1–H4) |
| 7 | `C-006_ruido.py` | Model-free noise estimate |
| 8 | `C-007_bms_sensibilidad.py`, `C-008_sensibilidad_comparadores.py` → `C-009_analisis_sensibilidad.py` | Sensitivity analyses (protocol §9, Annex A) |
| 9 | `C-010_reevaluar_timeouts.py`, `C-011_informe_s10.py` | Re-evaluation of `sympify` time-outs; protocol §10 report |
| 10 | `C-012_figuras.py` | Figures; also reads outputs of E-012 and E-013 |

Exploratory scripts read the confirmatory outputs they need (see their docstrings). E-005/E-006 (BMS recovery of
known formulas) and E-004, E-008 use synthetic data only. `E-018_revision_frontera/` is an adversarial review of
E-017: run `ataque1`–`ataque4` and then `figura.py`.

## Paper figures and tables

`articulo/componer_figuras.py` copies or joins the script outputs below into the paper figures;
`articulo/componer_figuras_v2.py` applies the final numbering.

| Paper | Produced by |
|---|---|
| Fig. 3 (data) | `C-012` → `Fig1_datos.pdf` |
| Fig. 5 (predictive accuracy) | `C-012` → `Fig2_mse_cv.pdf` + `Fig3_diferencias_rope.pdf` |
| Fig. 6 (expressions and motifs) | `C-012` → `Fig4_modelos_A.pdf` + `Fig5_motivos_prior.pdf` |
| Fig. 7 (partial effects) | `E-020` → `Fig7_efectos_parciales.pdf` |
| Fig. 8 (acceptance boundary) | `E-018_revision_frontera/figura.py` → `Fig_frontera_v2.pdf` |
| Fig. 9 (extrapolation) | `C-012` → `Fig6_extrapolacion.pdf` |
| Fig. S1 (synthetic recovery) | `E-020` → `FigS1_sintetico.pdf` |
| Fig. S2 (chain traces) | `C-012` → `FigS1_cadenas.pdf` |
| Fig. S3 (residual diagnostics) | `C-012` → `FigS2_residuos_A.pdf` |
| Fig. S4 (threshold forms) | `E-020` → `FigS4_umbral.pdf` |
| Fig. S5 (response surfaces) | `E-014` → `Fig_superficies.pdf` |
| Table 4 | `C-005` (means) and `E-021` (standard deviations between partitions) |
| Tables S5, S6 | `E-021` (from the outputs of `C-005` and `C-009`) |
| Table S7 | `E-020` |

## Reproducibility notes

- All stochastic steps use explicit seeds. The BMS uses Python's `random` module: scripts seed both `random` and
  `numpy.random`. Each BMS chain runs in its own process with its own seed; a chain that fails or exceeds the
  safety time limit (`bms_comun.planificar`, 12 h by default) is relaunched once with seed + 10000 (output saved with suffix `_r`).
- Exact numerical agreement is expected only in the reference environment; PySR runs with
  `deterministic=True` and serial parallelism, but results may still vary across platforms and Julia versions.
- Variable names in the BMS are `WT`, `WC` and `EF` (sympy treats `I`, `E`, `S`, `N`, `O`, `Q` as constants).
- Errors are always reported in kN².

## Changes with respect to the working repository

The scripts are copied unchanged from the project's working repository, except:
- `exploratorio/E-018_revision_frontera/ataque3_forma.py` and `ataque4_extra.py`: the input
  `ataque1_fronteras.csv` is read from `resultados/2026-10-19_E-018/` (where `ataque1_estimador.py` writes it)
  instead of a temporary folder of the original machine. No change in the computation.
- `analisis/requirements.txt` lists all direct dependencies; `analisis/obtener_bms.py` is new.

## Licence and citation

Code: MIT licence (see [`LICENSE`](LICENSE)). The BMS is © its authors and is not redistributed here.
If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)) and Guimerà, R. et al. (2020).
A Bayesian machine scientist to aid in the solution of challenging scientific problems. *Science Advances*
6(5), eaav6971. https://doi.org/10.1126/sciadv.aav6971

Development of the code was assisted by Claude (Anthropic) under the authors' supervision.
