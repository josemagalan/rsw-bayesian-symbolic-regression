# Dataset

The experimental data (242 resistance spot welds, AISI 304 austenitic stainless steel, 0.8 mm) were
produced and tested at the University of Valladolid by Óscar Martín and Pilar de Tiedra. The original
file used by all the analyses is included here, unchanged, so that the results can be reproduced:

| File | `datos_corregido_junio16_2_puntos20-69-88.xlsx` (sheet `Hoja1`) |
|---|---|
| Size | 28 818 bytes |
| SHA-256 | `ae881203f8f6027daf1baa3bccc56aa0d9d528b8d6cd2aaa0a87af47fc02e99b` |

Every script verifies this hash when it loads the data and stops if it does not match.
The file name (including "puntos20-69-88") is historical and is kept for traceability.
Columns used (Excel columns C-I): joint id, welding time in cycles, WT (s), WC (kA, measured RMS),
EF (N), TSLBC (kN), quality class (1 if TSLBC >= 5.93 kN).

## Citation and licence of the data

The dataset has been submitted to UVaDOC, the institutional repository of the University of
Valladolid, and is awaiting the assignment of a DOI. **If you use these data, please cite the
dataset:**

> Martín, Ó., & de Tiedra, P. (2026). *RSW-AISI304: a dataset of 242 resistance spot welds with
> tensile-shear strength measurements* (Version 1) [Data set]. Universidad de Valladolid.
> DOI pending.

The DOI will be added here once it has been assigned. The UVaDOC record contains the same values as
a CSV file with English column names, together with a README describing the material, the welding
conditions and the tests.

The data file is not covered by the MIT licence of the code in this repository; its terms of use are
those of the UVaDOC record.
