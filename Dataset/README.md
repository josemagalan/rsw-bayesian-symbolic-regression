# Dataset

The experimental data (242 resistance spot welds, AISI 304, 0.8 mm) were produced at the
University of Valladolid (O. Martín, P. de Tiedra) and are **not included** in this repository.
Data availability: see the paper's *Data availability* statement.

To run the analyses, place the original file here, unchanged:

| File | `datos_corregido_junio16_2_puntos20-69-88.xlsx` (sheet `Hoja1`) |
|---|---|
| Size | 28 818 bytes |
| SHA-256 | `ae881203f8f6027daf1baa3bccc56aa0d9d528b8d6cd2aaa0a87af47fc02e99b` |

Every script verifies this hash when it loads the data and stops if it does not match.
The file name (including "puntos20-69-88") is historical and is kept for traceability.
Columns used (Excel columns C-I): joint id, welding time in cycles, WT (s), WC (kA, measured RMS),
EF (N), TSLBC (kN), quality class (1 if TSLBC >= 5.93 kN).
