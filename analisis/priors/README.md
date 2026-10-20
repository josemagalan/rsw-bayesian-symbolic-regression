# BMS priors

Not redistributed. `python analisis/obtener_bms.py` extracts them from the official BMS repository
(https://bitbucket.org/rguimera/machine-scientist, commit `9defb7aa2a1382b4a8518f3060e8dd87a20acb99`,
folder `Prior/`) and verifies their SHA-256.

- Only change: in the **file name**, spaces and `:` become `_` (`:` is invalid on Windows). Content is identical.
- Selection: priors for 2, 3 and 4 variables (`nv2.*`, `nv3.np3`, `nv4.np8`, `nv4.np50`).
  The analyses use `nv2.np6` (main), `nv2.np4` (sensitivity) and `nv3.np3` (synthetic pilots).
- Please cite Guimerà et al. (2020), *Sci. Adv.* 6:eaav6971, if you use them.

`README_priors_original_es.md` is the original (Spanish) note of the working repository.
