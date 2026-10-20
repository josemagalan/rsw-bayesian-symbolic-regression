# Priors del BMS (copiados del repositorio oficial)

- Origen: https://bitbucket.org/rguimera/machine-scientist, rama `no_degeneracy`, commit `9defb7aa2a1382b4a8518f3060e8dd87a20acb99` (2026-07-22), carpeta `Prior/`.
- Se copian aquí porque los nombres originales contienen `:` (no válidos en Windows) y la carpeta `Prior/*.dat` se excluye del checkout de `analisis/vendor/` (sparse-checkout).
- Único cambio: en el **nombre** del fichero, espacios y `:` → `_`. Contenido idéntico.
- Selección: priors con 2, 3 y 4 variables (`nv2.*`, `nv3.np3`, `nv4.np8`, `nv4.np50`).
- Para 3 variables solo existe `nv3.np3` (máx. 3 parámetros). Ver D-006.
- Cita obligatoria si se usan: Guimerà et al. (2020), Sci. Adv. 6:eaav6971.
