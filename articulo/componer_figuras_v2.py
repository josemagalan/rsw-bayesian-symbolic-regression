"""Figuras de la versión corta (v2) del borrador con la numeración nueva.
No calcula nada: copia los PDF vectoriales de articulo/borrador/figuras/ (generados por componer_figuras.py)
a articulo/borrador/figuras_v2/ con los nombres nuevos y crea un PNG a 300 ppp de cada uno (para el Word).
Las Figs. 5 y 6 son los PDF compuestos (paneles a y b en una sola imagen).
Uso: python3 articulo/componer_figuras_v2.py  (requiere pdftoppm)"""
import shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'articulo'/'borrador'/'figuras'; OUT = ROOT/'articulo'/'borrador'/'figuras_v2'
OUT.mkdir(parents=True, exist_ok=True)
MAPA = {  # v2: v1
 'Fig03_data': 'Fig03_data', 'Fig04_analysis_workflow': 'Fig04_analysis_workflow',
 'Fig05_accuracy': 'Fig05_accuracy', 'Fig06_expressions_motifs': 'Fig06_expressions_motifs',
 'Fig07_partial_effects': 'Fig07_partial_effects', 'Fig08_acceptance_boundary': 'Fig09_acceptance_boundary',
 'Fig09_extrapolation': 'Fig10_extrapolation', 'FigS1_synthetic_recovery': 'FigS1_synthetic_recovery',
 'FigS2_chain_traces': 'FigS2_chain_traces', 'FigS3_residual_diagnostics': 'FigS3_residual_diagnostics',
 'FigS4_threshold_forms': 'FigS4_threshold_forms', 'FigS5_response_surfaces': 'Fig08_response_surfaces'}
for nuevo, viejo in MAPA.items():
    shutil.copyfile(SRC/f'{viejo}.pdf', OUT/f'{nuevo}.pdf')
    subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile', str(OUT/f'{nuevo}.pdf'), str(OUT/nuevo)], check=True)
    print(f'{viejo}.pdf -> {nuevo}.pdf/.png')
