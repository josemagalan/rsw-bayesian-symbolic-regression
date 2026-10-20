"""Compone las figuras del borrador en PDF vectorial (articulo/borrador/figuras/), con la numeración del artículo.
No calcula nada: copia las salidas de C-012, E-014 y E-018 y une paneles con pdflatex (sin rasterizar).
Uso (VM o cualquier equipo con pdflatex): python3 articulo/componer_figuras.py"""
import shutil, subprocess, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT/'resultados'; OUT = ROOT/'articulo'/'borrador'/'figuras'
OUT.mkdir(parents=True, exist_ok=True)
C12 = R/'C-012_figuras'
SIMPLES = {'Fig03_data.pdf': C12/'Fig1_datos.pdf',
           'Fig09_acceptance_boundary.pdf': R/'2026-10-19_E-018'/'Fig_frontera_v2.pdf',
           'Fig10_extrapolation.pdf': C12/'Fig6_extrapolacion.pdf',
           'FigS2_chain_traces.pdf': C12/'FigS1_cadenas.pdf',
           'FigS3_residual_diagnostics.pdf': C12/'FigS2_residuos_A.pdf',
           'Fig08_response_surfaces.pdf': R/'2026-10-19_E-014'/'Fig_superficies.pdf',
           'Fig07_partial_effects.pdf': R/'2026-10-19_E-020'/'Fig7_efectos_parciales.pdf',
           'FigS1_synthetic_recovery.pdf': R/'2026-10-19_E-020'/'FigS1_sintetico.pdf',
           'FigS4_threshold_forms.pdf': R/'2026-10-19_E-020'/'FigS4_umbral.pdf'}
def panel(letra, ruta, ancho):
    """Minipage alineada arriba: letra del panel encima, a la izquierda, y la figura debajo."""
    return (r'\begin{minipage}[t]{%s}\vspace{0pt}{\sffamily\bfseries\small %s}\\[-2pt]'
            r'\includegraphics[width=\linewidth]{%s}\end{minipage}' % (ancho, letra, Path(ruta).as_posix()))
COMPUESTAS = {  # nombre: cuerpo LaTeX
 'Fig05_accuracy.pdf': panel('a', C12/'Fig2_mse_cv.pdf', '0.53\\linewidth') + r'\hfill'
                       + panel('b', C12/'Fig3_diferencias_rope.pdf', '0.45\\linewidth'),
 'Fig06_expressions_motifs.pdf': panel('a', C12/'Fig4_modelos_A.pdf', '\\linewidth') + r'\\[6pt]'
                       + panel('b', C12/'Fig5_motivos_prior.pdf', '0.55\\linewidth'),
}
F4 = ROOT/'articulo'/'figuras'/'Fig4_flujo_analisis.pdf'   # generada aparte (analisis/figuras/Fig4_flujo_analisis.py)
if F4.exists():
    SIMPLES['Fig04_analysis_workflow.pdf'] = F4
for n, src in SIMPLES.items():
    shutil.copyfile(src, OUT/n)
plantilla = r"""\documentclass[varwidth=539pt,border=2pt]{standalone}
\usepackage{graphicx}\usepackage[scaled]{helvet}
\begin{document}\setlength{\linewidth}{539pt}\noindent\begin{minipage}[t]{539pt}\noindent %s\end{minipage}\end{document}"""
with tempfile.TemporaryDirectory() as tmp:
    for n, cuerpo in COMPUESTAS.items():
        tex = Path(tmp)/'f.tex'; tex.write_text(plantilla % cuerpo, encoding='utf-8')
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', tmp, str(tex)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:]
        shutil.copyfile(Path(tmp)/'f.pdf', OUT/n)
print('\n'.join(sorted(p.name for p in OUT.glob('*.pdf'))))
