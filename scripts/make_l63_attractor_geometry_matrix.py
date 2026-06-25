"""
Assemble a 2x2 Lorenz-63 attractor matrix from existing PDF panels.

This script does not regenerate trajectories; it composes the final matrix from
the exact panel PDFs you already have.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Create 2x2 L63 attractor matrix from existing PDFs.")
    p.add_argument(
        "--top_left",
        default="l63_output_folder_new/baseline_l63_ns0.1_xl300_bs20_ts100_s42/eval_noisy_trainval_clean_test/rollout_1000/test_on_noise_data_ns0.10/l63_attractor_geometry_001.pdf",
        type=str,
    )
    p.add_argument(
        "--top_right",
        default="l63_output_folder_new/wgan_l63_ns0.1_xl300_bs20_ts100_steps1_clip0.01_s42/eval_noisy_trainval_clean_test/rollout_1000/test_on_noise_data_ns0.10/l63_attractor_geometry_001.pdf",
        type=str,
    )
    p.add_argument(
        "--bottom_left",
        default="l63_output_folder_new/baseline_l63_ns0.15_xl300_bs20_ts100_s42/eval_noisy_trainval_clean_test/rollout_1000/test_on_noise_data_ns0.15/l63_attractor_geometry_001.pdf",
        type=str,
    )
    p.add_argument(
        "--bottom_right",
        default="l63_output_folder_new/wgan_l63_ns0.15_xl300_bs20_ts100_steps1_clip0.01_s42/eval_noisy_trainval_clean_test/rollout_1000/test_on_noise_data_ns0.15/l63_attractor_geometry_001.pdf",
        type=str,
    )
    p.add_argument(
        "--output_pdf",
        default="l63_output_folder_new/l63_attractor_geometry_comparison_matrix.pdf",
        type=str,
    )
    return p.parse_args()


def _check_pdf(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing panel PDF: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF panel, got: {path}")


def _tex_graphic_path(path: Path) -> str:
    # detokenize protects underscores and other special chars in file names.
    return f"\\detokenize{{{path.resolve().as_posix()}}}"


def main():
    args = parse_args()

    tl = Path(args.top_left)
    tr = Path(args.top_right)
    bl = Path(args.bottom_left)
    br = Path(args.bottom_right)
    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    for p in (tl, tr, bl, br):
        _check_pdf(p)

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex not found on PATH. Install MacTeX/TeX Live.")

    workdir = out_pdf.parent / ".l63_matrix_build"
    workdir.mkdir(parents=True, exist_ok=True)
    tex_path = workdir / "matrix.tex"

    tex = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=0.5in]{geometry}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{array}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        "\\begin{center}\n"
        "{\\LARGE \\textbf{Lorenz-63 Attractor Geometry Comparison}}\n"
        "\\vspace{0.6em}\n\n"
        "\\renewcommand{\\arraystretch}{1.1}\n"
        "\\begin{tabular}{>{\\centering\\arraybackslash}m{0.08\\textwidth} >{\\centering\\arraybackslash}m{0.43\\textwidth} >{\\centering\\arraybackslash}m{0.43\\textwidth}}\n"
        " & \\textbf{Baseline (No OT)} & \\textbf{WGAN (Learnable)} \\\\\n"
        "\\textbf{$\\sigma=0.1$} &\n"
        f"\\includegraphics[width=\\linewidth]{{{_tex_graphic_path(tl)}}} &\n"
        f"\\includegraphics[width=\\linewidth]{{{_tex_graphic_path(tr)}}} \\\\\n"
        "\\textbf{$\\sigma=0.15$} &\n"
        f"\\includegraphics[width=\\linewidth]{{{_tex_graphic_path(bl)}}} &\n"
        f"\\includegraphics[width=\\linewidth]{{{_tex_graphic_path(br)}}} \\\\\n"
        "\\end{tabular}\n"
        "\\end{center}\n"
        "\\end{document}\n"
    )

    tex_path.write_text(tex)

    cmd = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={workdir}",
        str(tex_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "pdflatex failed.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    built_pdf = workdir / "matrix.pdf"
    if not built_pdf.exists():
        raise FileNotFoundError(f"Expected output PDF missing: {built_pdf}")
    shutil.copyfile(built_pdf, out_pdf)
    print(f"Saved matrix PDF: {out_pdf}")


if __name__ == "__main__":
    main()
