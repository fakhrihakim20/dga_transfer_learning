# Tropical DGA Transfer Learning

Code, data tables, figures, paper, and a public report website for:

> **Tropical Domain Shift in Dissolved Gas Analysis: Transfer Learning from the IEC TC 10 Benchmark to PLN Indonesia for Transformer Fault Classification.**
> Submitted to ICT-PEP 2026.

## 🌐 Report website

A non-technical, interactive report lives in [`website/`](website/) and is auto-published to GitHub Pages:

**→ <https://fakhrihakim20.github.io/dga_transfer_learning/>**

All headline charts are interactive (hover, legend toggle, 7-class / 4-class tab).

## What this repo contains

| Path | What it is |
|---|---|
| `ICT_PEP_2026_DGA_Transfer_Learning.pdf` | The compiled conference paper |
| `ICT_PEP_2026_DGA_Transfer_Learning.tex` | LaTeX source |
| `T2_DGA_Transfer_Learning_Simulation.ipynb` | Reproducible Jupyter notebook |
| `T2_DGA_Transfer_Learning_Simulation_executed.ipynb` | Notebook with all outputs |
| `build_notebook.py` | Script that builds the notebook from sections |
| `verify_fixes.py` | Sanity checks on labelling, boundaries, ANOVA |
| `results/figures/` | Eight publication-quality PNG figures |
| `results/tables/` | Four CSV result tables |
| `results/models/` | Trained model artefacts |
| `gap_analysis_T2_tropical_DGA_transfer_learning.md` | Literature positioning notes |
| `website/` | The public-facing report website (static HTML/CSS/JS) |

## Headline results (at a glance)

- Only **CO** and **CO₂** distributions differ significantly between the IEC TC 10 source and the PLN Indonesia tropical target (KS test, p < 10⁻⁸).
- Pretrain-then-fine-tune lifts macro-F1 from **0.670** (no transfer) to **0.760** (full fine-tune) on the 7-class scheme.
- At **n = 75** target samples, the transfer-learned MLP beats a Random Forest trained only on target data by **+9.0 percentage points** of macro-F1.
- Model is **~3 k parameters**, runs in **< 1 ms** per sample on CPU — fits any substation.

## Reproduce

```bash
# (optional) create a Python 3.10+ environment
pip install -r requirements.txt   # if a requirements file is provided
jupyter notebook T2_DGA_Transfer_Learning_Simulation.ipynb
```

## Limitations

All numerical results are from a **simulation study with synthetic target data** drawn from published IEC 60599 distributions plus an author-chosen tropical shift (CO × 1.4–2.0, CO₂ × 1.3–1.8). Validation on real PLN field records is essential before operational deployment.

## License

The code is released under the MIT License unless otherwise noted. The paper PDF and figures remain © the authors.

## Cite

```bibtex
@inproceedings{hakim2026tropicaldga,
  title     = {Tropical Domain Shift in Dissolved Gas Analysis:
               Transfer Learning from the IEC TC 10 Benchmark
               to PLN Indonesia for Transformer Fault Classification},
  author    = {Hakim, Fakhri and others},
  booktitle = {Proc. ICT-PEP},
  year      = {2026}
}
```
