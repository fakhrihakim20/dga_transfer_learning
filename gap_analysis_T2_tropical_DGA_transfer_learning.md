# Gap Analysis: Tropical-Climate DGA Fault Classification with Transfer Learning (IEC TC 10 → PLN)

**Date:** 2026-05-03  
**Scope:** Systematic literature gap analysis for a DGA fault classifier using transfer learning from the IEC TC 10 benchmark to a small PLN Indonesia dataset, addressing both class imbalance and tropical-climate domain shift.

> **Verification notice:** This report was generated from web search snippets in a single session. No full-text papers were read (all WebFetch attempts failed). Numerical claims (accuracy percentages, sample counts, publication statistics) are marked with inline caveats (*italicized*) at point of use and must be verified against the actual publications before citing. One BibTeX DOI is a placeholder. "Not found" means not retrieved in this search — see Limitations for coverage gaps. All gap claims are qualified with confidence levels and per-gap caveats.

---

## Research Landscape

DGA-based transformer fault classification using machine learning is a mature and active field, with 132+ publications identified in the most recent systematic review (Dladla & Thango, 2025; *number from search-tool AI summary of query "SMOTE ENN DGA transformer fault Putra Prasojo 2024 2025" — not from raw paper text; verify in full SLR*). The dominant approaches are classical ML (SVM, Random Forest, KNN) applied to the IEC TC 10 benchmark or proprietary utility datasets. Deep learning (1D-CNN, LSTM, DBN) has gained traction since 2021. Class imbalance is widely recognized and addressed via SMOTE variants, GANs, and diffusion models. However, **domain adaptation for DGA appears to be an emerging niche based on English-language results retrieved in this session** (*this conclusion rests on incomplete coverage — CNKI, paywalled IEEE Xplore, and exhaustive arXiv enumeration were not performed*) — the earliest dedicated paper found is Mahmoodiyan et al. (arXiv, May 2025), and **no English-language publication found in this analysis explicitly models tropical-climate domain shift** as the source of distribution mismatch between a global benchmark and a regional utility dataset. A separate cross-domain study (MWBCAN, ScienceDirect Oct 2025) addresses transformer domain shift via vibration signals rather than DGA, confirming that cross-domain methods are entering the power-transformer space broadly but have not yet been applied to climate-driven DGA shift.

> **Important caveat:** China accounts for ~53% of DGA+ML publications (Dladla & Thango, 2025; *percentage from search snippet — full SLR not read*). Chinese-language journals indexed in CNKI/Wanfang but not in Web of Science may contain relevant work on transfer learning for DGA that this English-language search did not retrieve. See the Limitations section at the end of this report.

---

## Coverage Table

| Approach Category | Representative Papers | What They Cover | What's Missing |
|---|---|---|---|
| **Classical ML on DGA (SVM, RF, KNN)** | Aizpurua et al. (2017); HKIE 2024 study; Dladla & Thango (2025 SLR) | SVM (32%), ANN (17%), KNN (12%) on IEC TC 10 (*percentages from search-tool AI summary of query "domain adaptation dissolved gas analysis cross-domain transfer fault diagnosis 2023 2024 2025" — not from raw paper text; verify in full SLR*); RF achieves F1=0.88 (*from HKIE 2024 search snippet — verify*) | No transfer to small regional datasets; no climate-aware features |
| **Deep Learning for DGA (CNN, 1D-CNN, LSTM)** | Rao et al. (2023, SAGE); GWO-1D-CNN (2025, Arabian J.); ResNet+KD (2025, Entropy) | 1D-CNN with >95% accuracy (*aggregate claim from multiple search snippets — verify against specific papers*); knowledge distillation for lightweight models | Models trained & tested on same-domain data; no cross-domain evaluation |
| **Class Imbalance Handling** | Wang et al. (2024, Sci. Rep.) — SMOTE+NGO-GBDT; Li (2022, IET) — GAN; CWGAN-GP (2024); Diffusion-TS (2025, IEEE); Putra & Prasojo (2024, ICPERE) | SMOTE, ADASYN, B-SMOTE-DNN, GAN, VAE, Diffusion-TS for DGA augmentation | Oversampling within single domain only; no combined imbalance + domain shift |
| **Domain Adaptation / Transfer Learning for DGA** | Mahmoodiyan et al. (2025, arXiv) — MMD-CORAL; semi-supervised TL (2022, PMC); self-strengthening pretraining (2023, ScienceDirect); MWBCAN (2025, ScienceDirect) — vibration-based cross-domain | Feature-weighted MMD-CORAL across transformer types (reported +7.9% vs fine-tuning — *unverified; paper not fully read, number from search snippet*); semi-supervised TL; cross-domain via vibration signals | **No climate-driven domain shift**; no IEC TC 10→specific utility transfer; no tropical data; MWBCAN uses vibration not DGA |
| **Indonesian / Tropical DGA Studies** | Prasojo et al. (2020, Energies) — fuzzy DGA for 448 PLN transformers (*sample count from search snippet — verify*); Prasojo & Suwarno (2018) — SVM for paper insulation; Putra et al. (2025, Energies) — SMOTE for health index; Surawijaya et al. (2020, IEEE) — geothermal GSU case study | DGA interpretation, health index, DP estimation for PLN transformers; note elevated CO/CO₂ baselines | **No ML classifier specifically trained for tropical domain** (*based on retrieved English-language literature*); no transfer learning; no domain shift quantification |
| **Interpretability / XAI for DGA** | SHAP-BES-LGBM (2025, Energy Informatics); SHAP feature selection (2024); LIME+DL (2025) | SHAP for feature importance in DGA classifiers; top-13 feature selection | Not applied to transfer learning models; no explanation of *what shifts* between domains |
| **LLM / Foundation Models for DGA** | Li et al. (2026, Springer) — KD from LLM for DGA | Knowledge distillation from large language model to lightweight DGA classifier | Proof-of-concept; no domain adaptation component |
| **Benchmarking Studies on IEC TC 10** | arXiv:2505.06295 (2025); HKIE 2024; Kim (2013, IEEE) original TC 10 methods | Systematic comparison of ML/DL on IEC TC 10 (628 samples) | Benchmarks treat IEC TC 10 as a single domain; no out-of-distribution evaluation |

---

## Identified Gaps

| # | Gap Description | Confidence | Evidence of Absence | Why It Matters |
|---|---|---|---|---|
| **G1** | **No study found that models tropical-climate domain shift for DGA classification.** No English-language paper retrieved in this analysis frames the elevated CO/CO₂ baselines in tropical-humidity environments (documented by Prasojo/Suwarno) as a *distribution shift* problem solvable by domain adaptation. | **HIGH** (with caveats — see Limitations) | Searched 3+ phrasings: "tropical climate domain shift DGA", "humidity DGA threshold region", "equatorial transformer DGA", "domain shift DGA Indonesia Malaysia Thailand India" across Google Scholar, arXiv, IEEE Xplore. The MMD-CORAL paper (May 2025) discusses distribution shift due to transformer type and operational settings but does not mention climate. Prasojo et al. document the tropical effect but do not frame it as domain adaptation. **Caveat:** Chinese-language journals and IEEE Xplore paywalled content were not exhaustively searched. | Tropical utilities (PLN, TNB, MEA) apply IEC 60599 thresholds derived from mixed-climate data, risking misclassification. Quantifying and bridging this shift could improve diagnostic accuracy for transformer fleets in tropical zones. |
| **G2** | **No transfer learning study found that uses IEC TC 10 as the source domain and a specific utility's small dataset as the target.** Existing TL/DA work for DGA appears to use proprietary or synthetic domain pairs. | **HIGH** (with caveats) | Searched "IEC TC 10 transfer learning", "IEC TC 10 fine-tuning target domain", "pretrained IEC TC10 neural network" across arXiv, Google Scholar, MDPI. The IEC TC 10 appears to be used only as a single-domain benchmark. Mahmoodiyan et al. (2025) reportedly use proprietary datasets (full paper not verified). **Caveat:** arXiv coverage was limited to top search results; very recent preprints may have been missed. | IEC TC 10 is the only freely available, labeled DGA benchmark (628 samples, 4–7 fault classes). Using it as the pretrain source would enable *reproducible* transfer learning research. |
| **G3** | **No study found that simultaneously addresses class imbalance AND domain shift for DGA.** SMOTE/GAN papers appear to work within a single domain; domain adaptation papers appear to assume balanced data or ignore imbalance. | **MEDIUM–HIGH** | Cross-checked SMOTE+DGA papers (Wang 2024, Li 2022, Putra 2024) — none mention domain shift in retrieved abstracts/summaries. Cross-checked DA papers (Mahmoodiyan 2025, semi-supervised TL 2022) — SMOTE or imbalance handling not mentioned in retrieved summaries. **Caveat:** full-text reading of these papers was not performed; imbalance handling may be discussed in methods sections not captured by search snippets. | Real-world DGA data exhibits both problems simultaneously: the PLN dataset is small (50–200 samples), likely severely imbalanced (PD and D1 are rare), and shifted from IEC TC 10 distributions. |
| **G4** | **No simple MLP/1D-CNN transfer learning pipeline found for DGA** (vs. complex domain-adversarial or MMD methods). | **MEDIUM** | Searched "MLP transfer learning DGA", "1D-CNN fine-tuning DGA pretrained". Transfer learning for tabular DGA data with a simple freeze-and-retrain paradigm was not found. The field appears to jump from no-TL to sophisticated DA methods. **Caveat:** this gap is less defensible — a simple baseline may be considered too incremental for some venues. | A simple pretrain→fine-tune pipeline is more accessible to power engineering practitioners at utilities like PLN. It also establishes a lower-bound baseline for future, more complex DA methods. |
| **G5** | **No interpretability analysis (SHAP/LIME) found applied to a transfer-learned DGA model** to explain *what* shifts between source and target domains. | **MEDIUM** | SHAP papers for DGA (2024–2025) found in this search apply only to single-domain models. No paper found uses SHAP to compare feature importance pre- vs. post-transfer or to visualize domain shift. | Showing that CO/CO₂ features shift most between IEC TC 10 and tropical data would provide an *interpretable* validation of the domain-shift hypothesis and build trust with utility engineers. |

---

## Candidate Papers to Cite

> **Note:** BibTeX entries below are reconstructed from search results and cross-checked against available metadata. Entries marked ★ are verified against publisher metadata; others are best-effort from search results.

### Domain Adaptation & Transfer Learning for DGA

```bibtex
@article{Mahmoodiyan2025MMDCORAL,
  author    = {Hootan Mahmoodiyan and Maryam Ahang and Mostafa Abbasi and Homayoun Najjaran},
  title     = {Feature-Weighted {MMD-CORAL} for Domain Adaptation in Power Transformer Fault Diagnosis},
  journal   = {arXiv preprint arXiv:2505.14896},
  year      = {2025},
  note      = {UNVERIFIED: Full paper not read in this session (WebFetch failed). Numbers (+7.9\% over fine-tuning) are from search-result snippets only. Verify before citing.}
}

@article{MWBCAN2025,
  title     = {Power transformer cross-domain fault diagnosis method based on dynamic weight balancing of multidimensional indicators and deep adversarial networks},
  journal   = {Measurement},
  year      = {2025},
  publisher = {Elsevier},
  url       = {https://www.sciencedirect.com/science/article/abs/pii/S0263224125027848},
  note      = {DOI NOT RETRIEVED — look up via PII S0263224125027848. Uses vibration signals (not DGA) for cross-domain transformer fault diagnosis. Confirms DA is entering the power-transformer space but via a different modality.}
}

@article{SemiSupervisedTL2022,
  title     = {Fault Diagnosis for Power Transformers through Semi-Supervised Transfer Learning},
  journal   = {Sensors},
  year      = {2022},
  note      = {PMC9231397; semi-supervised transfer learning for DGA with limited labels}
}

@article{SelfStrengthening2023,
  title     = {Power transformer fault diagnosis based on a self-strengthening offline pre-training model},
  journal   = {Engineering Applications of Artificial Intelligence},
  year      = {2023},
  doi       = {10.1016/j.engappai.2023.107370},
  note      = {ResVAE + EL with pre-training for multi-class-imbalanced DGA data}
}
```

### Systematic Reviews & Surveys

```bibtex
@article{DladlaThango2025SLR, ★
  author    = {Sifiso Dladla and Israel Thango},
  title     = {Fault Classification in Power Transformers via Dissolved Gas Analysis and Machine Learning Algorithms: A Systematic Literature Review},
  journal   = {Applied Sciences},
  volume    = {15},
  number    = {5},
  pages     = {2395},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/app15052395}
}

@article{ArshadSOTA2023,
  title     = {The State of the Art in transformer fault diagnosis with artificial intelligence and Dissolved Gas Analysis: A Review of the Literature},
  journal   = {arXiv preprint arXiv:2304.11880},
  year      = {2023}
}
```

### Class Imbalance for DGA

```bibtex
@article{Wang2024SMOTENGOGBDT, ★
  author    = {Li-Zhong Wang and Jian-Fei Chi and Ye-Qiang Ding and Hai-Yan Yao and Qiang Guo and Hai-Qi Yang},
  title     = {Transformer fault diagnosis method based on {SMOTE} and {NGO-GBDT}},
  journal   = {Scientific Reports},
  volume    = {14},
  pages     = {7609},
  year      = {2024},
  publisher = {Nature},
  doi       = {10.1038/s41598-024-57509-w}
}

@article{Li2022GAN,
  author    = {Li, J. and others},
  title     = {Addressing imbalance of sample datasets in dissolved gas analysis by data augmentation: Generative adversarial networks},
  journal   = {IET Generation, Transmission \& Distribution},
  year      = {2022},
  doi       = {10.1049/gtd2.12610}
}

@inproceedings{DiffusionTS2025,
  title     = {Diffusion-{TS} Guided Small Sample Data Augmentation for Power Transformer Fault Diagnosis},
  booktitle = {IEEE Conference},
  year      = {2025},
  note      = {IEEE Xplore 11065524; diffusion model for synthetic DGA generation}
}
```

### Indonesian / Tropical DGA Studies

```bibtex
@article{Prasojo2020Fuzzy, ★
  author    = {Rahman Adi Prasojo and Hery Gumilang and Suwarno and Nana Ulya Maulidevi and Bambang Anggoro Soedjarno},
  title     = {A Fuzzy Logic Model for Power Transformer Faults' Severity Determination Based on Gas Level, Gas Rate, and Dissolved Gas Analysis Interpretation},
  journal   = {Energies},
  volume    = {13},
  number    = {4},
  pages     = {1009},
  year      = {2020},
  publisher = {MDPI},
  doi       = {10.3390/en13040109}
}

@article{PutraPrasojo2025HealthIndex, ★
  author    = {Muhamad Afrizal Amien Putra and Rahman Adi Prasojo and Suwarno and others},
  title     = {Improving Transformer Health Index Prediction Performance Using Machine Learning Algorithms with a Synthetic Minority Oversampling Technique},
  journal   = {Energies},
  volume    = {18},
  number    = {9},
  pages     = {2364},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/en18092364}
}

@inproceedings{Surawijaya2020Geothermal,
  author    = {Surawijaya, Adjat and others},
  title     = {Diagnosis of Power Transformer Condition using Dissolved Gas Analysis Technique: Case Studies at Geothermal Power Plants In Indonesia},
  booktitle = {IEEE Conference (ICPERE)},
  year      = {2020},
  doi       = {10.1109/ICPERE48726.2019.9011106}
}
```

### IEC TC 10 Dataset & Benchmarks

```bibtex
@article{Kim2013IECTC10,
  author    = {Y. Kim and others},
  title     = {New methods of {DGA} diagnosis using {IEC TC 10} and related databases {Part 1}: application of gas-ratio combinations},
  journal   = {IEEE Transactions on Dielectrics and Electrical Insulation},
  year      = {2013}
}

@misc{IEEEDataPort_DGA,
  title     = {{DGA} dataset},
  howpublished = {IEEE DataPort},
  url       = {https://ieee-dataport.org/documents/dga-dataset},
  note      = {IEC TC 10 benchmark: 628 labeled samples with H2, CH4, C2H6, C2H4, C2H2}
}

@article{MDPI2025DGABenchmark,
  title     = {Dissolved Gas Analysis for Fault Prediction in Power Transformers Using Machine Learning Techniques},
  journal   = {Applied Sciences},
  volume    = {15},
  number    = {1},
  pages     = {118},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/app15010118}
}
```

### Interpretability for DGA

```bibtex
@article{SHAPLGBM2025,
  title     = {Transformer fault diagnosis using machine learning: a method combining {SHAP} feature selection and intelligent optimization of {LGBM}},
  journal   = {Energy Informatics},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1186/s42162-025-00519-3}
}
```

### Deep Learning / 1D-CNN for DGA

```bibtex
@article{GWO1DCNN2025,
  title     = {Intelligent Fault Diagnosis in Oil-Immersed Transformers: A Deep Learning {1D-CNN} Framework with Gray Wolf Optimizer ({GWO})},
  journal   = {Arabian Journal for Science and Engineering},
  year      = {2025},
  doi       = {10.1007/s13369-025-10942-z}
}

@article{Rao2023CNN,
  author    = {Shaowei Rao and Shiyou Yang and Mauro Tucci and Sami Barmada},
  title     = {Convolutional neural networks applied to dissolved gas analysis for power transformers condition monitoring},
  journal   = {International Journal of Applied Electromagnetics and Mechanics},
  year      = {2023},
  doi       = {10.3233/JAE-230011}
}
```

### Knowledge Distillation / LLM for DGA

```bibtex
@inproceedings{Li2026KDLLM,
  author    = {X. Li and L. Zeng and Q. Zeng and Y. Lu and Y. Guo and L. Yang},
  title     = {{DGA}-Based Power Transformer Fault Diagnosis via Knowledge Distillation of Large Language Model},
  booktitle = {AIoTSys 2025, Lecture Notes in Electrical Engineering},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-981-95-2581-2\_22}
}
```

---

## Summary of Novelty Position

Based on this analysis, your proposed study appears to occupy an **unoccupied intersection** in the English-language literature retrieved in this search:

```
                    ┌─────────────────────────────────┐
                    │   Domain Adaptation for DGA      │
                    │   (Mahmoodiyan 2025 — operational │
                    │    shift, NOT climate shift)      │
                    └───────────┬─────────────────────┘
                                │
                                │  YOUR PAPER SITS HERE
                                │  (climate-driven shift +
                                │   class imbalance +
                                │   IEC TC 10 → PLN TL)
                                │
   ┌────────────────┐           │         ┌──────────────────┐
   │ Class Imbalance │◄──────────┼────────►│ Tropical DGA     │
   │ for DGA         │           │         │ (Prasojo/Suwarno │
   │ (SMOTE, GAN,    │           │         │  — no ML TL)     │
   │  Diffusion-TS)  │           │         └──────────────────┘
   └────────────────┘           │
                                │
                    ┌───────────┴─────────────────────┐
                    │   TL on IEC TC 10 as Source      │
                    │  (not found in this search)       │
                    └─────────────────────────────────┘
```

**Three strongest novelty claims (subject to verification — see Limitations):**

1. **Framing tropical-climate DGA threshold differences as a domain-shift problem** addressable by transfer learning — not found in retrieved literature (G1).
2. **Using IEC TC 10 as a pretrain source domain** for transfer to a specific utility's small target dataset — not found in retrieved literature (G2).
3. **Combined treatment of class imbalance + domain shift** for DGA fault classification — not found in retrieved literature (G3).

**Risk assessment:** Gap G1 is the most defensible and also the most impactful — it connects Prasojo/Suwarno's empirical observation (tropical baselines exceed IEC 60599) to a well-established ML methodology (domain adaptation/transfer learning). Based on search snippets (full paper not read), the MMD-CORAL paper (May 2025) appears to address operational/equipment-type shift but not climate effects, which would leave your climate-shift framing unchallenged. **However, you must read the full Mahmoodiyan et al. paper to confirm this before claiming the gap in your manuscript.**

---

## Limitations of This Analysis

This gap analysis was conducted via automated web search (Google, arXiv, Semantic Scholar, IEEE Xplore search previews, ResearchGate) during a single session on 2026-05-03. The following limitations apply:

1. **Unverified paper details.** Multiple WebFetch attempts to read full papers (arXiv, PMC, MDPI) failed due to connection errors or 403 blocks. Numerical claims about Mahmoodiyan et al. (2025) — specifically "+7.9% improvement over fine-tuning" and "+2.2% over MMD-CORAL" — are sourced from search-result snippets and arXiv abstract summaries, **not** from reading the full paper. These numbers must be verified before citing.

2. **Non-English literature not covered.** China accounts for ~53% of DGA+ML publications (per Dladla & Thango, 2025). Chinese-language journals indexed in CNKI, Wanfang, and CSCD — but not in Web of Science or Scopus — may contain relevant work on transfer learning or domain adaptation for DGA. Similarly, Japanese (IEEJ), Korean, and Indonesian-language publications were not searched systematically.

3. **Paywalled content.** IEEE Xplore, ScienceDirect, and Springer full texts behind paywalls were not read. Abstracts and search snippets were used as proxies. Methods sections of retrieved papers may contain domain adaptation or class imbalance handling not captured in abstracts.

4. **Incomplete arXiv coverage.** arXiv searches covered top-ranked results for relevant queries but did not exhaustively enumerate all Feb–May 2025 preprints in cs.LG, eess.SP, or cs.AI that might combine DGA with transfer learning. Very recent preprints (April–May 2025) may have been missed.

5. **"Absence of evidence ≠ evidence of absence."** Each gap identified in this report means "not found in this session's search results." It does **not** mean the work definitively does not exist. Before claiming a gap in your manuscript, you should:
   - Read the full text of Mahmoodiyan et al. (2025, arXiv:2505.14896) to confirm it does not address climate shift
   - Search CNKI (中国知网) for Chinese-language DGA + transfer learning papers
   - Check IEEE Xplore's full-text search behind institutional access
   - Check recent proceedings of CIGRE, CEIDP, CMD, and ICPADM conferences (2024–2025)
   - Search Scopus/Web of Science with targeted queries

6. **BibTeX entries are best-effort.** Entries marked ★ were cross-checked against publisher metadata visible in search results. Others are reconstructed from search snippets and may have errors in author names, page numbers, or DOIs. All entries should be verified against the actual publications before submission.

7. **Domain-specific conference proceedings.** Power engineering conferences (CIGRE, CEIDP, ICPERE, CMD, ICPADM) often publish work that is not well-indexed in general search engines. These venues are particularly relevant for Indonesian/Southeast Asian DGA studies and may contain unpublished or recently published work filling the identified gaps.

---

## Sources

- [DGA dataset — IEEE DataPort](https://ieee-dataport.org/documents/dga-dataset)
- [Kim et al. — IEC TC 10 DGA methods (Semantic Scholar)](https://www.semanticscholar.org/paper/New-methods-of-DGA-diagnosis-using-IEC-TC-10-and-1:-Kim-Kim/b7860e53dffc4f91b7e1455ba2d67e7b87d177e5)
- [Dladla & Thango 2025 — SLR on DGA+ML (MDPI)](https://www.mdpi.com/2076-3417/15/5/2395)
- [Mahmoodiyan et al. 2025 — MMD-CORAL for DA in DGA (arXiv)](https://arxiv.org/abs/2505.14896)
- [Wang et al. 2024 — SMOTE+NGO-GBDT (Sci. Rep.)](https://www.nature.com/articles/s41598-024-57509-w)
- [Prasojo et al. 2020 — Fuzzy DGA for PLN (MDPI Energies)](https://www.mdpi.com/1996-1073/13/4/1009)
- [Putra & Prasojo 2025 — SMOTE Health Index (MDPI Energies)](https://www.mdpi.com/1996-1073/18/9/2364)
- [Li et al. 2026 — KD from LLM for DGA (Springer)](https://link.springer.com/chapter/10.1007/978-981-95-2581-2_22)
- [SHAP+LGBM for DGA 2025 (Energy Informatics)](https://link.springer.com/article/10.1186/s42162-025-00519-3)
- [GWO-1D-CNN 2025 (Arabian J.)](https://link.springer.com/article/10.1007/s13369-025-10942-z)
- [Arshad 2023 — SOTA survey (arXiv)](https://arxiv.org/abs/2304.11880)
- [Diffusion-TS for DGA 2025 (IEEE Xplore)](https://ieeexplore.ieee.org/abstract/document/11065524/)
- [Li 2022 — GAN for DGA imbalance (IET)](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/gtd2.12610)
- [Self-strengthening pretraining 2023 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S095219762301326X)
- [Semi-supervised TL for DGA 2022 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9231397/)
- [MDPI Appl. Sci. 2025 — DGA ML benchmark](https://www.mdpi.com/2076-3417/15/1/118)
- [Surawijaya et al. 2020 — Indonesia geothermal DGA (IEEE)](https://ieeexplore.ieee.org/document/9011106/)
- [Rao et al. 2023 — CNN for DGA (SAGE)](https://journals.sagepub.com/doi/abs/10.3233/JAE-230011)
- [SMOTE+GBDT hybrid 2025 (Arabian J.)](https://link.springer.com/article/10.1007/s13369-025-10772-z)
- [KD+ResNet for DGA 2025 (Entropy)](https://www.mdpi.com/1099-4300/27/7/669)
- [Riemannian VAE for DGA augmentation 2025 (Springer)](https://link.springer.com/chapter/10.1007/978-981-96-4675-3_63)
- [MWBCAN cross-domain transformer fault via vibration 2025 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0263224125027848)
