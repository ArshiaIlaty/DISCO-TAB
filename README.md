# DISCO-TAB

**DIScriminator-guided COntrol for TABular synthesis** — a hierarchical
reinforcement-learning framework that couples a fine-tuned LLM generator with a
multi-granular discriminator ensemble and inverse-frequency reward shaping (IFRS)
to synthesize high-fidelity, clinically-constrained tabular data.

This repository contains the full, reproducible pipeline and benchmark used in the
paper *"DISCO-TAB: A Hierarchical Reinforcement Learning Framework for Clinically
Constrained Tabular Data Synthesis"* (under review).

📄 **Paper:** preprint link to be added here upon release.
💻 **Code & data:** this repository — <https://github.com/ArshiaIlaty/DISCO-TAB>

---

## What's here

```
src/        the pipeline (data, serialization, constraint discovery,
            discriminators, PPO+IFRS training, metrics, baselines, assembly)
results/    per-run metric JSONs (every generator × dataset × seed)
tables/     assembled benchmark tables (CSV + LaTeX) and the comprehensive PDF
```

## Method (src/)

| Component | File |
|---|---|
| Schema-aware serialization (standardize, `[MISSING]`, name-aliasing, target-early) | `serialize.py` |
| Automated constraint discovery (Pearson / Cramér's V / correlation-ratio → P_crit) | `constraints.py` |
| Hierarchical discriminators (token BiLSTM, sentence DistilBERT, feature/row MLP) | `discriminators.py` |
| Generator SFT + reward-shaped PPO + IFRS | `train.py` |
| End-to-end run orchestration | `run_experiment.py`, `run_all.py` |
| Evaluation: TSTR utility, fidelity (JS/KS/KL/Hellinger/Pearson), DCR, FAITH | `metrics.py` |
| Enriched metrics: C2ST, MMD, Fréchet-PCA, PRDC, SDMetrics, MIA, NNAA, … | `metrics_ext.py`, `evaluate_ext.py` |
| Baselines: SDV (CTGAN/TVAE/GaussianCopula), synthcity zoo, TabDDPM, LLM | `baselines*.py` |
| Significance tests (Wilcoxon / Friedman / Nemenyi / Holm) | `stats.py` |
| Table / figure / report assembly | `assemble_tables.py`, `make_comprehensive.py`, `make_tex_report.py` |

## Benchmark

- **24 datasets** — 6 UCI/clinical + 3 finance/social + 10 SurvSet clinical cohorts
  (event→binary) + 5 TabArena/OpenML sets. All fetched reproducibly:
  `python src/fetch_datasets.py && python src/fetch_new_datasets.py`.
- **~13 generators** across every architecture family: GAN (CTGAN, ADS-GAN, PATE-GAN),
  VAE (TVAE, RTVAE), diffusion (TabDDPM), normalizing flow (NFlow), tree (ARF),
  Bayesian/graphical (BayesNet), Gaussian copula, LLM (GReaT / RealTabFormer),
  and **DISCO-TAB**.
- **~37 metrics** across utility, statistical fidelity, privacy/disclosure,
  diversity/coverage, detection, SDMetrics, and the FAITH framework.

The full comparison is in `tables/comprehensive/benchmark_tables.pdf`
(all generators × 24 datasets × all metrics).

## Reproduce

```bash
# 1. environment (GPU recommended; GPT-2 small backbone)
pip install torch transformers scikit-learn scipy pandas numpy ucimlrepo \
            scikit-posthocs sdmetrics SurvSet
# baselines use isolated envs (SDV, synthcity) — see src/baselines*.py headers

# 2. data
python src/fetch_datasets.py
python src/fetch_new_datasets.py

# 3. train DISCO-TAB (one dataset) / full matrix
python src/run_experiment.py --dataset heart_failure --seed 0
python src/run_all.py --mode main --datasets <...> --seeds 0 1 2

# 4. baselines + metrics + tables
python src/evaluate_baselines.py
python src/evaluate_ext.py
python src/stats.py && python src/make_comprehensive.py
```

## Hardware note
The released runs were produced on a single NVIDIA T4 (shared); the design is
parameter-efficient (<6 GB VRAM) and runs on consumer GPUs.

## Citation
```bibtex
@article{ilaty_discotab_2026,
  title   = {DISCO-TAB: A Hierarchical Reinforcement Learning Framework for
             Clinically Constrained Tabular Data Synthesis},
  author  = {Ilaty, Arshia and others},
  year    = {2026},
  note    = {Under review. Code: \url{https://github.com/ArshiaIlaty/DISCO-TAB}}
}
```

## License
MIT — see `LICENSE`.
