# Execution Logs for Evo-MCTS

This directory contains complete execution logs, evaluation traces, and baseline comparison data for the paper:

**"Evo-MCTS: LLM-Guided Evolutionary Monte Carlo Tree Search for Automated Algorithm Discovery in Gravitational Wave Detection"**

Nature Communications submission NCOMMS-25-84133-T

---

## Purpose

These materials fulfill reproducibility commitments made in response to reviewer comments:

| Reviewer Comment | Section | Materials Provided |
|------------------|---------|-------------------|
| **Comment 1.2** (Reproducibility) | Supplementary S6 | Breakthrough node JSONs with complete prompt→code→evaluation traces |
| **Comment 1.3** (Computational Cost) | Supplementary S7 | All 5 run logs, budget sensitivity data |
| **Comment 1.4** (Comparative Fairness) | Supplementary S8 | MCTS-AHD & ReEvo: modifications, logs, configs |

---

## Directory Structure

```
execution_logs/
├── README.md                          # This file
├── evo-mcts/
│   ├── run1_april_2025/
│   │   ├── merged_log.log             # Primary discovery run (638 evals, April 13-17, 2025)
│   │   └── breakthrough_nodes/        # 6 key algorithmic breakthrough JSONs
│   ├── run2/merged_log.log            # Production run 2
│   ├── run3/merged_log.log            # Production run 3
│   ├── run4/merged_log.log            # Production run 4
│   └── run5/merged_log.log            # Production run 5
├── baselines/
│   ├── mcts-ahd/
│   │   ├── GW_MODIFICATIONS.md        # Documentation of GW-specific modifications
│   │   ├── cfg/problem/gw_mlgwsc1.yaml
│   │   ├── problems/gw_mlgwsc1/       # Evaluation pipeline + algorithm variants
│   │   ├── prompts/gw_mlgwsc1/        # Domain knowledge prompts
│   │   └── logs/                      # 9 job execution logs
│   └── reevo/
│       ├── GW_MODIFICATIONS.md        # Documentation of GW-specific modifications
│       ├── cfg/problem/gw_mlgwsc1.yaml
│       ├── problems/gw_mlgwsc1/       # Evaluation pipeline + algorithm variants
│       ├── prompts/gw_mlgwsc1/        # Domain knowledge prompts
│       └── logs/                      # 6 job execution logs
├── ablation/
│   ├── ablation_run_values.json       # Domain knowledge ablation data (Figure 6c)
│   └── ablation_run_summary.csv       # Ablation statistics
├── analysis/
│   ├── budget_sensitivity_data.txt    # Milestone performance (Section S7)
│   ├── prior_detected_distribution_0.05.png  # Chirp mass analysis (Section S9)
│   └── statistics_comment_111_results.txt
└── algorithms/
    ├── pipeline_v1.py                 # Baseline/seed algorithm
    └── pipeline_v2.py                 # PT-4 final implementation
```

---

## Key Performance Metrics

### Evo-MCTS (5 Production Runs)

| Run | Evaluations | Best Fitness (Mpc) | Discovery |
|-----|-------------|-------------------|-----------|
| **run1** | 638 | **5502.95** | PT-4 at eval 486 (5241.37 Mpc) |
| run2 | ~650 | ~3400 | - |
| run3 | ~900 | ~4000 | - |
| run4 | ~500 | ~1800 | - |
| run5 | ~300 | ~950 | - |

**Multi-run statistics**: 2670.37 ± 1879.93 Mpc (mean ± std, 5 runs)

### Baseline Frameworks

| Framework | Runs | Best (Mpc) | Mean ± Std (Mpc) | LLM Calls |
|-----------|------|------------|------------------|-----------|
| **MCTS-AHD** | 9 | 2494.10 | 1235.82 ± 485.14 | 488.2 ± 70.7 |
| **ReEvo** | 5 | 2899.40 | 1624.40 ± 766.79 | 596.0 ± 76.0 |

### Performance Improvements

- **Evo-MCTS vs MCTS-AHD**: +110.2% (PT-4: 5241.37 vs 2494.10 Mpc)
- **Evo-MCTS vs ReEvo**: +80.8% (PT-4: 5241.37 vs 2899.40 Mpc)
- **Evo-MCTS vs Sage**: +20.2% (PT-4: 5241.37 vs 4359.27 Mpc)

---

## Breakthrough Nodes (Section S6)

The `evo-mcts/run1_april_2025/breakthrough_nodes/` directory contains complete prompt→code→evaluation traces for 6 key algorithmic breakthroughs:

| File | Eval | Operator | Fitness (Mpc) | Innovation |
|------|------|----------|---------------|------------|
| `node_012_*.json` | 12 | PM (m7) | 936.51 | Adaptive whitening foundation |
| `node_028_*.json` | 28 | SC (m3) | 933.69 | Cross-detector coherence |
| `node_140_*.json` | 140 | SC (m3) | 2241.96 | Phase-aligned coherence + CWT |
| `node_151_*.json` | 151 | PC (e3) | 2612.77 | Baseline detrending + curvature |
| `node_333_*.json` | 333 | PC (e3) | 4559.26 | **PT-3**: Gradient-adaptive whitening |
| `node_486_*.json` | 486 | PC (e3) | 5241.37 | **PT-4**: Final optimized algorithm |

Each JSON file contains:
- `user_content`: Complete LLM prompt (8K-27K characters)
- `code`: Full Python implementation
- `reflection`: LLM's analysis of the generated code
- `fitness`: Evaluation result (AUC in Mpc)

---

## Fair Comparison Evidence (Section S8)

All baseline frameworks were executed under identical conditions:

1. **LLM Model**: o3-mini-medium (o3-mini-2025-01-31), temperature 1.0
2. **Dataset**: MLGWSC-1 Set 4 (7-day train, 1-day test)
3. **Fitness Metric**: AUC = ∫d_L d(log10 FAR) in Mpc
4. **Hardware**: 96-core cluster, 72 parallel workers
5. **LLM Budget**: ~490-650 calls per run (fair comparison)

The `GW_MODIFICATIONS.md` files in each baseline directory document all modifications made to adapt the frameworks for gravitational wave detection. Git history can verify changes against upstream repositories:
- MCTS-AHD: https://github.com/zz1358m/MCTS-AHD-master
- ReEvo: https://github.com/ai4co/reevo

---

## Computational Cost (Section S7)

From `run1_april_2025/merged_log.log`:

| Metric | Value |
|--------|-------|
| Wall-clock time | 103.96 hours (4.33 days) |
| Calendar span | April 13-17, 2025 |
| Total evaluations | 638 |
| Success rate | 54.2% (346/638) |
| Mean successful eval time | 588.5 seconds |
| LLM API calls | 1,842 requests |
| Estimated API cost | ~$200 |

Budget sensitivity milestones (from `analysis/budget_sensitivity_data.txt`):
- 69 evals: PT-1 discovered (1873.87 Mpc)
- 91 evals: PT-2 discovered (3159.50 Mpc)
- 333 evals: PT-3 discovered (4559.26 Mpc)
- 486 evals: PT-4 discovered (5241.37 Mpc)
- 637 evals: Final best (5502.95 Mpc)

---

## Verifying Results from Logs

### Extract fitness values from Evo-MCTS log:
```bash
grep "Objective value" merged_log.log | tail -10
```

### Find PT-4 discovery:
```bash
grep "5241.37" merged_log.log
```

### Count evaluations:
```bash
grep "Eval_times:" merged_log.log | tail -1
```

### Extract baseline best results:
```bash
# MCTS-AHD
grep "best_fitness" logs/log_*.log | sort -t: -k3 -n | tail -1

# ReEvo
grep "fitness" logs/log_*.log | sort -t: -k3 -n | tail -1
```

---

## Citation

If you use these execution logs in your research, please cite the main Evo-MCTS paper:

```bibtex
@article{wang2025automated,
  title={Automated Algorithmic Discovery for Gravitational-Wave Detection Guided by LLM-Informed Evolutionary Monte Carlo Tree Search},
  author={He Wang and Liang Zeng},
  year={2025},
  eprint={2508.03661},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2508.03661}
}
```

---

## License

These execution logs are provided under the same license as the main repository (GPL-3.0). See the repository root `LICENSE` file for details.

For questions or issues, please open an issue at: https://github.com/iphysresearch/evo-mcts/issues
