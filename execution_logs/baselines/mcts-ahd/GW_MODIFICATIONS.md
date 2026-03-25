# Gravitational Wave Detection Adaptations

## Original Framework
- Repository: https://github.com/zz1358m/MCTS-AHD-master
- Paper: Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design (ICML 2025)
- Authors: Zhi Zheng, Jie Zhang, Yueqing Zhang, Yan Jin
- ArXiv: https://arxiv.org/pdf/2501.08603
- License: MIT (2024)

## Modifications for MLGWSC-1 Benchmark

### New Problem Configuration
- **Config file**: `cfg/problem/gw_mlgwsc1.yaml`
- **Problem type**: Constructive heuristic design
- **Objective**: Maximize AUC metric for gravitational wave detection
- **Problem size parameter**: 22.5 (training dataset identifier)

### Files Added
1. `cfg/problem/gw_mlgwsc1.yaml` - Problem configuration
2. `problems/gw_mlgwsc1/eval.py` - Evaluation pipeline
3. `problems/gw_mlgwsc1/eval.sh` - Execution script
4. `problems/gw_mlgwsc1/eval_inj.py` - Injection processing
5. `problems/gw_mlgwsc1/gen_inst.py` - Instance generation
6. `problems/gw_mlgwsc1/plot_auc.py` - AUC visualization
7. `problems/gw_mlgwsc1/gpt_o3-mini-medium-job*.py` - 11 generated algorithm variants
8. `prompts/gw_mlgwsc1/func_desc.txt` - Function description prompt
9. `prompts/gw_mlgwsc1/func_signature.txt` - Function signature template
10. `prompts/gw_mlgwsc1/seed_func.txt` - Seed algorithm implementation
11. `prompts/gw_mlgwsc1/external_knowledge.txt` - Domain-specific knowledge

### Execution Results
- **Runs**: 9 independent executions (jobs 1-10, job-4 skipped; June 12-15, 2025)
- **LLM model**: o3-mini-medium
- **Output directory**: `outputs/gw_mlgwsc1-constructive/`
- **Note**: Removed 857 .npy image files to reduce repository size (likely spectrograms/visualizations)

### Key Parameters
- Max function evaluations: 1000
- Population size: 10
- Initial population: 8
- Timeout: 3600 seconds per job
- MCTS exploration coefficient: c_0 (time-decaying)
- Max tree depth: 10 levels

### Algorithm Structure
MCTS-AHD employs Monte Carlo Tree Search for systematic exploration:
1. **Initialization**: Seed algorithm for dual-detector signal processing
2. **Selection**: UCT-based node selection with exploration-exploitation trade-off
3. **Expansion**: Generate new heuristic variants via LLM
4. **Simulation**: Evaluate algorithm fitness on training data
5. **Backpropagation**: Update Q-values and visit counts up the tree
6. **Population**: Maintain elite algorithms for crossover operations

### Performance
- **Best fitness achieved**: 2494.10 AUC [Mpc] (best-of-9 runs; aligned with the revised manuscript)
- **Comparison with Evo-MCTS**: See the revised manuscript and supplementary comparison tables
- **110.2% improvement** by Evo-MCTS over MCTS-AHD using the PT-4 value 5241.37 Mpc

## Repository Cleanup

### Deleted Files
- **857 .npy files** in `outputs/gw_mlgwsc1-constructive/*/evaluations/`
  - Run 1 (2025-06-12_23-42-53): 79 files deleted
  - Run 2 (2025-06-14_11-12-58): 218 files deleted
  - Run 3 (2025-06-14_11-56-01): 131 files deleted
  - Run 4 (2025-06-15_00-24-16): 207 files deleted
  - Run 5 (2025-06-15_00-54-38): 122 files deleted
  - Run 6 (2025-06-15_13-33-09): 100 files deleted

### Preserved Outputs
- All execution logs (.log files)
- Text-based results and metrics
- JSON configuration files
- Algorithm source code (.py files)
- All essential data for reproducibility

## Reproducibility Notes

### Git History
This repository has been initialized with git to track modifications:
- Upstream remote: https://github.com/zz1358m/MCTS-AHD-master.git
- Initial commit shows complete GW detection adaptation
- Git diff against upstream shows exact changes made
- .gitignore configured to exclude large binary files

### Fair Comparison Protocol
All baseline frameworks (ReEvo, MCTS-AHD) were:
1. Run with identical LLM model (o3-mini-medium)
2. Tested on identical dataset (MLGWSC-1 Set 4)
3. Evaluated using same fitness metric (AUC in Mpc)
4. Executed with comparable computational budgets
5. Modified minimally to support GW detection problem type

### Modifications Summary
- **Minimal invasive changes**: Only added new problem type, no core framework modifications
- **Domain-specific prompts**: Gravitational wave physics encoded in external_knowledge.txt
- **Evaluation infrastructure**: Custom eval.py for dual-detector processing
- **Seed algorithm**: Baseline implementation for coherent signal processing

## Technical Details

### MCTS Tree Structure
- **Root node**: Seed algorithm (initial heuristic)
- **Child nodes**: Generated algorithmic variants
- **Node attributes**: Algorithm description, executable code, Q-value, visit count
- **Selection criterion**: UCT formula with normalized Q-values
- **Expansion strategy**: Multiple operators (PC, SC, PWC, PM) applied per node

### Evaluation Metrics
- **Primary metric**: AUC (Area Under Curve) = ∫ d_L d(log10 FAR)
- **Units**: Megaparsecs (Mpc)
- **Integration range**: log10(FAR) ∈ [log10(4.3), log10(1000)] ≈ [0.633, 3.0]
- **Dataset**: MLGWSC-1 Set 4 (7-day training, 1-day test)
- **Detectors**: H1 (Hanford) and L1 (Livingston) dual-detector coherence

### Computational Environment
- **Execution logs**: 9 analyzed runs spanning June 12-15, 2025
- **Parallel processing**: Multi-core injection processing
- **Time budget**: ~3600 seconds per evaluation
- **Storage optimization**: .npy files removed post-evaluation

## References

**Original Paper:**
```bibtex
@inproceedings{zheng2025mcts,
  title={Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design},
  author={Zheng, Zhi and Zhang, Jie and Zhang, Yueqing and Jin, Yan},
  booktitle={ICML},
  year={2025},
  note={arXiv:2501.08603}
}
```

**This Work:**
See main manuscript for citation details.
