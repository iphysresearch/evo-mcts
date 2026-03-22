# Gravitational Wave Detection Adaptations

## Original Framework
- Repository: https://github.com/ai4co/reevo
- Paper: ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution (NeurIPS 2024)
- Authors: Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, Guojie Song
- License: MIT (2023)

## Modifications for MLGWSC-1 Benchmark

### New Problem Type: gw_mlgwsc1
- **Location**: `problems/gw_mlgwsc1/`
- **Evaluation**: Dual-detector gravitational wave signal processing
- **Metric**: AUC (Area Under Curve) of sensitive distance vs log10(FAR)
- **Dataset**: MLGWSC-1 Set 4 (7-day training, 1-day test)

### Files Added
1. `cfg/problem/gw_mlgwsc1.yaml` - Problem configuration
2. `problems/gw_mlgwsc1/eval.py` - Evaluation pipeline
3. `problems/gw_mlgwsc1/eval.sh` - Execution script
4. `problems/gw_mlgwsc1/eval_inj.py` - Injection processing
5. `problems/gw_mlgwsc1/gen_inst.py` - Instance generation
6. `problems/gw_mlgwsc1/plot_auc.py` - AUC visualization
7. `problems/gw_mlgwsc1/gpt_gw_mlgwsc1_*.py` - 11 generated algorithm variants
8. `prompts/gw_mlgwsc1/func_desc.txt` - Function description prompt
9. `prompts/gw_mlgwsc1/func_signature.txt` - Function signature template
10. `prompts/gw_mlgwsc1/seed_func.txt` - Seed algorithm implementation
11. `prompts/gw_mlgwsc1/external_knowledge.txt` - Domain-specific knowledge

### Execution Results
- **Runs**: 6 independent executions (June 15-16, 2025)
- **LLM model**: o3-mini-medium
- **Output directory**: `outputs/gw_mlgwsc1-constructive/`
- **Total output size**: 189 MB (text files, logs, pickles)

### Key Parameters
- Max function evaluations: 500
- Population size: 10
- Mutation rate: 0.5
- Temperature: 1.0
- Reflection mode: Short-term (ST)

### Algorithm Structure
ReEvo employs a hyper-heuristic approach that evolves heuristic functions through:
1. **Initialization**: Seed algorithm for dual-detector signal processing
2. **Mutation**: Random modifications to existing algorithms
3. **Crossover**: Combining successful algorithmic components
4. **Reflection**: LLM-based analysis of performance feedback
5. **Selection**: Elite population maintenance based on AUC fitness

### Performance
- **Best fitness achieved**: 2899.40 AUC [Mpc]
- **Discovery point**: Iteration 18, code variant 9 (problem_iter18_code9.py)
- **Execution date**: June 17, 2025 06:02:36
- **Comparison with Evo-MCTS**: See main manuscript Table 2
- **Performance ranking**:
  - Evo-MCTS: 5241.37 Mpc (+80.7% vs ReEvo)
  - MCTS-AHD: 3293.85 Mpc (+13.6% vs ReEvo)
  - ReEvo: 2899.40 Mpc (baseline LLM framework)

## Reproducibility Notes

### Git History
This repository has been initialized with git to track modifications:
- Upstream remote: https://github.com/ai4co/reevo.git
- Initial commit shows complete GW detection adaptation
- Git diff against upstream shows exact changes made

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

## References

**Original Paper:**
```bibtex
@inproceedings{ye2024reevo,
  title={ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution},
  author={Ye, Haoran and Wang, Jiarui and Cao, Zhiguang and Berto, Federico and Hua, Chuanbo and Kim, Haeyeon and Park, Jinkyoo and Song, Guojie},
  booktitle={NeurIPS},
  year={2024}
}
```

**This Work:**
See main manuscript for citation details.
