# Results Directory

This directory contains experimental results and data from the Evo-MCTS research paper.

## Paper Data

### `paper_data/mcts_tree_nodes_pt5_algorithm.jsonl`

Complete MCTS tree structure data for the PT5 algorithm (node 486, fitness=5041.4) as referenced in **Figure 5** of the paper "Automated Algorithmic Discovery for Gravitational-Wave Detection Guided by LLM-Informed Evolutionary Monte Carlo Tree Search".

**File Format**: JSON Lines (.jsonl) - each line contains one tree node
**Total Nodes**: 38 algorithm nodes
**Node Range**: eval_times 1-486 (complete execution sequence)

#### Data Schema

Each JSON object contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `eval_times` | int | LLM execution sequence number (1-486) |
| `depth` | int | MCTS tree depth level (1-10) |
| `operator` | string | MCTS expansion type (PC, SC, PWC, PM) |
| `thinking` | string | DeepSeek reasoning results |
| `reflection` | string | DeepSeek reflection analysis |
| `code` | string | Generated algorithm implementation |
| `fitness` | float | Algorithm performance score (AUC) |
| `algorithm` | string | Post-thought algorithmic insights |

#### Usage Example

```python
import json

# Read the MCTS tree data
nodes = []
with open('paper_data/mcts_tree_nodes_pt5_algorithm.jsonl', 'r') as f:
    for line in f:
        nodes.append(json.loads(line))

# Find the best performing node
best_node = max(nodes, key=lambda x: x['fitness'])
print(f"Best node: {best_node['eval_times']}, fitness: {best_node['fitness']}")

# Analyze depth distribution
depth_counts = {}
for node in nodes:
    depth = node['depth']
    depth_counts[depth] = depth_counts.get(depth, 0) + 1
print(f"Depth distribution: {depth_counts}")
```

#### Research Applications

This dataset enables:
- **Reproducibility**: Full reconstruction of the MCTS tree exploration process
- **Algorithm Analysis**: Understanding how different operators contribute to performance
- **Evolution Tracking**: Following the path from initial exploration to optimal solution
- **Method Validation**: Verifying the systematic nature of the search process

For more details, see the main [README.md](../README.md) and the original paper on [arXiv:2508.03661](https://arxiv.org/abs/2508.03661).
