# Evo-MCTS: Evolutionary Monte Carlo Tree Search for Gravitational Wave Signal Detection

[![arXiv](https://img.shields.io/badge/arXiv-2508.03661-b31b1b.svg)](https://arxiv.org/abs/2508.03661)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![GitHub stars](https://img.shields.io/github/stars/iphysresearch/evo-mcts?style=social)](https://github.com/iphysresearch/evo-mcts/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/iphysresearch/evo-mcts?style=social)](https://github.com/iphysresearch/evo-mcts/network/members)

**Official implementation of "Automated Algorithmic Discovery for Gravitational-Wave Detection Guided by LLM-Informed Evolutionary Monte Carlo Tree Search"**

This repository contains the open-source, reproducible code for the research paper currently available as [arXiv:2508.03661](https://arxiv.org/abs/2508.03661). Our Evo-MCTS framework demonstrates substantial performance improvements: a **20.2% improvement** over the strongest reported challenge baseline on the [MLGWSC-1 benchmark dataset](https://github.com/gwastro/ml-mock-data-challenge-1) and a **59.1% improvement** in architectural ablation over the ReEvo-only mean. For clarity, these percentages refer to different comparisons: **20.2%** is PT-4 versus Sage on MLGWSC-1 Set 4, whereas **59.1%** is the integrated Evo-MCTS ablation mean versus the ReEvo component mean.

## 📋 Table of Contents

- [📄 Paper Information](#-paper-information)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Project Structure](#️-project-structure)
- [⚙️ Configuration](#️-configuration)
- [🔬 Algorithm Overview](#-algorithm-overview)
- [📊 Results](#-results)
- [🛠️ Development Guide](#️-development-guide)
- [📈 Usage Examples](#-usage-examples)
- [🔄 Portability and Deployment](#-portability-and-deployment)
- [📝 Important Notes](#-important-notes)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Contact](#-contact)

## 📄 Paper Information

**Title:** Automated Algorithmic Discovery for Gravitational-Wave Detection Guided by LLM-Informed Evolutionary Monte Carlo Tree Search

**Authors:** He Wang<sup>1,2</sup>, Liang Zeng<sup>3</sup>

<sub>1. International Centre for Theoretical Physics Asia-Pacific, University of Chinese Academy of Sciences, 100190, Beijing, China</sub>  
<sub>2. Taiji Laboratory for Gravitational Wave Universe, University of Chinese Academy of Sciences, 100049, Beijing, China</sub>  
<sub>3. Tsinghua University, 100084, Beijing, China </sub>

**Abstract:** From fundamental physics to gravitational-wave astronomy, computational scientific discovery increasingly relies on sophisticated algorithms to analyze complex datasets, yet reliable identification of gravitational-wave signals with unknown source parameters buried in dynamic detector noise remains a formidable challenge. While existing algorithmic approaches have achieved partial success, their core limitations arise from restrictive prior assumptions: traditional methods suffer from reliance on predefined theoretical priors, while neural network approaches introduce hidden biases and lack interpretability. We propose **Evolutionary Monte Carlo Tree Search (Evo-MCTS)**, the first integration of large language model (LLM) guidance with domain-aware physical constraints to generate interpretable solutions for automated gravitational wave detection. This framework systematically explores algorithmic solution spaces through tree-structured search enhanced by evolutionary optimization. Experimental validation demonstrates substantial performance improvements, achieving a **20.2%** improvement over the strongest reported MLGWSC-1 challenge baseline and a **59.1%** improvement in architectural ablation over the ReEvo-only mean. More fundamentally, our framework establishes a transferable methodology for automated algorithmic discovery across computational science domains through systematic exploration of novel algorithmic combinations.

**Citation:**
```bibtex
@article{wang2025automated,
      title={Automated Algorithmic Discovery for Gravitational-Wave Detection Guided by LLM-Informed Evolutionary Monte Carlo Tree Search}, 
      author={He Wang and Liang Zeng},
      year={2025},
      eprint={2508.03661},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.03661}, 
}
```

## 🚀 Quick Start

### Environment Setup

Create a reproducible conda environment with the following commands:

```bash
# 1. Create conda environment
conda create -n env_mcts -c conda-forge python=3.12

# 2. Activate environment
conda activate env_mcts

# 3. Install dependencies
pip install hydra-core tqdm numpy joblib openai h5py gwpy pandas lalsuite
pip install "scipy==1.12.0"
```

### Running the Code

#### Using run_template.py (Recommended)

The `run_template.py` script provides a programmatic and flexible way to execute Evo-MCTS with customizable parameters:

```bash
# Basic execution with default settings
python run_template.py

# Custom model and API configuration
python run_template.py --model gpt-4 --api-key your-openai-api-key --temperature 0.8

# Using environment variables
export MODEL=gpt-4
export API_KEY=your-openai-api-key
export TEMPERATURE=0.8
python run_template.py

# Dry run to validate configuration
python run_template.py --dry-run --model gpt-4 --api-key your-key
```

#### Direct Python Execution
```bash
python main.py \
  problem=gw_mlgwsc1 \
  llm_client.model=gpt-4 \
  llm_client.api_key=your-api-key \
  debug_mode=False
```

## 🏗️ Project Structure

```
Evo-MCTS/
├── main.py                 # Main program entry point
├── run_template.py         # Template script for programmatic execution
├── ahd_adapter.py          # Algorithm adapter
├── problem_adapter.py      # Problem adapter
├── .env                    # Environment variables (see .env.template)
├── source/                 # Core algorithm implementation
│   ├── evo_mcts.py        # Evolutionary Monte Carlo Tree Search
│   ├── evolution.py       # Evolution operations
│   ├── mcts.py           # Monte Carlo Tree Search
│   ├── getParas.py       # Parameter management
│   └── interface_LLM.py  # LLM interface
├── utils/                 # Utility functions
│   └── llm_client/       # LLM client implementations
├── cfg/                  # Hydra configuration files
│   ├── config.yaml       # Main configuration
│   ├── problem/          # Problem-specific configs
│   └── llm_client/       # LLM client configs
├── problems/             # Problem definitions
│   └── gw_mlgwsc1/      # Gravitational wave detection problem
├── prompts/             # LLM prompt templates
└── results/             # Research results and paper data
    └── paper_data/      # Published paper experimental data
        └── mcts_tree_nodes_pt5_algorithm.jsonl  # Historical filename; archived PT-4 node-486 tree data used in the paper
```

## ⚙️ Configuration

### Environment Variables

The project supports flexible configuration through environment variables. For easy deployment and portability, we provide template files:

#### Configuration Templates
- **`.env.template`**: Template for environment variables (replace sensitive information)
- **`run_template.py`**: Programmatic execution script with flexible configuration options

```bash
# LLM Configuration
MODEL=gpt-4                              # Model to use
API_KEY=your-api-key-here               # Your API key
BASE_URL=https://api.openai.com/v1      # API endpoint
TEMPERATURE=1.0                         # Generation temperature

# Path Configuration
ML_CHALLENGE_PATH=/path/to/ml-mock-data-challenge-1  # MLGWSC-1 repository path
DATA_DIR=/path/to/generated/datasets                 # Generated HDF5 datasets directory

# Performance Configuration
NUMEXPR_MAX_THREADS=96                  # Number of computation threads
CPU_USAGE_PERCENT=50                    # CPU usage percentage

# DeepSeek Configuration (Alternative LLM)
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### Data Configuration

#### ML_CHALLENGE_PATH
This should point to the cloned [MLGWSC-1 repository](https://github.com/gwastro/ml-mock-data-challenge-1):

```bash
# Clone the MLGWSC-1 repository
git clone https://github.com/gwastro/ml-mock-data-challenge-1.git
export ML_CHALLENGE_PATH=/path/to/ml-mock-data-challenge-1
```

#### DATA_DIR  
This directory contains the generated HDF5 datasets for gravitational wave detection. The dataset files follow the naming pattern defined in `problems/gw_mlgwsc1/gen_inst.py`:

**Dataset File Naming Convention:**
```
foreground_set{SET_NUMBER}_{DURATION}dur_seed{RANDOM_SEED}_start{START_TIME}.hdf
background_set{SET_NUMBER}_{DURATION}dur_seed{RANDOM_SEED}_start{START_TIME}.hdf
injections_set{SET_NUMBER}_{DURATION}dur_seed{RANDOM_SEED}_start{START_TIME}.hdf
```

**Example dataset files:**
- `foreground_set4_604800dur_seed40_start0.hdf`
- `background_set4_604800dur_seed40_start0.hdf`  
- `injections_set4_604800dur_seed40_start0.hdf`

**Dataset Generation:**
To generate datasets, you can use the script from this [gist](https://gist.github.com/iphysresearch/1b21358869a1fe0ae8712564a0bfa96e) or refer to the parameters in `gen_inst.py`:
- `SET_NUMBER = 4` (Dataset set number)
- `DURATION = 604800` (1 week in seconds)
- `RANDOM_SEED = 40` (Reproducibility seed)
- `START_TIME = 0` (Start offset)

### Hydra Configuration

The project uses Hydra for configuration management:
- `cfg/config.yaml` - Main configuration
- `cfg/problem/gw_mlgwsc1.yaml` - Problem-specific configuration
- `cfg/llm_client/openai.yaml` - LLM client configuration

## 🔬 Algorithm Overview

### Evo-MCTS Framework

Our framework represents the first integration of LLM guidance with evolutionary search for gravitational wave detection:

- **LLM Guidance**: Large language model integration with domain-aware physical constraints
- **Evolutionary Algorithm**: Global search with population diversity for systematic exploration
- **Monte Carlo Tree Search**: Tree-structured search with intelligent exploration
- **Interpretable Solutions**: Generates human-interpretable algorithmic pathways
- **Domain Awareness**: Incorporates physical constraints specific to gravitational wave detection

### Key Features

1. **First LLM-Guided Framework**: Novel integration of large language model guidance with domain-aware physical constraints
2. **Systematic Algorithm Space Exploration**: Tree-structured search enhanced by evolutionary optimization
3. **Interpretable Solutions**: Generates human-interpretable algorithmic pathways for automated gravitational wave detection
4. **Superior Performance**: 20.2% improvement over the strongest reported MLGWSC-1 challenge baseline and 59.1% improvement in architectural ablation over the ReEvo-only mean
5. **Novel Algorithm Discovery**: Discovers previously unknown algorithmic combinations
6. **Transferable Methodology**: Establishes a transferable approach for automated algorithmic discovery across computational science domains

### Gravitational Wave Detection Challenge

- **Challenge**: Detect gravitational-wave signals with unknown source parameters buried in dynamic detector noise
- **Input**: H1 and L1 dual-channel gravitational wave data with complex noise characteristics
- **Objective**: Automated construction of interpretable signal detection pipelines
- **Output**: Catalog of candidate gravitational wave signals with enhanced detection accuracy
- **Benchmark**: MLGWSC-1 dataset validation
- **Innovation**: First framework to overcome restrictive assumptions of traditional methods while maintaining interpretability

## 📊 Results

Our Evo-MCTS framework achieves substantial performance improvements:

### Performance Benchmarks
- **20.2% improvement** over Sage on the [MLGWSC-1 benchmark](https://github.com/gwastro/ml-mock-data-challenge-1) using the PT-4 result (5241.37 vs 4359.27 Mpc)
- **59.1% improvement** for the integrated Evo-MCTS architecture over the ReEvo-only ablation mean (2670.37 vs 1677.73 Mpc)
- Consistent performance across high-performing algorithm variants
- Superior handling of unknown source parameters in dynamic detector noise

### MCTS Tree Analysis and Reproducible Data

The complete MCTS tree structure associated with the archived PT-4 breakthrough (node 486, fitness = 5241.37 Mpc) is available in this repository. The file retains an earlier internal filename for compatibility, but its contents correspond to the PT-4 tree discussed in the manuscript. The tree data contains **38 algorithm nodes** with comprehensive execution details:

**Dataset**: [`results/paper_data/mcts_tree_nodes_pt5_algorithm.jsonl`](results/paper_data/mcts_tree_nodes_pt5_algorithm.jsonl) (historical filename; PT-4 tree data)

**Data Schema** (corresponding to Figure 5 in the paper):
- `eval_times`: LLM execution sequence number (1-486)
- `depth`: MCTS tree depth level (1-10)
- `operator`: MCTS expansion type (PC/SC/PWC/PM)
- `thinking`: DeepSeek reasoning results
- `reflection`: DeepSeek reflection analysis  
- `code`: Generated algorithm implementation
- `fitness`: Algorithm performance score (AUC)
- `algorithm`: Post-thought algorithmic insights

**Key Insights from Tree Analysis**:
- **Node 486**: Archived PT-4 breakthrough node with fitness score 5241.37 Mpc
- **Depth Distribution**: Nodes span 10 levels (1-10) showing systematic exploration
- **Operator Analysis**: Different MCTS operators (PC, SC, PWC, PM) contribute to diverse algorithmic variants
- **Evolution Trajectory**: Complete path from initial random exploration to optimized solution

This dataset enables full reproducibility of the MCTS tree construction process and provides detailed insights into the algorithmic discovery mechanism described in the paper.

### Execution Logs and Baseline Comparisons

Complete execution logs for reproducibility are available in [`execution_logs/`](execution_logs/):

```
execution_logs/
├── README.md                          # Documentation and verification guide
├── evo-mcts/                          # 5 production runs
│   ├── run1_april_2025/
│   │   ├── merged_log.log             # Primary discovery run (638 evals)
│   │   └── breakthrough_nodes/        # 6 key algorithmic breakthrough JSONs
│   ├── run2-5/                        # Additional production runs
├── baselines/
│   ├── mcts-ahd/                      # 9 job logs + GW adaptations
│   └── reevo/                         # 6 job logs + GW adaptations
├── ablation/                          # Domain knowledge ablation data
├── analysis/                          # Budget sensitivity, statistics
└── algorithms/                        # PT-4 implementation (pipeline_v2.py)
```

**Key Metrics (from execution logs):**

| Method | Runs | LLM Calls | Best (Mpc) | Mean ± Std (Mpc) |
|--------|------|-----------|------------|------------------|
| **Evo-MCTS** | 5 | 441.6 ± 186.7 | **5241.37** | 2670.37 ± 1879.93 |
| MCTS-AHD | 9 | 488.2 ± 70.7 | 2494.10 | 1235.82 ± 485.14 |
| ReEvo | 5 | 596.0 ± 76.0 | 2899.40 | 1624.40 ± 766.79 |

`LLM Calls` reports the mean ± standard deviation across independent production runs. The archived fiducial Evo-MCTS run discussed in the paper logged 638 evaluation attempts and reached PT-4 at eval 486.

**Performance Improvements:**
- **+110.2%** over MCTS-AHD (5241.37 vs 2494.10 Mpc)
- **+80.8%** over ReEvo (5241.37 vs 2899.40 Mpc)

All baseline frameworks were executed under identical conditions (same LLM model, dataset, hardware, and evaluation protocol). See [`execution_logs/README.md`](execution_logs/README.md) for verification instructions.

### Scientific Contributions
- **First LLM-guided approach** for gravitational wave detection with domain-aware physical constraints
- Discovery of novel algorithmic combinations previously unexplored
- Generation of human-interpretable algorithmic pathways
- Establishment of transferable methodology for computational science domains

## 🛠️ Development Guide

### Adding New Problems

1. Create problem directory under `problems/`
2. Implement evaluation script `eval.py`
3. Add configuration file in `cfg/problem/`

### Integrating New LLMs

1. Implement client in `utils/llm_client/`
2. Add configuration in `cfg/llm_client/`
3. Update client initialization in `utils.py`

### Customizing Evolution Parameters

Modify parameters in the configuration files or through environment variables:
- Population size: `pop_size`
- Maximum function evaluations: `max_fe`
- Timeout settings: `timeout`
- Debug mode: `debug_mode`

## 📈 Usage Examples

### Using run_template.py

```bash
# Basic execution with environment variables
export MODEL=gpt-4
export API_KEY=your-openai-api-key
python run_template.py

# Direct command line arguments
python run_template.py --model gpt-4 --api-key your-key --temperature 0.8

# Advanced configuration with multiple parameters
python run_template.py \
  --model gpt-4 \
  --api-key your-key \
  --temperature 0.5 \
  --cpu-usage 75 \
  --numexpr-threads 64

# Using DeepSeek model
python run_template.py \
  --deepseek-model deepseek-chat \
  --deepseek-api-key your-deepseek-key \
  --deepseek-base-url https://api.deepseek.com/v1

# Validate configuration without running
python run_template.py --dry-run --model gpt-4 --api-key your-key

# Load custom environment file
python run_template.py --env-file custom.env
```

### Direct Python Execution

```bash
# Set custom timeout
python main.py problem=gw_mlgwsc1 timeout=3600 llm_client.api_key=your-key

# Enable debug mode
python main.py problem=gw_mlgwsc1 debug_mode=True llm_client.api_key=your-key
```

## 🔄 Portability and Deployment

The `run_template.py` script is designed for easy deployment across different environments:

### Configuration Management

1. **Environment File**: Create `.env` from template
   ```bash
   cp .env.template .env
   # Edit .env with your specific configuration
   ```

2. **Command Line Arguments**: Override any setting directly
   ```bash
   python run_template.py --model gpt-4 --api-key your-key
   ```

3. **Environment Variables**: Set system-wide configuration
   ```bash
   export MODEL=gpt-4
   export API_KEY=your-key
   python run_template.py
   ```

### Configuration Precedence

The script follows this priority order:
1. **Command line arguments** (highest priority)
2. **Environment variables**
3. **Default values** (lowest priority)

This approach ensures:
- **Security**: Sensitive information can be kept in environment files
- **Portability**: Easy adaptation to different computing environments  
- **Flexibility**: Multiple ways to configure the same parameter
- **Reproducibility**: Consistent results across different setups

## 📝 Important Notes

- **API Keys**: Ensure API keys are correctly configured before running
- **Data Paths**: 
  - Set `ML_CHALLENGE_PATH` to point to the cloned [MLGWSC-1 repository](https://github.com/gwastro/ml-mock-data-challenge-1)
  - Set `DATA_DIR` to the directory containing generated HDF5 datasets
  - Verify both paths exist and are accessible before execution
- **Dataset Requirements**: Generate or obtain the required HDF5 datasets using the naming convention in `gen_inst.py`
- **Performance**: Adjust performance parameters based on your hardware capabilities
- **Resource Monitoring**: Monitor computational resource usage during execution
- **Reproducibility**: Use the same random seeds and model versions for consistent results

## 🤝 Contributing

We welcome contributions to improve the framework! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We gratefully acknowledge:
- The organizers of the MLGWSC-1 challenge, particularly the Albert Einstein Institute team that generated and curated the benchmark dataset used in this work
- The broader gravitational-wave research community for open data, software, and benchmarking infrastructure
- The open-source scientific computing community

## 📞 Contact

For questions about the paper or code:
- **Paper**: [arXiv:2508.03661](https://arxiv.org/abs/2508.03661)
- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Email**: Contact authors through the arXiv paper

---

**🌊 Start your gravitational wave signal detection research journey with Evo-MCTS!**