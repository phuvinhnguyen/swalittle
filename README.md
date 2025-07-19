# SWE-Little: Reinforcement Learning for Software Engineering

A reinforcement learning framework for training language models on software engineering tasks using the SWE-bench dataset. This project provides a containerized environment where agents can interact with real codebases, make edits, and learn from their actions.

## 🚀 Quick Start

### Prerequisites

- **Linux** (tested on Arch Linux)
- **Python 3.8+**
- **Apptainer/Singularity** for containerization
- **Git** for repository cloning

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd swalittle
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Apptainer:**
   ```bash
   # On Arch Linux
   sudo pacman -S apptainer
   
   # On Ubuntu/Debian
   sudo apt-get install apptainer
   
   # On CentOS/RHEL
   sudo yum install apptainer
   ```

4. **Setup tools directory:**
   ```bash
   mkdir -p ext/tools
   # Add your custom tools here (see Tools section below)
   ```

## 📁 Project Structure

```
swalittle/
├── configs/                 # Training configuration files
│   └── ppos_llama_tiny.yaml
├── docs/                    # Documentation
│   └── ppo.md
├── env/                     # Container environment files
│   ├── singularity.def
│   └── singularity.sif
├── notebooks/               # Jupyter notebooks for experiments
│   ├── b.ipynb
│   └── c.ipynb
├── src/
│   ├── environments/        # RL environments
│   │   ├── base.py
│   │   ├── ContainerEnv.py  # Main containerized environment
│   │   └── Game7.py
│   ├── models/              # Model implementations
│   │   ├── rllm.py
│   │   └── llm.py          # HuggingFace LLM wrapper
│   ├── tools/               # Custom tools for agents
│   │   └── search/
│   │       ├── bin/
│   │       ├── config.yaml
│   │       └── install.sh
│   ├── trainer/             # Training algorithms
│   │   ├── base.py
│   │   └── grpo.py
│   └── train.py            # Main training script
└── requirements.txt
```

## 🎯 Core Components

### 1. Container Environment (`ContainerEnv`)

The main environment that provides a sandboxed container where agents can:
- Clone real GitHub repositories
- Execute commands safely
- Track file changes with git diff
- Receive rewards based on their actions

**Key Features:**
- Isolated `/home` directory (doesn't affect host system)
- Automatic git diff computation after each step
- Support for custom tools and scoring functions
- Safe command execution with error handling

### 2. Dataset Integration

Supports SWE-bench datasets with automatic:
- Repository cloning and checkout
- Problem statement extraction
- True patch comparison
- Environment setup

### 3. Model Wrapper (`LLM`)

A flexible wrapper for HuggingFace models supporting:
- PEFT/LoRA fine-tuning
- Quantization (4-bit, 8-bit)
- Training and inference modes
- Dynamic model loading

### 4. Training Algorithms

- **GRPO**: Group Relative Policy Optimization
- **REINFORCE**: Vanilla policy gradient
- **Behavioral Cloning**: Imitation learning from demonstrations

## 🛠️ Usage Examples

### Basic Environment Usage

```python
from src.environments.ContainerEnv import ContainerEnv

# Create environment
env = ContainerEnv(
    data_name_or_path="princeton-nlp/SWE-bench_Lite",
    split="test",
    sif_folder="./env",
    base_tools_path="./src/tools",
    tool_list=["search"]
)

# Reset to get a new task
initial_state = env.reset()
print(f"Problem: {initial_state['problem_statement']}")

# Execute commands
message, reward, done, extra = env.step("<execute>ls -la</execute>")
print(f"Output: {extra['output']}")
print(f"Reward: {reward}")

# Clean up
env.close()
```

### Training a Model

```python
from src.train import main
import yaml

# Load configuration
with open("configs/ppos_llama_tiny.yaml", "r") as f:
    config = yaml.safe_load(f)

# Start training
main(config)
```

### Using Custom Tools

1. **Create a tool directory:**
   ```
   src/tools/my_tool/
   ├── bin/
   │   └── my_script
   ├── config.yaml
   └── install.sh
   ```

2. **Install script (`install.sh`):**
   ```bash
   #!/bin/bash
   chmod +x bin/my_script
   ```

3. **Use in environment:**
   ```python
   env = ContainerEnv(
       tool_list=["my_tool"],
       base_tools_path="./src/tools"
   )
   ```

## 🔧 Configuration

### Training Configuration

Create YAML configs in `configs/` directory:

```yaml
# configs/my_experiment.yaml
environment:
  class: "ContainerEnv"
  data_name_or_path: "princeton-nlp/SWE-bench_Lite"
  split: "test"
  sif_folder: "./env"
  base_tools_path: "./src/tools"
  tool_list: ["search"]

model:
  class: "LLM"
  model_name: "microsoft/DialoGPT-medium"
  use_peft: true
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]

trainer:
  class: "GRPO"
  learning_rate: 1e-4
  batch_size: 4
  max_episodes: 1000
  save_every: 100
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # GPU to use
export HF_HOME=/path/to/cache  # HuggingFace cache
export APPTAINER_CACHEDIR=/path/to/cache  # Apptainer cache
```

## 🧪 Running Experiments

### 1. Quick Test

```bash
# Test environment setup
python -c "
from src.environments.ContainerEnv import ContainerEnv
with ContainerEnv() as env:
    env.reset()
    msg, reward, done, extra = env.step('<execute>pwd</execute>')
    print(f'Success! Output: {extra[\"output\"]}')
"
```

### 2. Training Run

```bash
# Train with specific config
python src/train.py --config configs/ppos_llama_tiny.yaml

# Train with custom parameters
python src/train.py \
    --config configs/ppos_llama_tiny.yaml \
    --model.model_name "microsoft/DialoGPT-large" \
    --trainer.learning_rate 5e-5
```

### 3. Jupyter Notebooks

```bash
# Start Jupyter
jupyter lab notebooks/

# Or run specific notebook
jupyter nbconvert --to notebook --execute notebooks/b.ipynb
```

## 📊 Monitoring and Logging

### Training Logs

Training automatically logs:
- Episode rewards and losses
- Model checkpoints
- Environment statistics
- Git diff summaries

### Visualization

```python
import matplotlib.pyplot as plt
import json

# Load training logs
with open("logs/training_log.json", "r") as f:
    logs = json.load(f)

# Plot rewards
plt.plot(logs["episode_rewards"])
plt.title("Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
```

## 🔍 Debugging

### Common Issues

1. **Apptainer not found:**
   ```bash
   # Install apptainer
   sudo pacman -S apptainer  # Arch
   sudo apt-get install apptainer  # Ubuntu
   ```

2. **Permission denied:**
   ```bash
   # Fix tool permissions
   chmod +x src/tools/*/bin/*
   ```

3. **Out of memory:**
   ```bash
   # Reduce batch size in config
   trainer:
     batch_size: 2  # Reduce from 4
   ```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
from src.environments.ContainerEnv import ContainerEnv
env = ContainerEnv()
env.reset()
# ... debug commands
```

## 🤝 Contributing

### Adding New Components

1. **New Environment:**
   - Inherit from `base.Environment`
   - Implement `reset()`, `step()`, `close()`

2. **New Model:**
   - Inherit from `base.Model`
   - Implement `act()`, `train()`, `save()`

3. **New Trainer:**
   - Inherit from `base.Trainer`
   - Implement `train_episode()`, `compute_loss()`

### Code Style

- Use type hints
- Follow PEP 8
- Add docstrings
- Include tests

## 📚 References

- [SWE-bench Dataset](https://github.com/princeton-nlp/SWE-bench)
- [Apptainer Documentation](https://apptainer.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SWE-bench team for the dataset
- HuggingFace for the transformers library
- Apptainer community for containerization tools

---

**Happy coding! 🚀**
