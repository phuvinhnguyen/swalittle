# SWE-Little: RL for Software Engineering

A framework for reinforcement learning with language models on real-world software engineering tasks (SWE-bench). This project provides containerized, safe environments for agents to interact with codebases, and supports flexible configuration and extensibility.

---

## 1. Installing Apptainer

**Apptainer** (formerly Singularity) is required for containerized environments.

### On Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y apptainer
```

### On Arch Linux
```bash
sudo pacman -S apptainer
```

### On CentOS/RHEL
```bash
sudo yum install apptainer
```

### On HPC cluster
```bash
cd swalittle
bash ext/scripts/apptainer.sh
```

For more details, see [Apptainer installation docs](https://apptainer.org/docs/user/latest/quick_start.html).

---

## 2. Creating a New Experiment Config File

All experiments are configured via YAML files. Place your config in the `ext/configs/` directory. You can check some example configs in this folder.

- **environment**: Which environment to use and its parameters
- **model**: Model class and HuggingFace/PEFT/LoRA options
- **algorithm**: Training algorithm to use (GRPO, REINFORCE, ...)
- **trainer**: Training algorithm and hyperparameters

You can add or override fields as needed for your experiment.

---

## 3. Running the Project, Experiments, and Training

After setting up your config file, run training or experiments with:

```bash
python src/trainer.py --config ./ext/configs/my_experiment.yaml
```

- Replace `./ext/configs/my_experiment.yaml` with your config path.
- Training logs, checkpoints, and outputs will be saved as specified in your config.

---

## 4. Creating New Modules (Algorithms, Environments, Models)

The framework is modular. You can add new algorithms, environments, or models by creating new Python classes in the appropriate directory.

### A. New Algorithm (Trainer)
- Place in `src/algorithms/`
- **Required methods:**
  - `compute_loss(self, ...)` : Computes loss for optimization

### B. New Environment
- Place in `src/environments/`
- Inherit from `base.Environment`
- **Required methods:**
  - `reset(self)` : Resets environment, returns initial state
  - `step(self, action)` : Applies action, returns (state, reward, done, info)
  - `close(self)` : Cleans up resources

### C. New Model
- Place in `src/models/`
- Inherit from `base.Model`
- **Required methods:**
  - `act(self, state)` : Returns action given state
  - `forward(self, trajectories)` : Returns loss for each trajectory's transition
  - `save(self, path)` : Saves model weights
  - `load(self, path)` : Loads model weights
