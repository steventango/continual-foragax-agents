# AI Coding Assistant Instructions for continual-foragax-agents

## Project Overview
This is a reinforcement learning research codebase focused on continual learning in foraging environments. It uses JAX for neural network computations and PyExpUtils for experiment management.

## Architecture & Key Components

### Core Entry Points
- **`src/main.py`**: Single experiment runs (episodic tasks)
- **`src/continuing_main.py`**: Batch/continual learning experiments with JAX vmap
- **`src/optuna_tuning.py`**: Hyperparameter optimization

### Experiment Structure
Experiments are defined as JSON files in `experiments/` with this pattern:
```json
{
    "agent": "DQN",
    "problem": "ForagaxTwoBiome-v1",
    "total_steps": 1000000,
    "metaParameters": {
        "batch": 32,
        "buffer_size": 10000,
        "optimizer": {"name": "ADAM", "alpha": 0.001}
    }
}
```

### Directory Structure
- **`src/algorithms/`**: RL agents (DQN variants, EQRC, etc.)
- **`src/environments/`**: Environment wrappers (Foragax, Gym, Minatar)
- **`src/problems/`**: Problem definitions combining agent + environment + representation
- **`experiments/`**: Experiment configs and analysis scripts
- **`results/`**: SQLite databases + numpy arrays from runs

## Critical Workflows

### Local Development
You must use the provided virtual environment `.venv` and activate it before running any scripts.

```bash
# Test single experiment
source .venv/bin/activate
python src/main.py -e experiments/example/MountainCar/EQRC.json -i 0

# Run continual learning experiment
source .venv/bin/activate
python src/continuing_main.py -e experiments/foragax/ForagaxTwoBiome-v1/DQN.json -i 0

# Analyze results
source .venv/bin/activate
python experiments/example/learning_curve.py
```

### Cluster Execution (Compute Canada)
```bash
# Schedule jobs
source .venv/bin/activate
python scripts/slurm.py --clusters clusters/cedar.json --runs 5 -e experiments/foragax/*.json

# Check missing results and reschedule
python scripts/slurm.py --clusters clusters/cedar.json --runs 5 -e experiments/foragax/*.json
```

### Environment Setup
```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install -e .[dev]
pre-commit install
```

## Project-Specific Patterns

### JAX-Based Agents
- Use `@partial(jax.jit, static_argnums=0)` for performance
- Implement `getAgent()` method returning JAX-compatible agent
- Handle batch processing with `jax.vmap` in continuing_main.py

### Checkpointing System
- Automatic checkpointing every 1M steps in long runs
- State saved as compressed pickle + numpy arrays
- Resume capability for interrupted experiments

### Result Storage
- SQLite databases for metadata (returns, episodes, steps)
- Numpy `.npz` files for high-frequency metrics (losses, TD errors)
- Path structure: `results/{experiment}/{agent}/{params}/`

### Foragax Environments
- Custom foraging environments for continual learning research
- Position tracking: `datas["pos"]` for trajectory analysis
- Privileged information access via `privileged=True` parameter

### Instrumentation
```python
collector = Collector(config={
    "return": Identity(),
    "loss": Pipe(MovingAverage(0.99), Subsample(100))
})
```

## Development Guidelines

### Code Organization
- One agent per file in `src/algorithms/`
- Environment wrappers in `src/environments/`
- Analysis utilities in `src/analysis/`
- Shared utilities in `src/utils/`

### Dependencies
- JAX ecosystem: `jax`, `haiku`, `optax`, `chex`
- Research libraries: `PyExpUtils`, `PyFixedReps`, `RlGlue`
- Custom: `foragerenv`, `continual-foragax`

### Python Version
- **Python 3.11+ required** (uses modern typing features)
- Use `.venv/bin/python` interpreter (not system python)

### Commit Style
Follow conventional commits: `feat:`, `fix:`, `refactor:`, etc.
Automated versioning via commitizen.

## Common Pitfalls

### JAX Gotchas
- Use `jax.numpy` instead of `numpy` for agent computations
- Be careful with in-place operations (use `.at[].set()`)
- Static arguments need `static_argnums` in jit decorators

### Experiment Configuration
- `total_steps` vs `episode_cutoff` distinction
- Batch size parameters must be consistent across runs for vmap
- Buffer sizes affect memory usage significantly

### Result Analysis
- Results load from SQLite + numpy files
- Use `ResultCollection` and `groupby_directory()` for analysis
- Missing results are automatically detected and rescheduled

## Key Files to Reference
- `src/continuing_main.py`: Batch execution with JAX vmap
- `src/environments/Foragax.py`: Foragax environment wrapper
- `experiments/example/learning_curve.py`: Result plotting pattern
- `scripts/slurm.py`: Cluster job scheduling
- `pyproject.toml`: Dependencies and project config
