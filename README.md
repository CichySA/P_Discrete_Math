# p-discrete-math

Discrete mathematics project — flow networks, max-flow optimisation, and fairness analysis.

## Quick start (uv + venv)

```bash
# 1. Create virtual environment
uv venv

# 2. Activate it
#    Windows (PowerShell)
.venv\Scripts\Activate.ps1
#    macOS / Linux
source .venv/bin/activate

# 3. Sync dependencies
uv sync
```

All dependencies are declared in `pyproject.toml` and will be installed with sync.

## Repository guide

| File | Description |
|------|-------------|
| `flow_network.py` | Visualizations and helpers used by Szymon's notebook. |
| `Main_szymon.ipynb` | Theoretical exploration of max-flow / min-cut, LP formulations (cvxpy), α-fairness sweeps, Pareto frontiers, Lagrangian duality, and KKT conditions. **NOTE: some cells are dependant on others and must be executed in order.** |
| `main_Maxime.ipynb` | Applied graph optimisation for epidemic-resource allocation in Wrocław — capacity sensitivity, most-vital-edge analysis, and budget-constrained network design (greedy vs. MILP with PuLP). |
