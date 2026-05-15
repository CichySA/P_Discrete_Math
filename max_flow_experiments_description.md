# Max Flow Network Experiments — Extension Plan

## Overview

This plan extends the existing discrete mathematics project with four
mathematically rich experiments centered on **max flow networks**, the
**max-flow/min-cut duality**, **fairness-constrained flow optimization**,
and **Lagrangian duality** with economic interpretations.

The existing project (`flow_network.py` + `Main.ipynb`) already demonstrates:
- Basic max flow computation via NetworkX.
- A convex (log-utility) flow optimization via CVXPY, hinting at fairness.
- 3D visualization of the objective surface.

The extensions below deepen each of these themes into standalone,
self-contained educational examples.

---

## Files Created

### 1. `example_max_flow_min_cut.py`
**Concept:** Max Flow / Min s-t Cut Duality (Ford-Fulkerson theorem)

**What it demonstrates:**
- Algorithmic extraction of the **minimum s-t cut** via residual graph reachability
  after computing max flow (Edmonds-Karp under the hood).
- Explicit enumeration of **bottleneck edges** (S → T edges at full capacity).
- **LP formulation** of both the primal (max flow) and dual (min cut) using CVXPY,
  verifying strong duality numerically.
- Visualization of the cut partition: S-side in green, T-side in red, cut edges
  highlighted.

**Key methods/constructs:**
- `nx.maximum_flow` + residual graph BFS for the min-cut partition.
- `nx.incidence_matrix` for building constraint matrices.
- CVXPY `cp.Variable` with linear objective and equality constraints.
- The dual LP exposes the integrality property: optimal π ∈ {0,1} and d ∈ {0,1}.

**Why educational:** Shows the duality from two angles — algorithmic (constructive)
and algebraic (LP). The residual graph BFS is the classic proof of the max-flow
min-cut theorem.

---

### 2. `example_fairness_flow.py`
**Concept:** α-Fairness Objectives in Flow Optimization

**What it demonstrates:**
- The **α-fairness family** of utility functions:
  - α = 0 → utilitarian (standard max flow, linear).
  - α = 1 → proportional fairness / Nash bargaining (sum of logs).
  - α = 2 → harmonic mean fairness (minimizes sum of 1/f).
  - α → ∞ → max-min / Rawlsian fairness.
- Side-by-side comparison of edge flow distributions across α values.
- **Lexicographic max-min fairness**: iteratively maximize the smallest flow,
  fix it, then maximize the next-smallest.
- Bar chart comparing per-edge flows under different fairness regimes.

**Key methods/constructs:**
- CVXPY `cp.log`, `cp.inv_pos`, `cp.power` for different α.
- Iterative max-min via successively fixing lower bounds.
- `nx.incidence_matrix` for flow conservation constraints.

**Why educational:** The parametric α family unifies classic fairness notions.
The lexicographic algorithm is the standard approach for max-min fair
bandwidth allocation (as in ATM/MPLS networks). This shows that fairness
is not a single concept but a spectrum.

---

### 3. `example_fairness_dual.py`
**Concept:** Lagrangian Dual of Fairness Flow + KKT Conditions

**What it demonstrates:**
- **Analytical derivation** of the Lagrangian dual for log-utility flow:
  $$L(f,\lambda,\mu) = \sum \log(f_e) - \sum \lambda_e(f_e - c_e) - \mu^T A_{\text{cons}} f$$
- Stationarity condition: $f_e^* = \frac{1}{\lambda_e + (A_{\text{cons}}^T \mu)_e}$
- Numerical solution of the dual via CVXPY, verifying recovery of primal flows.
- **Complementary slackness** verification: λ_e · (c_e - f_e) ≈ 0.
- **Economic interpretation**: λ_e as congestion tolls, μ_u as node potentials.
- **Sensitivity analysis**: how dual variables change as α varies from 0 to ∞.
- Comparison with the standard max-flow dual (which has integral 0-1 solutions).

**Key methods/constructs:**
- Lagrangian mechanics: construct dual function g(λ, μ) = max_f L(f, λ, μ).
- KKT multiplier extraction from convex programs.
- `cp.log`, `cp.Minimize` for the dual objective.

**Why educational:** The dual of fair flow is a rich, continuous optimization
problem — unlike the 0-1 min-cut dual of standard max flow. This reveals
the economic meaning of KKT multipliers. The shadow price interpretation is
directly applicable to congestion pricing in networks.

---

### 4. `example_price_of_fairness.py`
**Concept:** Quantifying the Efficiency-Fairness Trade-off

**What it demonstrates:**
- **Price of Fairness (PoF)** = (max_flow − fair_flow) / max_flow, computed
  across different network topologies (parallel, mesh, random).
- **Jain's Fairness Index** $J = (\sum x_i)^2 / (n \sum x_i^2)$ applied to
  source-edge flow distributions.
- **Pareto frontier** computation via ε-constrained optimization:
  sweep a target throughput and maximize fairness; sweep a target Jain index
  and maximize throughput.
- **Multi-commodity fairness**: two source-sink pairs sharing a bottleneck
  edge, comparing utilitarian, proportional, and max-min allocations.
- Visualization: Pareto frontier scatter + α-sweep table.

**Key methods/constructs:**
- Jain index as a scalar fairness measure (from networking literature).
- SOCP-representable Jain constraint: $(\sum x)^2 \ge J \cdot n \cdot \sum x^2$.
- Multi-commodity flow decomposition with shared capacity constraints.
- `matplotlib` Pareto frontier plot with annotated extreme points.

**Why educational:** Directly answers "what do we lose by being fair?"
with numbers. The multi-commodity extension is the gateway to understanding
network utility maximization (NUM), the foundation of modern congestion
control (TCP, etc.). The Pareto frontier visualization makes the trade-off
tangible.

---

## Implementation Process (Step by Step)

### Phase 1: Duality Foundations
1. Start with a non-trivial flow network (not just Wikipedia's 6-node example).
2. Compute max flow via NetworkX.
3. Build the residual graph by comparing flow to capacity.
4. Run BFS from source in the residual graph → S-side of min cut.
5. Verify cut capacity equals max flow value.
6. Formulate the LP pair in CVXPY, solve, verify strong duality.
→ **Produces `example_max_flow_min_cut.py`**.

### Phase 2: Fairness Spectrum
1. Define the α-fairness utility function mathematically.
2. For each α ∈ {0, 0.5, 1, 2, 5, ∞}, build the CVXPY problem.
3. Handle edge cases: α=0 (linear, not strictly concave), α=1 (log), α→∞ (max-min via iterative fixing).
4. Compare edge flow distributions and total throughput.
5. Visualize the "price of fairness" curve.
→ **Produces `example_fairness_flow.py`**.

### Phase 3: Lagrangian Duality
1. Start with log-utility (α=1) for analytical tractability.
2. Write the Lagrangian L(f, λ, μ).
3. Derive the stationarity condition: $f_e = 1/(\lambda_e + \text{potential difference})$.
4. Substitute into the Lagrangian to get the dual function g(λ, μ).
5. Solve the dual numerically via CVXPY.
6. Verify KKT conditions: primal feasibility, dual feasibility, complementary slackness, stationarity.
7. Interpret λ_e (congestion toll) and μ_u (node potential) economically.
8. Extend to general α — note that the dual structure is similar but the stationarity condition becomes $f_e = (\lambda_e + a_e)^{-1/\alpha}$.
→ **Produces `example_fairness_dual.py`**.

### Phase 4: Efficiency-Fairness Frontier
1. Define Jain's fairness index and compute it for different α solutions.
2. Build the ε-constrained problems: (a) fix throughput, max fairness; (b) fix Jain, max throughput.
3. Sweep both constraints to trace the Pareto frontier.
4. Repeat for different network topologies to see how topology affects PoF.
5. Add the multi-commodity extension (shared bottleneck, two commodities).
6. Plot the Pareto frontier with extreme points annotated.
→ **Produces `example_price_of_fairness.py`**.

---

## Key Mathematical Constructs Used

| Construct | Where Used | Purpose |
|---|---|---|
| `nx.incidence_matrix` (oriented) | All files | Flow conservation: A·f = 0 at internal nodes |
| `nx.maximum_flow` | `max_flow_min_cut` | Baseline max flow (Edmonds-Karp) |
| Residual graph + BFS | `max_flow_min_cut` | Min cut extraction |
| CVXPY `cp.Variable` | All files | Decision variable for flow on each edge |
| `cp.log`, `cp.power`, `cp.inv_pos` | `fairness_flow`, `fairness_dual`, `price_of_fairness` | Concave utility functions |
| `cp.sum_squares` | `price_of_fairness` | SOCP representation of Jain constraint |
| Lagrangian KKT | `fairness_dual` | Duality gap, shadow prices |
| Lexicographic iteration | `fairness_flow` | Max-min fairness |
| Pareto ε-constraint sweep | `price_of_fairness` | Efficiency-fairness frontier |
| Multi-commodity decomposition | `price_of_fairness` | Shared resource allocation |

---

## Visualization Approaches — Conceptual Map

Visualizations for flow/fairness problems fall into five categories,
each illuminating a different facet of the mathematics:

### A. Graph-Centric (Topology + Flow)
**What you see:** The network itself, with flow/capacity on edges.
- Edge thickness ∝ flow volume
- Edge color ∝ utilization (green = slack, red = saturated)
- Node coloring by partition (S-side vs T-side for cuts)
- Highlighting of bottleneck/cut edges

**Tool:** `nx.draw_networkx_edges` with per-edge styling.
**Best for:** Intuition about which routes are congested and where the bottleneck lies.

### B. Constraint-Space (Polytope)
**What you see:** The feasible region as a geometric shape.
- For $n$ edges, the flow polytope lives in $\mathbb{R}^n$.
- Project to 2D by fixing $n-2$ variables to see the feasible polygon.
- Capacity constraints become half-planes; the optimum is a vertex.
- Objective contours show the gradient direction.

**Tool:** Matplotlib patches (`Rectangle`, `Polygon`) + contour lines.
**Best for:** Understanding why LP solutions lie at vertices, and how changing a
capacity constraint "pushes" a face of the polytope.

### C. Duality-Space (Gap + Saddle)
**What you see:** Primal/dual convergence and KKT geometry.
- **Gap plot:** Primal value rising, dual value falling — they meet at optimality.
- **Complementary slackness heatmap:** $\lambda_e \cdot (c_e - f_e)$ as a grid —
  should be zero everywhere at optimality (verifies KKT).
- **Saddle surface:** 3D plot of $L(f, \lambda)$ — the saddle point is where
  $\nabla L = 0$, the primal maximizes over $f$ and the dual minimizes over $\lambda$.

**Tool:** 3D `plot_surface` + 2D `contour` for the Lagrangian.
**Best for:** Proving to yourself that duality isn't abstract — it's a geometric
saddle on a surface you can see.

### D. Fairness-Space (Trade-off Curves)
**What you see:** How the solution changes as fairness preference varies.
- **Bar chart matrix:** grouped bars of flow per edge, one group per α.
- **Price-of-fairness curve:** total throughput vs α (monotonically decreasing).
- **Jain index curve:** fairness metric vs α (monotonically increasing).
- **Pareto frontier:** Jain index vs throughput — the boundary of achievable
  (efficiency, fairness) pairs.

**Tool:** Multi-panel `plt.subplots` dashboard, dual-axis plots.
**Best for:** The central trade-off question: "How much throughput do I sacrifice
for a given level of fairness?"

### E. Temporal (Animation & Interaction)
**What you see:** The solution *evolving*.
- **Augmenting path sequence:** each frame = one Ford-Fulkerson augmentation,
  showing the residual graph path and cumulative flow.
- **α-sweep morphing:** continuous animation of flow redistribution as α
  slides from 0 (utilitarian) to ∞ (max-min).
- **Interactive slider:** drag α and watch the graph + bar chart update in
  real time (Jupyter + ipywidgets).

**Tool:** `matplotlib.animation.FuncAnimation`, `ipywidgets.interactive_output`.
**Best for:** Teaching — the augmenting-path animation makes Ford-Fulkerson
concrete; the α slider makes the fairness spectrum *feel* real.

### When to use which

| Question | Best Visualization |
|---|---|
| "Where is the bottleneck?" | Graph (utilization coloring + cut partition) |
| "Why is this LP optimal?" | Polytope (vertex + objective contours) |
| "Is duality tight?" | Gap plot + complementary slackness heatmap |
| "What does fairness cost?" | Fairness dashboard (throughput + Jain vs α) |
| "How does the algorithm work?" | Augmenting-path animation |
| "Let me explore α myself" | Interactive ipywidgets slider |

---

## What Was Intentionally Omitted

- **Dinic's / Push-Relabel algorithms**: The focus is on optimization formulations
  and duality, not on algorithmic efficiency. NetworkX's default suffices.
- **Integer flow requirements**: All flows are continuous (fractional). Integrality
  is not needed for the convex formulations studied.
- **Minimum-cost flow / circulation**: These are classical but orthogonal to the
  fairness theme. They could be a separate extension.
- **Dynamic / time-varying flows**: Out of scope; the focus is static networks.
- **Stochastic / robust flow**: Probability distributions over capacities are
  interesting but would require a different mathematical toolkit.
- **GPU acceleration / large-scale**: The examples use small networks for clarity.
  Scaling to millions of edges would need specialized solvers (e.g., Gurobi, Mosek).
- **LaTeX / formal proofs**: The examples are computational, not proof-theoretic.

---

## How to Extend in a Real Project

1. **Large-scale fairness**: Replace CVXPY with a specialized convex solver
   (Mosek, Gurobi) or a distributed ADMM implementation for networks with
   millions of edges (e.g., internet topology).

2. **Online / incremental fairness**: Extend to streaming settings where flows
   arrive and depart dynamically; maintain approximate fairness with regret bounds.

3. **Game-theoretic flow**: Model each commodity as a selfish agent playing a
   routing game (Wardrop equilibrium / Braess's paradox). Compare Nash equilibrium
   with the socially optimal fair allocation (price of anarchy).

4. **Fairness with deadlines**: Add time constraints (flow must reach sink within
   k time steps). This connects to the literature on timely flow networks.

5. **Charging / tolling mechanisms**: Use the dual variables λ_e (shadow prices)
   to design congestion tolls that incentivize users to choose fair routings.

6. **Fairness in machine learning pipelines**: Replace flow networks with
   computation graphs (DAGs of tensor operations); use α-fairness to allocate
   computational resources (memory, FLOPs) across competing ML tasks.

7. **Integration with the existing project**: Add these examples as cells in
   `Main.ipynb` or import them as modules. The `draw_network` function in
   `flow_network.py` can be extended to color edges by flow/capacity ratio
   or to highlight the min-cut partition.

---

### 5. `example_flow_visualizations.py`
**Concept:** Comprehensive visualization catalog for flow & fairness

**Visualizations provided:**

| # | Function | Type | What It Shows |
|---|---|---|---|
| 1 | `draw_flow_utilization` | Static graph | Edge colors by utilization (green→yellow→red), edge thickness by flow volume |
| 2 | `draw_min_cut_partition` | Static graph | S-side (blue) / T-side (orange) node coloring, cut edges in thick red, residual-graph BFS extraction |
| 3 | `draw_feasible_polytope_2d` | Constraint space | 2D projection of flow polytope for a 3-edge network; capacity half-planes, objective contours, optimum vertex |
| 4 | `draw_duality_visualizations` | Dual analysis | Left: primal-dual gap convergence plot. Right: complementary slackness heatmap (f, c, slack, λ, λ·(c−f) per edge) |
| 5 | `draw_fairness_dashboard` | Multi-panel | Top-left: grouped bar chart of edge flows by α. Top-right: total throughput vs α (price of fairness area). Bottom: Jain index + throughput dual-axis plot |
| 6 | `animate_augmenting_paths` | Animation | Ford-Fulkerson style: each frame highlights one augmenting path with cumulative flow, using matplotlib.animation |
| 7 | `animate_alpha_sweep` | Animation | Sweeps α from 0→5, updating both the flow graph (edge thickness) and bar chart per frame — shows how fairness redistributes flow |
| 8 | `interactive_alpha_slider` | Interactive (ipywidgets) | Real-time α slider that re-solves the convex program and updates graph + bar chart in a Jupyter notebook |
| 9 | `draw_saddle_point_dual` | 3D surface | Lagrangian $L(f,\lambda) = \log(f) - \lambda(f-c)$ for a 1-edge toy problem; 3D surface + 2D contour with saddle point marked |

**Key libraries used:**
- `matplotlib.animation.FuncAnimation` — frame-by-frame animations
- `ipywidgets.FloatSlider` + `interactive_output` — real-time interactivity in Jupyter
- `matplotlib.colors.LinearSegmentedColormap` / `Normalize` — utilization heatmaps
- `mpl_toolkits.mplot3d.Axes3D` — 3D saddle surface
- `plt.cm.RdYlGn` — diverging colormap for utilization (green=slack, red=saturated)

**Why educational:**
- The polytope visualization makes the "LP is optimizing over a convex polytope" idea concrete.
- The saddle point surface shows that Lagrangian duality is a geometric fact, not an algebraic trick — the saddle is where the primal (max over f) and dual (min over λ) meet.
- The augmenting-path animation turns the abstract Ford-Fulkerson algorithm into a visible process.
- The α-sweep animation shows the continuous deformation from utilitarianism to egalitarianism.
- The interactive slider lets users *feel* the fairness-efficiency trade-off by dragging α and watching flows redistribute.

**Usage in a notebook:**
```python
from example_flow_visualizations import *

G, pos = demo_network()
flow_val, flow_dict = nx.maximum_flow(G, "s", "t")

# Static visuals
draw_flow_utilization(G, flow_dict, pos)
draw_min_cut_partition(G, flow_dict, pos=pos)

# Animation (returned as object; display with HTML or save as gif)
ani = animate_augmenting_paths(G)
# In notebook: from IPython.display import HTML; HTML(ani.to_jshtml())

# Interactive (notebook only)
interactive_alpha_slider(G)
```
