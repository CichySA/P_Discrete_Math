"""
Example: Fairness Objectives in Flow Optimization
===================================================
Extends the standard max flow problem with a family of fairness-aware
objectives. The classical max flow maximizes total throughput ∑ f_{s,v},
which can starve certain paths. Fairness objectives redistribute flow
more equitably across competing source-to-sink routes.

Fairness family covered:
  1. α-fairness (parametric):  U_α(x) = x^(1-α) / (1-α)  for α ≥ 0, α ≠ 1
                                U_α(x) = log(x)           for α = 1
     - α = 0  →  utilitarian (linear, standard max flow)
     - α = 1  →  proportional fairness (Nash bargaining, sum of logs)
     - α = 2  →  harmonic mean fairness (minimizes potential delay)
     - α → ∞  →  max-min fairness (egalitarian, Rawlsian)

  2. Weighted fairness: each source-edge flow gets a weight w_i.

  3. Lexicographic max-min: iteratively maximize the smallest flow,
     then the second-smallest, etc.

Educational goals:
  - Show how the choice of α dramatically changes the flow distribution.
  - Demonstrate that fairness reduces total throughput (price of fairness).
  - Visualize the Pareto frontier of efficiency vs. fairness.
"""

import networkx as nx
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import combinations


# ── 1. Build a network where fairness matters ───────────────────────────────
def build_asymmetric_network():
    """
    A network with one "wide" path (high capacity) and one "narrow" path.
    Under max-flow, the wide path dominates; under fairness, the narrow
    path receives more flow.
    """
    G = nx.DiGraph()
    # Two parallel s→t paths:
    #   s → a → t  (wide: capacity 100)
    #   s → b → t  (narrow: capacity 3)
    edges = [
        ("s", "a", 100), ("a", "t", 100),
        ("s", "b", 3),   ("b", "t", 3),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    return G


def build_mesh_network():
    """
    A richer network with multiple source edges and intermediate nodes,
    where fairness trade-offs become non-trivial.
    """
    G = nx.DiGraph()
    edges = [
        # Source edges (the "supply" that fairness distributes among)
        ("s", "a", 10), ("s", "b", 8), ("s", "c", 6),
        # Intermediate mesh
        ("a", "b", 3), ("a", "d", 7),
        ("b", "c", 4), ("b", "d", 5), ("b", "e", 3),
        ("c", "e", 8),
        ("d", "t", 12),
        ("e", "d", 2), ("e", "t", 6),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    return G


# ── 2. α-fairness utility function ──────────────────────────────────────────
def alpha_utility(x, alpha):
    """
    Returns the α-fairness utility of scalar/vector x.
    α = 0:  linear   U(x) = x           (utilitarian)
    α = 1:  log      U(x) = log(x)      (proportional fairness)
    α = 2:  negative reciprocal  U(x) = -1/x   (minimizes potential delay)
    α → ∞: max-min   U(x) = min(x)      (egalitarian — use separate formulation)
    """
    x = np.maximum(x, 1e-9)   # avoid log(0) or division by zero
    if alpha == 0:
        return x
    elif alpha == 1:
        return np.log(x)
    else:
        return np.power(x, 1 - alpha) / (1 - alpha)


# ── 3. Solve α-fairness flow ────────────────────────────────────────────────
def solve_alpha_fair_flow(G, source="s", sink="t", alpha=1.0, weights=None):
    """
    Maximize  ∑_i w_i · U_α(f_i)  subject to capacity and flow conservation,
    where f_i are the source-outgoing edge flows (or all edge flows).

    Returns (optimal_value, flow_per_edge_dict, edge_order).
    """
    edges = list(G.edges())
    n_edges = len(edges)
    nodes = list(G.nodes())
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)

    # Incidence matrix
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    if weights is None:
        weights = np.ones(n_edges)

    # Decision variable
    f = cp.Variable(n_edges, nonneg=True)

    # Objective: weighted sum of α-utilities
    if alpha == 0:
        # Linear: maximize weighted sum of source outflows
        s_idx = nodes.index(source)
        q_source = A[s_idx, :]
        objective = cp.Maximize(weights @ (q_source * f))
        # Note: weights on source edges directly
    elif alpha == 1:
        # Proportional fairness: maximize weighted sum of logs
        objective = cp.Maximize(weights @ cp.log(f + 1e-9))
    elif alpha == 2:
        # Harmonic fairness: minimize sum of 1/x (equivalent to max -∑ 1/x)
        objective = cp.Maximize(cp.sum(-weights / (f + 1e-9)))
    else:
        # General α-fairness using power cone / general power
        # For CVXPY compatibility, use the power atom when possible
        if 0 < alpha < 1:
            objective = cp.Maximize(weights @ (cp.power(f + 1e-9, 1 - alpha) / (1 - alpha)))
        elif alpha > 1:
            objective = cp.Maximize(weights @ (-cp.power(f + 1e-9, 1 - alpha) / (alpha - 1)))
        else:
            raise ValueError(f"alpha={alpha} not supported")

    constraints = [
        f <= cap,
        # Flow conservation at non-terminal nodes
        A[[nodes.index(n) for n in nodes if n not in (source, sink)], :] @ f == 0,
    ]

    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()

    # Build flow dict
    flow_dict = {u: {} for u in G.nodes()}
    for j, (u, v) in enumerate(edges):
        flow_dict[u][v] = float(f.value[j]) if f.value is not None else 0.0

    return opt_val, flow_dict, edges, f.value


# ── 4. Lexicographic max-min fairness ───────────────────────────────────────
def solve_max_min_fair_flow(G, source="s", sink="t", n_iter=10):
    """
    Iteratively maximize the minimum flow across source edges.
    After each step, fix the flows that hit their lower bound and
    maximize the next-smallest.

    Returns (flow_dict, source_flow_values).
    """
    edges = list(G.edges())
    n_edges = len(edges)
    nodes = list(G.nodes())
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    # Identify source edges
    source_mask = np.array([u == source for u, v in edges], dtype=bool)
    source_edges_idx = np.where(source_mask)[0]

    f = cp.Variable(n_edges, nonneg=True)
    constraints = [
        f <= cap,
        A[[nodes.index(n) for n in nodes if n not in (source, sink)], :] @ f == 0,
    ]

    fixed_lower = np.zeros(n_edges)

    for iteration in range(n_iter):
        # Maximize the minimum source flow (only among unfixed)
        if len(source_edges_idx) == 0:
            break

        # Objective: maximize min_{i in unfixed source edges} f_i
        t = cp.Variable()
        # For each unfixed source edge, require t ≤ f_i
        constraints_current = constraints + [t <= f[i] for i in source_edges_idx]
        constraints_current += [f[i] >= fixed_lower[i] for i in range(n_edges)]

        obj = cp.Maximize(t)
        prob = cp.Problem(obj, constraints_current)
        prob.solve()

        if f.value is None:
            break

        # Find the smallest source flow among unfixed
        source_vals = {i: f.value[i] for i in source_edges_idx}
        min_idx = min(source_vals, key=source_vals.get)
        min_val = f.value[min_idx]

        # Fix this edge's flow at its current value
        fixed_lower[min_idx] = min_val
        source_edges_idx = [i for i in source_edges_idx if i != min_idx]

        if len(source_edges_idx) == 0:
            break

    flow_dict = {u: {} for u in G.nodes()}
    for j, (u, v) in enumerate(edges):
        val = f.value[j] if f.value is not None else 0.0
        flow_dict[u][v] = float(val)

    return flow_dict, {edges[i]: fixed_lower[i] for i in np.where(source_mask)[0]}


# ── 5. Utility function (for evaluating any flow distribution) ──────────────
def compute_total_flow(flow_dict, source="s"):
    """Sum of all outflow from source."""
    return sum(flow_dict.get(source, {}).values())


# ── 6. Main demonstration ───────────────────────────────────────────────────
if __name__ == "__main__":
    G = build_mesh_network()

    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}

    print("=== α-Fairness Flow Optimization ===\n")
    for alpha in alphas:
        opt_val, flow_dict, edges, f_vals = solve_alpha_fair_flow(
            G, alpha=alpha
        )
        total = compute_total_flow(flow_dict)
        results[alpha] = {"total_flow": total, "flow_dict": flow_dict,
                          "f_vals": f_vals, "edges": edges}

        label = ("utilitarian" if alpha == 0 else
                 "proportional" if alpha == 1 else
                 f"α={alpha}")
        print(f"α={alpha} ({label:>20s}): total flow = {total:.4f}")

    # --- Also compute standard max flow and max-min fair ---
    std_flow_val, std_flow_dict = nx.maximum_flow(G, "s", "t", capacity="capacity")
    print(f"\nNetworkX max flow (LP):   total flow = {std_flow_val:.4f}")

    mm_flow_dict, mm_source = solve_max_min_fair_flow(G)
    mm_total = compute_total_flow(mm_flow_dict)
    print(f"Max-min fair (iterative): total flow = {mm_total:.4f}")
    print(f"  Source edge flows: {mm_source}")

    # --- Visualize: total flow vs α ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    alphas_plot = list(results.keys())
    totals = [results[a]["total_flow"] for a in alphas_plot]
    ax.plot(alphas_plot, totals, "o-", linewidth=2, markersize=8, color="#377eb8")
    ax.axhline(y=std_flow_val, color="grey", linestyle="--",
               label=f"Standard max flow = {std_flow_val:.2f}")
    ax.set_xlabel("α (fairness parameter)")
    ax.set_ylabel("Total flow")
    ax.set_title("Price of Fairness: Throughput vs. α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Visualize: per-edge flow distribution for selected α's ---
    ax = axes[1]
    edges_labels = [f"{u}→{v}" for u, v in results[0.0]["edges"]]
    x = np.arange(len(edges_labels))
    width = 0.15

    for i, alpha in enumerate([0.0, 1.0, 5.0]):
        f_vals = results[alpha]["f_vals"]
        offset = (i - 1) * width
        label = "α=0 (linear)" if alpha == 0 else f"α={alpha}"
        ax.bar(x + offset, f_vals, width, label=label, alpha=0.8)

    ax.set_xlabel("Edge")
    ax.set_ylabel("Flow")
    ax.set_title("Edge Flow Distribution by α")
    ax.set_xticks(x)
    ax.set_xticklabels(edges_labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print("\nKey insight: As α increases, the optimizer cares less about total")
    print("throughput and more about equitable distribution across edges.")
    print("α=0 is pure utilitarianism (standard max flow).")
    print("α=1 is proportional fairness (Nash bargaining solution).")
    print("α→∞ approaches max-min (Rawlsian) fairness.")
