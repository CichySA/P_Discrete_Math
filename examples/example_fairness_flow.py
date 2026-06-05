"""
Example: Fairness Objectives in Multi-Sink Flow Optimization
=============================================================
Extends the standard max flow problem with a family of fairness-aware
objectives that operate over **per-sink inflows** rather than per-edge
flows.  This naturally generalises to networks with multiple sinks.

The fairness unit is now the total flow arriving at each sink:
  S_k = ∑_{(u,k) ∈ E} f_{u,k}

Fairness family covered:
  1. α-fairness (parametric):  U_α(x) = x^(1-α) / (1-α)  for α ≥ 0, α ≠ 1
                                U_α(x) = log(x)           for α = 1
     - α = 0  →  utilitarian (linear, standard max flow)
     - α = 1  →  proportional fairness (Nash bargaining, sum of logs)
     - α = 2  →  harmonic mean fairness (minimizes potential delay)
     - α → ∞  →  max-min fairness (egalitarian, Rawlsian)

  2. Weighted fairness: each sink inflow gets a weight w_k.

  3. Lexicographic max-min: (removed — use α→∞ approximation via large α)

Educational goals:
  - Show how the choice of α dramatically changes flow distribution across sinks.
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


def build_multi_sink_network():
    """
    A network with three sinks (t1, t2, t3) sharing a common source.
    Fairness is measured over the total inflow arriving at each sink,
    not over individual edges.  This highlights the difference between
    per-edge and per-sink fairness.

    Topology:
        s → a → t1      (wide path to t1, capacity 50)
        s → b → t2      (wide path to t2, capacity 40)
        s → c → t3      (wide path to t3, capacity 30)
        s → d → t2      (narrow path to t2, capacity 5)
        s → d → t3      (narrow path to t3, capacity 5)
    Under utilitarian (α=0), the optimizer floods the wide paths.
    Under high α, the flow is split more evenly across all three sinks.
    """
    G = nx.DiGraph()
    edges = [
        ("s", "a", 50), ("a", "t1", 50),
        ("s", "b", 40), ("b", "t2", 40),
        ("s", "c", 30), ("c", "t3", 30),
        ("s", "d",  5), ("d", "t2",  5), ("d", "t3",  5),
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


# ── 3. Solve α-fairness flow (multi-sink aware) ─────────────────────────────
def solve_alpha_fair_flow(G, source="s", sinks=None, alpha=1.0, weights=None):
    """
    Maximize  ∑_k w_k · U_α(S_k)  subject to capacity and flow conservation,
    where S_k = ∑_{(u,k) ∈ E} f_{u,k} is the total inflow at sink k.

    Parameters
    ----------
    G        : networkx.DiGraph with 'capacity' edge attribute.
    source   : single source node.
    sinks    : list of sink nodes.  A single string is wrapped in a list.
    alpha    : fairness parameter (0 = linear, 1 = log, 2 = harmonic, …).
    weights  : per-sink weight vector (default: all ones).

    Returns (optimal_value, flow_per_edge_dict, edge_order, f_values).
    """
    # Normalise sinks to a list
    if sinks is None:
        sinks = ["t"]
    elif isinstance(sinks, str):
        sinks = [sinks]

    edges = list(G.edges())
    n_edges = len(edges)
    nodes = list(G.nodes())
    n_sinks = len(sinks)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)

    # Incidence matrix
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    # Build sink-inflow matrix B:  B[k, j] = 1 iff edge j points to sinks[k]
    B = np.zeros((n_sinks, n_edges))
    for j, (u, v) in enumerate(edges):
        for k, sk in enumerate(sinks):
            if v == sk:
                B[k, j] = 1.0

    if weights is None:
        weights = np.ones(n_sinks)

    # Decision variable
    f = cp.Variable(n_edges, nonneg=True)

    # Sink inflow vector  S = B @ f   (linear expression)
    S = B @ f

    # Objective: weighted sum of α-utilities applied to *per-sink inflows*
    if alpha == 0:
        # Linear: maximize net outflow from source (equivalently total sink inflow)
        s_idx = nodes.index(source)
        q_source = -np.asarray(A[s_idx, :]).flatten()
        objective = cp.Maximize(q_source @ f)
    elif alpha == 1:
        # Proportional fairness: maximize weighted sum of log sink inflows
        objective = cp.Maximize(cp.sum(cp.multiply(weights, cp.log(S + 1e-9))))
    elif alpha == 2:
        # Harmonic fairness: maximize -∑ w_k / S_k
        objective = cp.Maximize(cp.sum(-cp.multiply(weights, cp.inv_pos(S + 1e-9))))
    else:
        # General α-fairness
        if 0 < alpha < 1:
            term = cp.power(S + 1e-9, 1 - alpha) / (1 - alpha)
            objective = cp.Maximize(cp.sum(cp.multiply(weights, term)))
        elif alpha > 1:
            term = -cp.power(S + 1e-9, 1 - alpha) / (alpha - 1)
            objective = cp.Maximize(cp.sum(cp.multiply(weights, term)))
        else:
            raise ValueError(f"alpha={alpha} not supported")

    # Flow conservation at non-terminal nodes
    terminal = {source} | set(sinks)
    cons_idx = [nodes.index(n) for n in nodes if n not in terminal]

    constraints = [
        f <= cap,
        A[cons_idx, :] @ f == 0,
    ]

    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()

    # Build flow dict
    flow_dict = {u: {} for u in G.nodes()}
    for j, (u, v) in enumerate(edges):
        flow_dict[u][v] = float(f.value[j]) if f.value is not None else 0.0

    return opt_val, flow_dict, edges, f.value


# ── 5. Utility functions ─────────────────────────────────────────────────────
def compute_total_flow(flow_dict, source="s", sinks=None):
    """
    For a multi-sink network, total flow = sum of all inflow at every sink.
    Falls back to source outflow when sinks is not given.
    """
    if sinks is not None:
        if isinstance(sinks, str):
            sinks = [sinks]
        total = 0.0
        for sk in sinks:
            # sum over all edges pointing *into* sk
            for u, neighbours in flow_dict.items():
                total += neighbours.get(sk, 0.0)
        return total
    # Fallback: source outflow (single-sink compatible)
    return sum(flow_dict.get(source, {}).values())


# ── 6. Main demonstration (multi-sink) ──────────────────────────────────────
if __name__ == "__main__":
    G = build_multi_sink_network()
    SINKS = ["t1", "t2", "t3"]

    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}

    print("=== Multi-Sink α-Fairness Flow Optimization ===\n")
    for alpha in alphas:
        opt_val, flow_dict, edges, f_vals = solve_alpha_fair_flow(
            G, sinks=SINKS, alpha=alpha
        )
        total = compute_total_flow(flow_dict, sinks=SINKS)
        results[alpha] = {
            "total_flow": total,
            "flow_dict":   flow_dict,
            "edges":       edges,
            "f_vals":      f_vals,
        }

        label = ("utilitarian" if alpha == 0 else
                 "proportional" if alpha == 1 else
                 f"α={alpha}")
        print(f"α={alpha} ({label:>20s}): total flow = {total:.4f}")

    # ── Plot A: throughput vs α  +  per-sink inflow breakdown ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Left: total flow vs α  (edge utilization / flow comparison)
    ax = axes[0]
    alphas_plot = list(results.keys())
    totals = [results[a]["total_flow"] for a in alphas_plot]
    ax.plot(alphas_plot, totals, "o-", linewidth=2, markersize=8,
            color="#377eb8")
    ax.set_xlabel("α (fairness parameter)")
    ax.set_ylabel("Total flow into all sinks")
    ax.set_title("Price of Fairness: Throughput vs. α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: per-sink inflow for selected α's (the NEW fairness unit)
    ax = axes[1]
    sink_labels = SINKS
    x = np.arange(len(sink_labels))
    width = 0.18
    for i, alpha in enumerate([0.0, 1.0, 5.0]):
        flow_dict = results[alpha]["flow_dict"]
        sink_vals = []
        for sk in SINKS:
            inflow = sum(
                flow_dict[u].get(sk, 0.0)
                for u in flow_dict
            )
            sink_vals.append(inflow)
        offset = (i - 1) * width
        label = "α=0 (linear)" if alpha == 0 else f"α={alpha}"
        ax.bar(x + offset, sink_vals, width, label=label, alpha=0.8)

    ax.set_xlabel("Sink")
    ax.set_ylabel("Inflow")
    ax.set_title("Per-Sink Inflow by α")
    ax.set_xticks(x)
    ax.set_xticklabels(sink_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Right: per-edge flow distribution (kept for utilisation comparison)
    ax = axes[2]
    edges_labels = [f"{u}→{v}" for u, v in results[0.0]["edges"]]
    x_e = np.arange(len(edges_labels))
    for i, alpha in enumerate([0.0, 1.0, 5.0]):
        f_vals = results[alpha]["f_vals"]
        offset = (i - 1) * 0.15
        label = "α=0 (linear)" if alpha == 0 else f"α={alpha}"
        ax.bar(x_e + offset, f_vals, 0.15, label=label, alpha=0.8)

    ax.set_xlabel("Edge")
    ax.set_ylabel("Flow")
    ax.set_title("Edge Flow Distribution by α")
    ax.set_xticks(x_e)
    ax.set_xticklabels(edges_labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print("\nKey insight (multi-sink): The fairness unit is now the TOTAL flow")
    print("arriving at each sink, not individual edge flows.  The log-sum")
    print("objective ensures the optimizer balances inflow across sinks.")
    print("α=0  → pure utilitarian: maximise total throughput.")
    print("α=1  → proportional fairness (Nash) across sinks.")
    print("α→∞  → max-min (Rawlsian): equalise the smallest sink inflow.")
