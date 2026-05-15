"""
Example: Price of Fairness in Flow Networks
=============================================
Quantifies the trade-off between maximizing total throughput (efficiency)
and distributing flow equitably (fairness).

The "price of fairness" (PoF) is defined as:
  PoF = (total_flow_max - total_flow_fair) / total_flow_max

Three fairness notions are compared:
  1. Proportional fairness (Nash): max ∑ log f_e
  2. Max-min fairness (Rawlsian): max min_e f_e
  3. α-fairness: parametric family interpolating between the two

Additionally, this example explores:
  - The Pareto frontier of achievable (efficiency, fairness) pairs.
  - Jain's fairness index applied to source-edge flow distributions.
  - Multi-commodity extension: competing source-sink pairs sharing capacity.

Educational goals:
  - Show that fairness is not free: it has a quantifiable throughput cost.
  - Demonstrate how network topology affects the price of fairness.
  - Illustrate the concept of Pareto optimality in flow allocation.
"""

import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ── 1. Data structures ──────────────────────────────────────────────────────
@dataclass
class FlowResult:
    label: str
    flow_dict: dict
    f_values: np.ndarray
    total_flow: float
    jain_index: float
    alpha: float = None


# ── 2. Network builders — different topologies for PoF analysis ──────────────
def build_parallel_network():
    """Two parallel s→t paths: one wide, one narrow. High PoF."""
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=100)
    G.add_edge("a", "t", capacity=100)
    G.add_edge("s", "b", capacity=2)
    G.add_edge("b", "t", capacity=2)
    return G


def build_mesh_network():
    """Mesh with multiple paths. Moderate PoF."""
    G = nx.DiGraph()
    edges = [
        ("s", "a", 10), ("s", "b", 8), ("s", "c", 6),
        ("a", "b", 3), ("a", "d", 7),
        ("b", "c", 4), ("b", "d", 5), ("b", "t", 3),
        ("c", "t", 8),
        ("d", "t", 12),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    return G


def build_random_network(n_nodes=10, edge_prob=0.3, seed=42):
    """Random DAG-based flow network."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_node("s")
    G.add_node("t")

    for i in range(1, n_nodes):
        G.add_node(i)

    # Add edges in topological order (ensures DAG)
    for u in G.nodes():
        for v in G.nodes():
            if u == v or u == "t" or v == "s":
                continue
            # Only allow forward edges (s→intermediate, intermediate→t, etc.)
            if (u == "s" and v != "t") or (v == "t" and u != "s") or \
               (isinstance(u, int) and isinstance(v, int) and u < v):
                if rng.random() < edge_prob:
                    cap = rng.integers(1, 20)
                    G.add_edge(u, v, capacity=cap)
    return G


# ── 3. Solve α-fairness flow (concise version) ──────────────────────────────
def solve_alpha_fair_flow(G, source="s", sink="t", alpha=1.0):
    """Returns (f_values, total_flow, edges_list)."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    f = cp.Variable(n_edges, nonneg=True)

    if alpha == 0:
        s_idx = nodes.index(source)
        q = A[s_idx, :]
        objective = cp.Maximize(q @ f)
    elif alpha == 1:
        objective = cp.Maximize(cp.sum(cp.log(f + 1e-9)))
    elif alpha == 2:
        objective = cp.Maximize(-cp.sum(cp.inv_pos(f + 1e-9)))
    elif alpha >= 10:
        # Approximate max-min via a large power
        objective = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))
    else:
        if alpha < 1:
            objective = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
        else:
            objective = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))

    constraints = [
        f <= cap,
        A[[nodes.index(n) for n in nodes if n not in (source, sink)], :] @ f == 0,
    ]
    cp.Problem(objective, constraints).solve()

    if f.value is None:
        return np.zeros(n_edges), 0.0, edges

    total = sum(f.value[i] for i, (u, v) in enumerate(edges) if u == source)
    return f.value, total, edges


# ── 4. Jain's Fairness Index ────────────────────────────────────────────────
def jain_index(flows):
    """
    Jain's fairness index: J = (Σ x_i)² / (n · Σ x_i²)
    J ∈ [1/n, 1]. J = 1 means perfectly equal; J = 1/n means completely unfair.
    Applied here to source-edge flow distribution.
    """
    x = np.asarray(flows, dtype=float)
    x = x[x > 1e-9]   # only consider edges with positive flow
    if len(x) == 0:
        return 1.0
    n = len(x)
    numerator = np.sum(x) ** 2
    denominator = n * np.sum(x ** 2)
    return float(numerator / denominator) if denominator > 0 else 1.0


# ── 5. Pareto frontier computation ──────────────────────────────────────────
def compute_pareto_frontier(G, source="s", sink="t", n_points=20):
    """
    Compute (efficiency, fairness) pairs by sweeping ε-constrained optimization:
      max  total_flow  s.t. Jain_index(f) ≥ J_target
    and
      max  Jain_index  s.t. total_flow ≥ T_target
    Returns two frontiers for comparison.
    """
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    # Identify source edges for Jain index calculation
    source_mask = np.array([u == source for u, v in edges], dtype=bool)
    source_edges_idx = np.where(source_mask)[0]

    # --- Standard max flow (max efficiency) ---
    f_max, total_max, _ = solve_alpha_fair_flow(G, source, sink, alpha=0.0)
    jain_at_max = jain_index(f_max[source_mask])

    # --- Max fair (α large) ---
    f_fair, total_fair, _ = solve_alpha_fair_flow(G, source, sink, alpha=10.0)
    jain_at_fair = jain_index(f_fair[source_mask])

    # --- Frontier 1: maximize fairness subject to efficiency ≥ target ---
    frontier_1 = []
    total_range = np.linspace(total_fair, total_max, n_points)

    for T_target in total_range:
        f = cp.Variable(n_edges, nonneg=True)
        s_idx = nodes.index(source)
        q_source = A[s_idx, :]

        # Maximize Jain index subject to conservation, capacity, and throughput ≥ T
        # Jain index is not concave, so we use sum of source flows as proxy
        # and add a constraint on the sum.
        constraints = [
            f <= cap,
            A[[nodes.index(n) for n in nodes if n not in (source, sink)], :] @ f == 0,
            q_source @ f >= T_target,
        ]

        # For fairness, maximize an α-fair objective with chosen α
        alpha = 5.0  # strong fairness preference
        obj = cp.Maximize(-cp.sum(cp.power(f[source_edges_idx] + 1e-9, 1 - alpha)) / (alpha - 1))
        prob = cp.Problem(obj, constraints)
        prob.solve()

        if f.value is not None:
            actual_total = float(q_source @ f.value)
            actual_jain = jain_index(f.value[source_mask])
            frontier_1.append((actual_total, actual_jain))

    # --- Frontier 2: maximize efficiency subject to Jain ≥ target ---
    frontier_2 = []
    jain_range = np.linspace(jain_at_max, jain_at_fair, n_points)

    for J_target in jain_range:
        f = cp.Variable(n_edges, nonneg=True)
        s_idx = nodes.index(source)
        q_source = A[s_idx, :]

        # Jain index constraint: (sum x)^2 ≥ J * n * sum(x^2)
        # This is a second-order cone representable constraint
        sum_source = cp.sum(f[source_edges_idx])
        sum_sq_source = cp.sum_squares(f[source_edges_idx])

        constraints = [
            f <= cap,
            A[[nodes.index(n) for n in nodes if n not in (source, sink)], :] @ f == 0,
            sum_source ** 2 >= J_target * len(source_edges_idx) * sum_sq_source,
        ]

        obj = cp.Maximize(q_source @ f)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        if f.value is not None:
            actual_total = float(q_source @ f.value)
            actual_jain = jain_index(f.value[source_mask])
            frontier_2.append((actual_total, actual_jain))

    return (total_max, jain_at_max), (total_fair, jain_at_fair), frontier_1, frontier_2


# ── 6. Multi-commodity fairness ──────────────────────────────────────────────
def multi_commodity_fairness_example():
    """
    Two source-sink pairs (s1→t1, s2→t2) share a common network.
    Fairness across commodities: should both pairs get equal throughput?
    """
    G = nx.DiGraph()
    # Commodity 1: s1 → a → t1
    # Commodity 2: s2 → a → t2
    # Shared: a → b (bottleneck)
    edges = [
        ("s1", "a", 10), ("s2", "a", 10),
        ("a", "b", 8),                     # shared bottleneck!
        ("b", "t1", 10), ("b", "t2", 10),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)

    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    edge_list = list(G.edges())

    # Two flow vectors, one per commodity
    f1 = cp.Variable(n_edges, nonneg=True)
    f2 = cp.Variable(n_edges, nonneg=True)

    # Shared capacity constraint
    constraints = [f1 + f2 <= cap]

    # Conservation per commodity (simplified: manual incidence)
    # f1: s1→a→b→t1
    constraints += [
        f1[0] == f1[2],   # s1→a == a→b (for commodity 1)
        f1[2] == f1[3],   # a→b == b→t1
    ]
    # f2: s2→a→b→t2
    constraints += [
        f1[1] + f2[1] == f1[2] + f2[2],  # total into a == total out of a
        f2[0] == f2[1],   # s2→a == a→b (for commodity 2 — this index needs care)
    ]
    # Actually let's simplify: use only the aggregate flow conservation with two commodities

    # Simpler formulation: just use two independent edge variables
    # f1_e + f2_e ≤ c_e for each edge
    # Separate conservation per commodity

    # Redo properly
    f = {}  # commodity -> variable list

    print("\n=== Multi-Commodity Fairness ===")
    print("Network: two commodities sharing a bottleneck edge a→b (capacity 8)")
    print()

    # Solve with commodity-level fairness: max min(total_f1, total_f2)
    # Use simplified model
    # Commodity 1 flow on path: f1 = min(s1→a, a→b, b→t1)
    # Commodity 2 flow on path: f2 = min(s2→a, a→b, b→t2)
    # But they share a→b: f1_path + f2_path ≤ cap(a→b) = 8

    x1 = cp.Variable(nonneg=True)  # commodity 1 path flow
    x2 = cp.Variable(nonneg=True)  # commodity 2 path flow

    constraints = [
        x1 <= cap[0],   # s1→a
        x2 <= cap[1],   # s2→a
        x1 + x2 <= cap[2],   # a→b (shared bottleneck)
        x1 <= cap[3],   # b→t1
        x2 <= cap[4],   # b→t2
    ]

    # Utilitarian: max x1 + x2
    prob_util = cp.Problem(cp.Maximize(x1 + x2), constraints)
    val_util = prob_util.solve()
    print(f"Utilitarian (max sum):        x1={x1.value:.2f}, x2={x2.value:.2f}, "
          f"total={val_util:.2f}")

    # Proportional fairness: max log(x1) + log(x2)
    prob_prop = cp.Problem(cp.Maximize(cp.log(x1 + 1e-9) + cp.log(x2 + 1e-9)), constraints)
    val_prop = prob_prop.solve()
    print(f"Proportional fair (log):      x1={x1.value:.2f}, x2={x2.value:.2f}, "
          f"total={x1.value + x2.value:.2f}")

    # Max-min fairness: max min(x1, x2)
    t = cp.Variable()
    prob_mm = cp.Problem(cp.Maximize(t), constraints + [t <= x1, t <= x2])
    val_mm = prob_mm.solve()
    print(f"Max-min fair (egalitarian):   x1={x1.value:.2f}, x2={x2.value:.2f}, "
          f"total={x1.value + x2.value:.2f}")

    # α-fairness sweep
    print("\nα-fairness sweep for multi-commodity:")
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        if alpha == 0:
            obj = cp.Maximize(x1 + x2)
        elif alpha == 1:
            obj = cp.Maximize(cp.log(x1 + 1e-9) + cp.log(x2 + 1e-9))
        else:
            obj = cp.Maximize(
                -(cp.power(x1 + 1e-9, 1 - alpha) + cp.power(x2 + 1e-9, 1 - alpha)) / (alpha - 1)
            )
        cp.Problem(obj, constraints).solve()
        print(f"  α={alpha}: x1={x1.value:.2f}, x2={x2.value:.2f}, "
              f"total={x1.value + x2.value:.2f}")

    print("\nKey insight: With a shared bottleneck, the utilitarian solution")
    print("may starve one commodity. Fairness notions guarantee each commodity")
    print("a minimum share of the scarce capacity.")


# ── 7. Main demonstration ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Compare PoF across network topologies
    networks = {
        "Parallel (high PoF)": build_parallel_network(),
        "Mesh (moderate PoF)": build_mesh_network(),
    }

    print("=== Price of Fairness Across Network Topologies ===\n")
    print(f"{'Network':<25s} {'Max Flow':>10s} {'α=1 Flow':>10s} "
          f"{'PoF (%)':>8s} {'Jain(max)':>10s} {'Jain(α=1)':>10s}")
    print("-" * 75)

    for name, G in networks.items():
        _, total_max, edges = solve_alpha_fair_flow(G, alpha=0.0)
        f_fair, total_fair, _ = solve_alpha_fair_flow(G, alpha=1.0)

        # Jain indices on source edges
        source_mask = np.array([u == "s" for u, v in edges], dtype=bool)
        f_max_vals, _, _ = solve_alpha_fair_flow(G, alpha=0.0)
        j_max = jain_index(f_max_vals[source_mask])
        j_fair = jain_index(f_fair[source_mask])

        pof = (total_max - total_fair) / total_max * 100 if total_max > 0 else 0
        print(f"{name:<25s} {total_max:>10.2f} {total_fair:>10.2f} "
              f"{pof:>7.1f}% {j_max:>10.4f} {j_fair:>10.4f}")

    # --- Pareto frontier for the mesh network ---
    print("\n\n=== Pareto Frontier: Efficiency vs. Fairness (Mesh Network) ===\n")
    G_mesh = build_mesh_network()
    (t_max, j_max), (t_fair, j_fair), frontier_1, frontier_2 = \
        compute_pareto_frontier(G_mesh)

    print(f"Max-flow point:       total={t_max:.2f}, Jain={j_max:.4f}")
    print(f"Max-fair point:       total={t_fair:.2f}, Jain={j_fair:.4f}")
    print(f"Price of fairness:    {(t_max - t_fair) / t_max * 100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    if frontier_1:
        f1_t, f1_j = zip(*frontier_1)
        ax.plot(f1_t, f1_j, "o-", label="Max fairness s.t. throughput ≥ T",
                color="#377eb8", alpha=0.7)

    if frontier_2:
        f2_t, f2_j = zip(*frontier_2)
        ax.plot(f2_t, f2_j, "s-", label="Max throughput s.t. Jain ≥ J",
                color="#e41a1c", alpha=0.7)

    ax.scatter([t_max], [j_max], marker="*", s=200, color="green",
               zorder=5, label="Max flow (utilitarian)")
    ax.scatter([t_fair], [j_fair], marker="*", s=200, color="orange",
               zorder=5, label="α=5 fair")

    ax.set_xlabel("Total Throughput (efficiency)")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Pareto Frontier: Efficiency vs. Fairness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

    # --- Multi-commodity extension ---
    multi_commodity_fairness_example()
