"""
Example: Max Flow / Min s-t Cut Duality
=========================================
Demonstrates the classic Ford-Fulkerson / max-flow min-cut theorem:
  max flow value = min s-t cut capacity

The duality is explored from three angles:
  1. LP primal-dual pair (flow LP vs cut LP)
  2. Algorithmic view (residual graph → saturated edges → min cut)
  3. Geometric view (capacity polytope and separating hyperplane)

Educational goals:
  - Show that the min cut is exactly the set of nodes reachable from s
    in the residual graph after a max flow is pushed.
  - Visualize the cut partition and the bottleneck edges.
  - Explicitly solve both the primal (max flow LP) and dual (min cut LP)
    using CVXPY and verify strong duality.
"""

import networkx as nx
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from collections import deque


# ── 1. Build a non-trivial flow network ──────────────────────────────────────
def build_network():
    """Return a DiGraph with capacities. Source = 's', sink = 't'."""
    G = nx.DiGraph()
    edges = [
        ("s", "a", 10), ("s", "b", 8),
        ("a", "b", 2),  ("a", "c", 6),  ("a", "d", 4),
        ("b", "c", 5),  ("b", "e", 7),
        ("c", "d", 3),  ("c", "e", 2),  ("c", "t", 8),
        ("d", "t", 9),
        ("e", "d", 1),  ("e", "t", 6),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    return G


# ── 2. Compute max flow and extract min cut via residual reachability ───────
def max_flow_and_min_cut(G, source="s", sink="t"):
    """
    Returns (flow_value, flow_dict, min_cut_partition).
    min_cut_partition is the set of nodes reachable from source in the
    residual graph — this is the S-side of the minimum s-t cut.
    """
    # Compute max flow (NetworkX uses Edmonds-Karp or shortest-augmenting-path)
    flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity="capacity")

    # Build residual graph: an edge (u,v) is in residual if flow < capacity
    R = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data["capacity"]
        f = flow_dict[u][v]
        if f < cap:
            R.add_edge(u, v, residual=cap - f)         # forward residual
        if f > 0:
            R.add_edge(v, u, residual=f)                # backward residual

    # Nodes reachable from source in the residual graph form the S-side
    S = set()
    queue = deque([source])
    S.add(source)
    while queue:
        u = queue.popleft()
        for v in R.successors(u):
            if v not in S:
                S.add(v)
                queue.append(v)

    T = set(G.nodes()) - S   # T-side
    return flow_value, flow_dict, S, T


# ── 3. Compute min cut capacity directly (sum of capacities crossing S→T) ───
def cut_capacity(G, S, T):
    """Sum capacities of edges from S to T."""
    total = 0
    for u, v, data in G.edges(data=True):
        if u in S and v in T:
            total += data["capacity"]
    return total


# ── 4. LP formulations: primal (max flow) and dual (min cut) ─────────────────
def solve_flow_lp_pair(G, source="s", sink="t"):
    """
    Solves both the primal max-flow LP and the dual min-cut LP using CVXPY.
    Returns (primal_value, dual_value, x_opt, y_opt).
    Strong duality guarantees primal_value == dual_value.

    Primal (max flow):
      max  ∑_v f_{s,v} - ∑_u f_{u,s}
      s.t. 0 ≤ f_{u,v} ≤ c_{u,v}  ∀ edges
           ∑_v f_{u,v} - ∑_v f_{v,u} = 0  ∀ u ∉ {s,t}   (conservation)

    Dual (min cut):
      min  ∑_{(u,v)} c_{u,v} · d_{u,v}
      s.t. π_u - π_v + d_{u,v} ≥ 0  ∀ edges
           π_s = 1, π_t = 0
           d_{u,v} ≥ 0
    """
    edges = list(G.edges())
    n_edges = len(edges)
    nodes = list(G.nodes())

    # Capacity vector
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)

    # Incidence matrix A: A[node_idx, edge_idx] = +1 if edge enters, -1 if leaves
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    # --- Primal ---
    f = cp.Variable(n_edges, nonneg=True)
    # Source net outflow: rows corresponding to source
    s_idx = nodes.index(source)
    q = -np.asarray(A[s_idx, :]).flatten()   # negate: A[s,:] = -1 for outgoing

    primal_obj = cp.Maximize(q @ f)
    primal_constraints = [
        f <= cap,
        # Flow conservation at non-terminal nodes
        A[[nodes.index(n) for n in nodes if n not in (source, sink)], :] @ f == 0,
    ]
    primal = cp.Problem(primal_obj, primal_constraints)
    primal_value = primal.solve()

    # --- Dual ---
    # Variables: π_u (potential) for each node, d_{u,v} (edge indicator) for each edge
    pi = cp.Variable(len(nodes))
    d = cp.Variable(n_edges, nonneg=True)

    dual_obj = cp.Minimize(cap @ d)
    dual_constraints = []
    for j, (u, v) in enumerate(edges):
        ui = nodes.index(u)
        vi = nodes.index(v)
        # π_u - π_v + d_{u,v} ≥ 0
        dual_constraints.append(pi[ui] - pi[vi] + d[j] >= 0)
    # Fix potentials at terminals
    dual_constraints.append(pi[nodes.index(source)] == 1)
    dual_constraints.append(pi[nodes.index(sink)] == 0)

    dual = cp.Problem(dual_obj, dual_constraints)
    dual_value = dual.solve()

    return primal_value, dual_value, f.value, d.value, pi.value, edges, nodes


# ── 5. Main demonstration ───────────────────────────────────────────────────
if __name__ == "__main__":
    G = build_network()

    # --- Algorithmic approach ---
    flow_val, flow_dict, S, T = max_flow_and_min_cut(G)
    cut_cap = cut_capacity(G, S, T)

    print("=== Max Flow / Min Cut (Algorithmic) ===")
    print(f"Max flow value:       {flow_val}")
    print(f"S-side of min cut:    {S}")
    print(f"T-side of min cut:    {T}")
    print(f"Min cut capacity:     {cut_cap}")
    print(f"Duality gap:          {flow_val - cut_cap:.8f}")
    assert abs(flow_val - cut_cap) < 1e-9, "Duality violation!"

    # --- List bottleneck edges (S → T edges at capacity) ---
    print("\nBottleneck edges (crossing S→T, fully saturated):")
    for u, v, data in G.edges(data=True):
        if u in S and v in T:
            f_uv = flow_dict[u][v]
            cap_uv = data["capacity"]
            print(f"  {u}→{v}: flow={f_uv}, capacity={cap_uv}, "
                  f"{'SATURATED' if f_uv == cap_uv else 'slack=' + str(cap_uv - f_uv)}")

    # --- LP duality approach ---
    print("\n=== Max Flow / Min Cut (LP Duality) ===")
    pv, dv, f_opt, d_opt, pi_opt, edges, nodes = solve_flow_lp_pair(G)
    print(f"Primal (max flow) value: {pv:.6f}")
    print(f"Dual (min cut) value:    {dv:.6f}")
    print(f"Duality gap:             {pv - dv:.8f}")

    # Interpret dual: d_{u,v} ≈ 1 for cut edges, ≈ 0 otherwise
    print("\nDual edge variables d_{u,v} (≈1 means in the cut):")
    for j, (u, v) in enumerate(edges):
        if d_opt[j] > 0.5:   # threshold to separate cut edges
            print(f"  {u}→{v}: d={d_opt[j]:.4f}, capacity={G[u][v]['capacity']}")

    # --- Visualize the cut partition ---
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))

    # Color nodes by partition
    node_colors = ["#4daf4a" if n in S else "#e41a1c" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)

    # Draw all edges in grey
    nx.draw_networkx_edges(G, pos, edge_color="grey", alpha=0.3,
                           connectionstyle="arc3,rad=0.1")

    # Highlight cut edges (S→T) in red
    cut_edges = [(u, v) for u, v in G.edges() if u in S and v in T]
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color="red",
                           width=2.5, connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_labels(G, pos, font_size=12, font_color="white")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(u, v): f"{flow_dict[u][v]}/{data['capacity']}"
                     for u, v, data in G.edges(data=True)},
        font_size=8, connectionstyle="arc3,rad=0.1",
    )

    plt.title(f"Min s-t Cut: S (green) | T (red)\n"
              f"Max flow = Min cut = {flow_val}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\nKey insight: The max-flow min-cut theorem holds because the")
    print("LP for max flow and the LP for min cut are strong duals.")
    print("The S-side of the min cut is exactly the set of nodes")
    print("reachable from s in the residual graph after max flow.")
