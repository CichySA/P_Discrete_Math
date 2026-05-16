"""
demo_experiments.py — Standalone functions for every section of Demo_Experiments.ipynb
======================================================================================
Each function corresponds to one notebook code cell. This eliminates stale .pyc cache
issues and makes the notebook trivially simple: just import and call.

Notebook usage:
    from demo_experiments import *
    demo_section1_max_flow_min_cut()
    demo_section2_alpha_fairness()
    demo_section3_lagrangian_dual()
    demo_section4_price_of_fairness()
    demo_section5_visualizations()
"""

import networkx as nx
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import HTML, display, clear_output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Max Flow / Min s-t Cut Duality
# ═══════════════════════════════════════════════════════════════════════════════

def _build_network_mf():
    """Build non-trivial flow network for max-flow demo."""
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


def _max_flow_and_min_cut(G, source="s", sink="t"):
    """Returns (flow_value, flow_dict, S_set, T_set)."""
    flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity="capacity")

    R = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data["capacity"]
        f = flow_dict[u][v]
        if f < cap:
            R.add_edge(u, v)
        if f > 0:
            R.add_edge(v, u)

    S = set()
    q = deque([source])
    S.add(source)
    while q:
        u = q.popleft()
        for v in R.successors(u):
            if v not in S:
                S.add(v)
                q.append(v)
    T = set(G.nodes()) - S
    return flow_value, flow_dict, S, T


def _cut_capacity(G, S, T):
    return sum(data["capacity"] for u, v, data in G.edges(data=True) if u in S and v in T)


def demo_section1_max_flow_min_cut():
    """Demonstrate Max Flow / Min Cut duality (algorithmic + LP)."""
    G = _build_network_mf()

    # ── Algorithmic approach ──
    flow_val, flow_dict, S, T = _max_flow_and_min_cut(G)
    cut_cap = _cut_capacity(G, S, T)

    print("=== Max Flow / Min Cut (Algorithmic) ===")
    print(f"Max flow value:       {flow_val}")
    print(f"S-side of min cut:    {S}")
    print(f"T-side of min cut:    {T}")
    print(f"Min cut capacity:     {cut_cap}")
    print(f"Duality gap:          {flow_val - cut_cap:.8f}")

    print("\nBottleneck edges (crossing S→T, fully saturated):")
    for u, v, data in G.edges(data=True):
        if u in S and v in T:
            f_uv = flow_dict[u][v]
            cap_uv = data["capacity"]
            sat = "SATURATED" if abs(f_uv - cap_uv) < 1e-9 else f"slack={cap_uv - f_uv}"
            print(f"  {u}→{v}: flow={f_uv}, capacity={cap_uv}, {sat}")

    # ── LP Duality ──
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    # Primal (max flow): negate source row because A[s,:] has -1 for outgoing
    f = cp.Variable(n_edges, nonneg=True)
    s_idx = nodes.index("s")
    q = -np.asarray(A[s_idx, :]).flatten()
    primal_obj = cp.Maximize(q @ f)
    primal_constraints = [
        f <= cap,
        A[[nodes.index(n) for n in nodes if n not in ("s", "t")], :] @ f == 0,
    ]
    primal = cp.Problem(primal_obj, primal_constraints)
    pv = primal.solve()

    # Dual (min cut)
    pi = cp.Variable(len(nodes))
    d = cp.Variable(n_edges, nonneg=True)
    dual_obj = cp.Minimize(cap @ d)
    dual_constraints = []
    for j, (u, v) in enumerate(edges):
        ui, vi = nodes.index(u), nodes.index(v)
        dual_constraints.append(pi[ui] - pi[vi] + d[j] >= 0)
    dual_constraints.append(pi[nodes.index("s")] == 1)
    dual_constraints.append(pi[nodes.index("t")] == 0)
    dual = cp.Problem(dual_obj, dual_constraints)
    dv = dual.solve()

    print("\n=== Max Flow / Min Cut (LP Duality) ===")
    print(f"Primal (max flow) value: {pv:.6f}")
    print(f"Dual (min cut) value:    {dv:.6f}")
    print(f"Duality gap:             {pv - dv:.8f}")

    print("\nDual edge variables d_{u,v} (=1 means in the cut):")
    for j, (u, v) in enumerate(edges):
        if d.value[j] > 0.5:
            print(f"  {u}→{v}: d={d.value[j]:.4f}, capacity={G[u][v]['capacity']}")

    # ── Visualize cut partition ──
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    node_colors = ["#4daf4a" if n in S else "#e41a1c" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(G, pos, edge_color="grey", alpha=0.3,
                           connectionstyle="arc3,rad=0.1")
    cut_edges_list = [(u, v) for u, v in G.edges() if u in S and v in T]
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges_list, edge_color="red",
                           width=2.5, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="white")
    nx.draw_networkx_edge_labels(G, pos,
        edge_labels={(u, v): f"{flow_dict[u][v]}/{data['capacity']}"
                     for u, v, data in G.edges(data=True)},
        font_size=8, connectionstyle="arc3,rad=0.1")
    ax.set_title(f"Min s-t Cut: S (green) | T (red)\nMax flow = Min cut = {flow_val}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. α-Fairness Flow Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mesh_network():
    """Mesh network for fairness demo."""
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


def _solve_alpha_fair_flow(G, source="s", sink="t", alpha=1.0):
    """Solve α-fairness flow optimization using CVXPY."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    f = cp.Variable(n_edges, nonneg=True)
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]
    constraints = [f <= cap, A[cons_idx, :] @ f == 0]

    if alpha == 0:
        s_idx = nodes.index(source)
        q_source = -np.asarray(A[s_idx, :]).flatten()
        objective = cp.Maximize(q_source @ f)
    elif alpha == 1:
        objective = cp.Maximize(cp.sum(cp.log(f + 1e-9)))
    elif alpha == 2:
        objective = cp.Maximize(-cp.sum(cp.inv_pos(f + 1e-9)))
    elif alpha < 1:
        objective = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
    else:  # alpha > 1
        objective = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))

    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()

    flow_dict = {u: {} for u in G.nodes()}
    for j, (u, v) in enumerate(edges):
        flow_dict[u][v] = float(f.value[j]) if f.value is not None else 0.0

    return opt_val, flow_dict, edges, f.value


def _solve_max_min_fair_flow(G, source="s", sink="t", n_iter=10):
    """Iterative lexicographic max-min fairness."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]

    source_mask = np.array([u == source for u, v in edges], dtype=bool)
    source_edges_idx = list(np.where(source_mask)[0])

    f = cp.Variable(n_edges, nonneg=True)
    base_constraints = [f <= cap, A[cons_idx, :] @ f == 0]
    fixed_lower = np.zeros(n_edges)

    for iteration in range(n_iter):
        if len(source_edges_idx) == 0:
            break

        t = cp.Variable()
        constraints = base_constraints + [t <= f[i] for i in source_edges_idx]
        constraints += [f[i] >= fixed_lower[i] for i in range(n_edges)]

        prob = cp.Problem(cp.Maximize(t), constraints)
        prob.solve()

        if f.value is None:
            break

        source_vals = {i: f.value[i] for i in source_edges_idx}
        min_idx = min(source_vals, key=source_vals.get)
        min_val = f.value[min_idx]

        fixed_lower[min_idx] = min_val
        source_edges_idx = [i for i in source_edges_idx if i != min_idx]

    flow_dict = {u: {} for u in G.nodes()}
    for j, (u, v) in enumerate(edges):
        flow_dict[u][v] = float(f.value[j]) if f.value is not None else 0.0

    return flow_dict, {edges[i]: fixed_lower[i] for i in np.where(source_mask)[0]}


def _compute_total_flow(flow_dict, source="s"):
    return sum(flow_dict.get(source, {}).values())


def demo_section2_alpha_fairness():
    """Demonstrate α-Fairness Flow Optimization."""
    G = _build_mesh_network()

    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}

    print("=== α-Fairness Flow Optimization ===\n")
    for alpha in alphas:
        opt_val, flow_dict, edges, f_vals = _solve_alpha_fair_flow(G, alpha=alpha)
        total = _compute_total_flow(flow_dict)
        results[alpha] = {"total_flow": total, "flow_dict": flow_dict,
                          "f_vals": f_vals, "edges": edges}
        label = ("utilitarian" if alpha == 0 else
                 "proportional" if alpha == 1 else f"α={alpha}")
        print(f"α={alpha} ({label:>20s}): total flow = {total:.4f}")

    std_flow_val, std_flow_dict = nx.maximum_flow(G, "s", "t", capacity="capacity")
    print(f"\nNetworkX max flow (LP):   total flow = {std_flow_val:.4f}")

    mm_flow_dict, mm_source = _solve_max_min_fair_flow(G)
    mm_total = _compute_total_flow(mm_flow_dict)
    print(f"Max-min fair (iterative): total flow = {mm_total:.4f}")
    print(f"  Source edge flows: {mm_source}")

    # ── Visualize ──
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

    print("\nKey insight: α=0 is utilitarianism, α=1 is proportional fairness,")
    print("α→∞ approaches max-min (Rawlsian) fairness.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Lagrangian Dual & KKT Conditions
# ═══════════════════════════════════════════════════════════════════════════════

def _build_simple_network():
    """Minimal 4-node network for duality exposition."""
    G = nx.DiGraph()
    edges = [
        ("s", "a", 5), ("s", "b", 3),
        ("a", "t", 4), ("b", "t", 3),
        ("a", "b", 1),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    return G


def _solve_primal_fairness(G, source="s", sink="t", alpha=1.0):
    """Solve α-fairness primal."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    f = cp.Variable(n_edges, nonneg=True)
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]
    constraints = [f <= cap, A[cons_idx, :] @ f == 0]

    if alpha == 0:
        s_idx = nodes.index(source)
        q = -np.asarray(A[s_idx, :]).flatten()
        objective = cp.Maximize(q @ f)
    elif alpha == 1:
        objective = cp.Maximize(cp.sum(cp.log(f + 1e-9)))
    elif alpha < 1:
        objective = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
    else:
        objective = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))

    prob = cp.Problem(objective, constraints)
    opt_val = prob.solve()
    return opt_val, f.value, edges, nodes, A, cap


def _lagrangian_analysis(G, f_opt, edges, nodes, A, cap, source="s", sink="t"):
    """Construct Lagrangian dual for log-utility flow, verify KKT."""
    n_edges = len(edges)
    n_nodes = len(nodes)

    print("=== Lagrangian Dual of Log-Fairness Flow ===\n")

    source_edges = [(u, v) for u, v in edges if u == source]
    sink_edges = [(u, v) for u, v in edges if v == sink]
    internal_edges = [(u, v) for u, v in edges if u != source and v != sink]

    print("Network structure:")
    print(f"  Source edges: {source_edges}")
    print(f"  Internal edges: {internal_edges}")
    print(f"  Sink edges: {sink_edges}")

    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]
    A_cons = A[cons_idx, :]

    # ── Dual variables ──
    lam = cp.Variable(n_edges, nonneg=True)
    mu_all = cp.Variable(n_nodes)
    mu_int = mu_all[cons_idx]   # only internal node potentials matter

    a = A_cons.T @ mu_int
    denom = lam + a + 1e-12
    g_lambda_mu = -cp.sum(cp.log(denom)) - n_edges + lam @ cap

    dual_obj = cp.Minimize(g_lambda_mu)
    dual_prob = cp.Problem(dual_obj)
    dual_val = dual_prob.solve()

    lam_opt = lam.value
    mu_all_opt = mu_all.value
    mu_opt = mu_all_opt   # full vector
    a_opt = A_cons.T @ mu_all_opt[cons_idx]
    f_dual = 1.0 / (lam_opt + a_opt + 1e-12)

    print(f"\nPrimal optimal value:     {sum(np.log(np.maximum(f_opt, 1e-12))):.6f}")
    print(f"Dual optimal value:       {-dual_val:.6f} (negated since we minimized)")

    print("\n--- Dual Reconstruction of Primal Flows ---")
    for j, (u, v) in enumerate(edges):
        print(f"  {u}→{v}: primal f={f_opt[j]:.4f}, dual-recovered f={f_dual[j]:.4f}, "
              f"λ={lam_opt[j]:.4f}, a={a_opt[j]:.4f}")

    print("\n--- Complementary Slackness (λ_e · (c_e - f_e) ≈ 0) ---")
    for j, (u, v) in enumerate(edges):
        slack = lam_opt[j] * (cap[j] - f_opt[j])
        status = "✓" if abs(slack) < 1e-4 else "VIOLATION"
        print(f"  {u}→{v}: λ={lam_opt[j]:.4f}, c-f={cap[j]-f_opt[j]:.4f}, product={slack:.6f} {status}")

    print("\n--- Economic Interpretation ---")
    print("λ_e  = shadow price of capacity on edge e (congestion toll)")
    print("μ_u  = potential value of one unit of flow at node u")

    print("\n--- Comparison: Standard Max Flow Dual (Min Cut) ---")
    print("In standard max flow, the dual has an integral optimal solution")
    print("with μ ∈ {0,1} (cut indicator) and λ ∈ {0,1} (cut edge indicator).")

    return lam_opt, mu_opt, f_dual


def _dual_sensitivity(G, source="s", sink="t"):
    """Trace dual variables across α values."""
    print("\n\n=== Dual Sensitivity Across α Values ===\n")

    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        opt_val, f_opt, edges, nodes, A, cap = _solve_primal_fairness(G, source, sink, alpha)
        total_flow = sum(f_opt[i] for i, (u, v) in enumerate(edges) if u == source)

        label = ("utilitarian" if alpha == 0 else
                 "proportional" if alpha == 1 else f"α={alpha}")

        lam_est = np.zeros(len(edges))
        for j, (u, v) in enumerate(edges):
            if f_opt[j] >= cap[j] * 0.999:
                lam_est[j] = 1.0 / (f_opt[j] ** alpha) if f_opt[j] > 1e-9 else 1e6

        print(f"{label:>20s} (α={alpha}): total flow={total_flow:.4f}")
        print(f"  Estimated λ (capacity shadow prices):")
        for j, (u, v) in enumerate(edges):
            if lam_est[j] > 0.001:
                print(f"    {u}→{v}: λ≈{lam_est[j]:.4f} (f={f_opt[j]:.4f}, cap={cap[j]:.4f})")

    print("\nKey insight: As α increases, the marginal utility of flow")
    print("diminishes more rapidly, making the optimizer more willing")
    print("to sacrifice total throughput for equal distribution.")


def demo_section3_lagrangian_dual():
    """Demonstrate Lagrangian Dual & KKT Conditions."""
    G = _build_simple_network()
    opt_val, f_opt, edges, nodes, A, cap = _solve_primal_fairness(G, alpha=1.0)

    print("Primal solution (log utility, proportional fairness):")
    for j, (u, v) in enumerate(edges):
        print(f"  f({u}→{v}) = {f_opt[j]:.4f}  (capacity = {cap[j]:.0f})")

    print(f"\nOptimal log-utility value: {opt_val:.6f}")

    lam_opt, mu_opt, f_dual = _lagrangian_analysis(G, f_opt, edges, nodes, A, cap)

    print(f"\n{'─' * 50}")
    print("Dual variables μ (node potentials):")
    for j, node in enumerate(nodes):
        if node not in ("s", "t"):
            print(f"  μ_{node} = {mu_opt[j]:.6f}")

    print("\nDual variables λ (capacity shadow prices):")
    for j, (u, v) in enumerate(edges):
        if lam_opt[j] > 0.001:
            print(f"  λ_({u}→{v}) = {lam_opt[j]:.6f} (edge saturated)")

    _dual_sensitivity(G)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Price of Fairness & Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parallel_network():
    """Two parallel s→t paths: wide + narrow."""
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=100)
    G.add_edge("a", "t", capacity=100)
    G.add_edge("s", "b", capacity=2)
    G.add_edge("b", "t", capacity=2)
    return G


def _jain_index(flows):
    """J = (Σx)² / (n·Σx²). J ∈ [1/n, 1]."""
    x = np.asarray(flows, dtype=float)
    x = x[x > 1e-9]
    if len(x) <= 1:
        return 1.0
    n = len(x)
    return float(np.sum(x) ** 2 / (n * np.sum(x ** 2)))


def _compute_pareto_frontier(G, source="s", sink="t", n_points=20):
    """Compute (efficiency, fairness) pairs via ε-constrained optimization."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]

    source_mask = np.array([u == source for u, v in edges], dtype=bool)
    source_edges_idx = list(np.where(source_mask)[0])

    # Extremes
    _, flow_dict_max, _, f_max = _solve_alpha_fair_flow(G, source, sink, alpha=0.0)
    total_max = _compute_total_flow(flow_dict_max)
    jain_at_max = _jain_index(f_max[source_mask])

    _, flow_dict_fair, _, f_fair = _solve_alpha_fair_flow(G, source, sink, alpha=10.0)
    total_fair = _compute_total_flow(flow_dict_fair)
    jain_at_fair = _jain_index(f_fair[source_mask])

    # Frontier 1: max fairness s.t. throughput ≥ T
    frontier_1 = []
    for T_target in np.linspace(total_fair, total_max, n_points):
        f = cp.Variable(n_edges, nonneg=True)
        s_idx = nodes.index(source)
        q_source = -np.asarray(A[s_idx, :]).flatten()

        constraints = [
            f <= cap,
            A[cons_idx, :] @ f == 0,
            q_source @ f >= T_target,
        ]
        # max fairness ≡ max sum of logs on source edges  (α=1, proportional)
        obj = cp.Maximize(cp.sum(cp.log(f[source_edges_idx] + 1e-6)))
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve()
        except cp.error.DCPError:
            # fallback: maximize negative inverse of source flows (α=2 fairness)
            obj_fb = cp.Maximize(-cp.sum(cp.inv_pos(f[source_edges_idx] + 1e-6)))
            prob = cp.Problem(obj_fb, constraints)
            prob.solve()

        if f.value is not None:
            actual_total = float(q_source @ f.value)
            actual_jain = _jain_index(f.value[source_mask])
            frontier_1.append((actual_total, actual_jain))

    # Frontier 2: max throughput s.t. Jain ≥ J
    frontier_2 = []
    n_source = len(source_edges_idx)
    for J_target in np.linspace(jain_at_max, jain_at_fair, n_points):
        f = cp.Variable(n_edges, nonneg=True)
        s_idx = nodes.index(source)
        q_source = -np.asarray(A[s_idx, :]).flatten()

        # Jain constraint: (Σx)² ≥ J · n · Σx²  ⇔  Σx ≥ √(J·n) · ||x||₂   (SOC)
        # Use cp.SOC(t, x) which means ||x||₂ ≤ t
        t = cp.Variable()
        sum_source = cp.sum(f[source_edges_idx])

        constraints = [
            f <= cap,
            A[cons_idx, :] @ f == 0,
            cp.SOC(t, f[source_edges_idx]),
            t <= sum_source / np.sqrt(J_target * n_source),
        ]
        obj = cp.Maximize(q_source @ f)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        if f.value is not None:
            actual_total = float(q_source @ f.value)
            actual_jain = _jain_index(f.value[source_mask])
            frontier_2.append((actual_total, actual_jain))

    return (total_max, jain_at_max), (total_fair, jain_at_fair), frontier_1, frontier_2


def _multi_commodity_fairness():
    """Two source-sink pairs sharing a bottleneck edge."""
    G = nx.DiGraph()
    edges = [
        ("s1", "a", 10), ("s2", "a", 10),
        ("a", "b", 8), ("b", "t1", 10), ("b", "t2", 10),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)

    cap = np.array([G[u][v]["capacity"] for u, v, _ in edges], dtype=float)

    x1 = cp.Variable(nonneg=True)
    x2 = cp.Variable(nonneg=True)

    constraints = [
        x1 <= cap[0], x2 <= cap[1],
        x1 + x2 <= cap[2],
        x1 <= cap[3], x2 <= cap[4],
    ]

    print("\n=== Multi-Commodity Fairness ===")
    print("Network: two commodities sharing a bottleneck edge a→b (capacity 8)\n")

    # Utilitarian
    cp.Problem(cp.Maximize(x1 + x2), constraints).solve()
    print(f"Utilitarian (max sum):        x1={x1.value:.2f}, x2={x2.value:.2f}, "
          f"total={x1.value + x2.value:.2f}")

    # Proportional
    cp.Problem(cp.Maximize(cp.log(x1 + 1e-9) + cp.log(x2 + 1e-9)), constraints).solve()
    print(f"Proportional fair (log):      x1={x1.value:.2f}, x2={x2.value:.2f}, "
          f"total={x1.value + x2.value:.2f}")

    # Max-min
    t = cp.Variable()
    cp.Problem(cp.Maximize(t), constraints + [t <= x1, t <= x2]).solve()
    print(f"Max-min fair (egalitarian):   x1={x1.value:.2f}, x2={x2.value:.2f}, "
          f"total={x1.value + x2.value:.2f}")

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


def demo_section4_price_of_fairness():
    """Demonstrate Price of Fairness & Pareto Frontier."""
    networks = {
        "Parallel (high PoF)": _build_parallel_network(),
        "Mesh (moderate PoF)": _build_mesh_network(),
    }

    print("=== Price of Fairness Across Network Topologies ===\n")
    print(f"{'Network':<25s} {'Max Flow':>10s} {'α=1 Flow':>10s} "
          f"{'PoF (%)':>8s} {'Jain(max)':>10s} {'Jain(α=1)':>10s}")
    print("-" * 75)

    for name, G in networks.items():
        _, flow_dict_max, edges, f_vals_max = _solve_alpha_fair_flow(G, alpha=0.0)
        total_max = _compute_total_flow(flow_dict_max)
        _, flow_dict_fair, _, f_vals_fair = _solve_alpha_fair_flow(G, alpha=1.0)
        total_fair = _compute_total_flow(flow_dict_fair)

        source_mask = np.array([u == "s" for u, v in edges], dtype=bool)
        j_max = _jain_index(f_vals_max[source_mask])
        j_fair = _jain_index(f_vals_fair[source_mask])

        pof = (total_max - total_fair) / total_max * 100 if total_max > 0 else 0
        print(f"{name:<25s} {total_max:>10.2f} {total_fair:>10.2f} "
              f"{pof:>7.1f}% {j_max:>10.4f} {j_fair:>10.4f}")

    print("\n\n=== Pareto Frontier: Efficiency vs. Fairness (Mesh Network) ===\n")
    G_mesh = _build_mesh_network()
    (t_max, j_max), (t_fair, j_fair), frontier_1, frontier_2 = _compute_pareto_frontier(G_mesh)

    print(f"Max-flow point:       total={t_max:.2f}, Jain={j_max:.4f}")
    print(f"Max-fair point:       total={t_fair:.2f}, Jain={j_fair:.4f}")
    print(f"Price of fairness:    {(t_max - t_fair) / t_max * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 6))
    if frontier_1:
        f1_t, f1_j = zip(*frontier_1)
        ax.plot(f1_t, f1_j, "o-", label="Max fairness s.t. throughput ≥ T",
                color="#377eb8", alpha=0.7)
    if frontier_2:
        f2_t, f2_j = zip(*frontier_2)
        ax.plot(f2_t, f2_j, "s-", label="Max throughput s.t. Jain ≥ J",
                color="#e41a1c", alpha=0.7)
    ax.scatter([t_max], [j_max], marker="*", s=200, color="green", zorder=5,
               label="Max flow (utilitarian)")
    ax.scatter([t_fair], [j_fair], marker="*", s=200, color="orange", zorder=5,
               label="α=5 fair")
    ax.set_xlabel("Total Throughput (efficiency)")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Pareto Frontier: Efficiency vs. Fairness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    _multi_commodity_fairness()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization Catalog
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_alpha_quick(G, s, t, alpha):
    """Quick inline α-fairness solve for visualization dashboard."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    n = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
    ci = [nodes.index(n) for n in nodes if n not in (s, t)]
    f = cp.Variable(n, nonneg=True)
    if alpha == 0:
        si = nodes.index(s)
        obj = cp.Maximize(-np.asarray(A[si, :]).flatten() @ f)
    elif alpha == 1:
        obj = cp.Maximize(cp.sum(cp.log(f + 1e-9)))
    elif alpha < 1:
        obj = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
    else:
        obj = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))
    cp.Problem(obj, [f <= cap, A[ci, :] @ f == 0]).solve(verbose=False)
    return f.value, edges


def demo_section5_visualizations():
    """Run all visualization functions from example_flow_visualizations."""
    from example_flow_visualizations import (
        demo_network, draw_flow_utilization, draw_min_cut_partition,
        draw_feasible_polytope_2d, draw_duality_visualizations,
        draw_fairness_dashboard, animate_augmenting_paths,
        animate_alpha_sweep, draw_saddle_point_dual
    )

    G, pos = demo_network()
    flow_val, flow_dict = nx.maximum_flow(G, "s", "t", capacity="capacity")
    print(f"Demo network: max flow value = {flow_val}")
    print(f"Flow: { {e: flow_dict[e[0]][e[1]] for e in G.edges()} }")

    # 5a: Edge utilization
    print("\n[5a] Edge utilization visualization")
    draw_flow_utilization(G, flow_dict, pos)
    plt.show()

    # 5b: Min-cut partition
    print("\n[5b] Min-cut partition")
    fig2, _, S, T, cut_edges = draw_min_cut_partition(G, flow_dict, pos=pos)
    print(f"  S-side: {S}  |  T-side: {T}  |  Cut edges: {cut_edges}")
    plt.show()

    # 5c: Feasible polytope
    print("\n[5c] Feasible flow polytope (2D projection)")
    draw_feasible_polytope_2d()
    plt.show()

    # 5d: Duality gap + slackness heatmap
    print("\n[5d] Duality gap + complementary slackness heatmap")
    draw_duality_visualizations(G, flow_dict)
    plt.show()

    # 5e: Fairness dashboard
    print("\n[5e] Fairness dashboard (α ∈ {0, 0.5, 1, 2, 5})")
    results = {}
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        f_vals, edges = _solve_alpha_quick(G, "s", "t", alpha)
        source_mask = np.array([u == "s" for u, v in edges], dtype=bool)
        total = sum(f_vals[source_mask])
        j = _jain_index(f_vals[source_mask])
        results[alpha] = {"total": total, "flows": f_vals, "edges": edges, "jain": j}
    draw_fairness_dashboard(results)
    plt.show()

    # 5f: Lagrangian saddle point
    print("\n[5f] Lagrangian saddle point (3D + 2D contours)")
    draw_saddle_point_dual()
    plt.show()

    # 5g: Augmenting path animation
    print("\n[5g] Generating augmenting-path animation...")
    ani_aug = animate_augmenting_paths(G)
    print("  Animation created. Rendering as HTML...")
    html_aug = ani_aug.to_jshtml()
    plt.close("all")
    display(HTML(html_aug))

    # 5h: α-sweep animation
    print("\n[5h] Generating α-sweep animation (pre-computing 30 solves)...")
    ani_sweep = animate_alpha_sweep(G)
    print("  Animation created. Rendering...")
    html_sweep = ani_sweep.to_jshtml()
    plt.close("all")
    display(HTML(html_sweep))

    # 5i: Interactive slider
    print("\n[5i] Launching interactive α slider...")
    from example_flow_visualizations import interactive_alpha_slider
    print("  Drag the slider to explore the fairness-efficiency trade-off in real time.")
    # interactive_alpha_slider(G)
