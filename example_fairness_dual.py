"""
Example: Lagrangian Dual of Fairness-Constrained Flow
======================================================
Explores the dual problem of fairness-aware flow optimization.
When we change the objective from linear (max flow) to a concave
fairness utility, the dual structure changes in interesting ways.

Duality covered:
  1. Standard max flow: primal is max flow LP, dual is min cut LP.
     Strong duality: max flow = min cut capacity (integral).

  2. Fairness flow (log utility): primal is a convex program with
     concave objective. The Lagrangian dual reveals:
       - Dual variables λ_{u,v} ≥ 0: shadow prices of capacity constraints
       - Dual variables μ_u: potentials at nodes (conservation prices)
       - The dual function decomposes edge-wise.

  3. KKT conditions: at optimality,
       - If f_{u,v} < cap_{u,v} then λ_{u,v} = 0 (slack capacity → no price)
       - If λ_{u,v} > 0 then f_{u,v} = cap_{u,v} (positive price → saturated)
       - Conservation: flow in = flow out (μ_u arbitrage-free)

  4. Economic interpretation:
       - λ_{u,v} is the "congestion toll" on edge (u,v).
       - μ_u is the "node potential" — value of one unit of flow at u.
       - At optimum, tolls at bottlenecks exactly balance the marginal
         utility of pushing additional flow.

Educational goals:
  - Show that the dual of fair flow is NOT a simple min cut.
  - Understand KKT conditions as economic equilibrium.
  - Derive the dual analytically and verify numerically.
"""

import numpy as np
import cvxpy as cp
import networkx as nx


# ── 1. Build a simple network for clean duality exposition ──────────────────
def build_simple_network():
    """Minimal 4-node network for analytical duality."""
    G = nx.DiGraph()
    edges = [
        ("s", "a", 5), ("s", "b", 3),
        ("a", "t", 4), ("b", "t", 3),
        ("a", "b", 1),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    return G


# ── 2. Primal: α-fairness flow ──────────────────────────────────────────────
def solve_primal_fairness(G, source="s", sink="t", alpha=1.0):
    """
    Primal:
      max  ∑_{(u,v)} U_α(f_{u,v})
      s.t. 0 ≤ f_{u,v} ≤ c_{u,v}
           ∑_v f_{u,v} - ∑_v f_{v,u} = 0  for u ≠ s,t
    where U_α is the α-fairness utility.

    Returns (opt_value, f_opt, edges, nodes, A, cap).
    """
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
    else:
        if alpha < 1:
            objective = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
        else:
            objective = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))

    constraints = [
        f <= cap,
        A[[nodes.index(n) for n in nodes if n not in (source, sink)], :] @ f == 0,
    ]

    prob = cp.Problem(objective, constraints)
    opt_val = prob.solve()

    return opt_val, f.value, edges, nodes, A, cap


# ── 3. Construct the Lagrangian and dual function analytically ───────────────
def lagrangian_analysis(G, f_opt, edges, nodes, A, cap, source="s", sink="t"):
    """
    After solving the primal, extract dual information via KKT multipliers.

    The primal (log utility, α=1):
      max  ∑ log(f_e)
      s.t. f_e ≤ c_e      →  λ_e ≥ 0  (capacity shadow price)
           A_cons @ f = 0  →  μ_u       (conservation shadow price, unrestricted)

    Lagrangian:
      L(f, λ, μ) = ∑ log(f_e) - ∑ λ_e (f_e - c_e) - ∑ μ_u (A_cons @ f)_u

    KKT stationarity (∂L/∂f_e = 0):
      1/f_e - λ_e - ∑_u μ_u · A_cons[u,e] = 0
      →  f_e = 1 / (λ_e + ∑_u μ_u · A_cons[u,e])

    Complementary slackness:
      λ_e · (c_e - f_e) = 0,  λ_e ≥ 0

    Returns analysis as text and numerical verification.
    """
    print("=== Lagrangian Dual of Log-Fairness Flow ===\n")

    # Identify source edges, internal edges, sink edges
    source_edges = [(u, v) for u, v in edges if u == source]
    sink_edges = [(u, v) for u, v in edges if v == sink]
    internal_edges = [(u, v) for u, v in edges
                      if u != source and v != sink]

    print("Network structure:")
    print(f"  Source edges: {source_edges}")
    print(f"  Internal edges: {internal_edges}")
    print(f"  Sink edges: {sink_edges}")

    # --- Solve dual explicitly using CVXPY ---
    n_edges = len(edges)
    n_nodes = len(nodes)

    # Dual variables
    lam = cp.Variable(n_edges, nonneg=True)   # capacity multipliers λ_e ≥ 0
    mu = cp.Variable(n_nodes)                  # conservation multipliers μ_u (unrestricted)

    # Incidence matrix rows for conservation (non-terminal nodes)
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]
    A_cons = A[cons_idx, :]

    # Lagrangian evaluated at optimum of inner minimization:
    # For log utility: f_e*(λ,μ) = 1 / (λ_e + (A_cons^T μ)[e] - [source/sink adjustments])
    # Actually, we need to incorporate all terms properly.
    #
    # The full Lagrangian:
    # L(f, λ, μ) = Σ log(f_e) - Σ λ_e·f_e + Σ λ_e·c_e - μ^T A_cons f
    #
    # Maximizing L over f ≥ 0 gives:
    # ∂L/∂f_e = 1/f_e - λ_e - (A_cons^T μ)_e = 0  →  f_e = 1/(λ_e + (A_cons^T μ)_e)
    #
    # Then the dual function g(λ, μ) = max_{f ≥ 0} L(f, λ, μ)
    #   = Σ [log(1/(λ_e + a_e)) - 1] + λ^T c
    # where a_e = (A_cons^T μ)_e

    # Dual objective: minimize g(λ, μ)
    a = A_cons.T @ mu   # a_e for each edge
    denom = lam + a + 1e-12

    # g(λ, μ) = Σ [log(1/denom_e) - 1] + λ^T c
    #          = -Σ log(denom_e) - n_edges + λ^T c
    g_lambda_mu = -cp.sum(cp.log(denom)) - n_edges + lam @ cap

    dual_obj = cp.Minimize(g_lambda_mu)
    dual_prob = cp.Problem(dual_obj)
    dual_val = dual_prob.solve()

    lam_opt = lam.value
    mu_opt = mu.value
    a_opt = A_cons.T @ mu_opt

    # Recover primal from dual
    f_dual = 1.0 / (lam_opt + a_opt + 1e-12)

    print(f"\nPrimal optimal value:     {sum(np.log(np.maximum(f_opt, 1e-12))):.6f}")
    print(f"Dual optimal value:       {-dual_val:.6f} (negated since we minimized)")

    # Verify dual reconstruction
    print("\n--- Dual Reconstruction of Primal Flows ---")
    for j, (u, v) in enumerate(edges):
        print(f"  {u}→{v}: primal f={f_opt[j]:.4f}, dual-recovered f={f_dual[j]:.4f}, "
              f"λ={lam_opt[j]:.4f}, a={a_opt[j]:.4f}")

    # Complementary slackness check
    print("\n--- Complementary Slackness (λ_e · (c_e - f_e) ≈ 0) ---")
    for j, (u, v) in enumerate(edges):
        slack = lam_opt[j] * (cap[j] - f_opt[j])
        status = "✓" if abs(slack) < 1e-4 else "VIOLATION"
        print(f"  {u}→{v}: λ={lam_opt[j]:.4f}, "
              f"c-f={cap[j]-f_opt[j]:.4f}, product={slack:.6f} {status}")

    # Economic interpretation
    print("\n--- Economic Interpretation ---")
    print("λ_e  = shadow price of capacity on edge e (congestion toll)")
    print("μ_u  = potential value of one unit of flow at node u")
    print("a_e  = (A_cons^T μ)_e = μ_u - μ_v for edge e=(u,v) not incident to s,t")
    print()
    print("At optimality:")
    print("  • If an edge is NOT saturated (f_e < c_e), then λ_e = 0")
    print("    (no congestion → no toll)")
    print("  • If an edge IS saturated (f_e = c_e), then λ_e > 0")
    print("    (bottleneck → positive congestion price)")
    print("  • f_e = 1 / (λ_e + μ_u - μ_v)")
    print("    Flow is inversely proportional to effective cost λ_e + μ_u - μ_v")

    # Compare with standard max-flow dual (min cut)
    print("\n--- Comparison: Standard Max Flow Dual (Min Cut) ---")
    # The standard dual has μ_s = 1, μ_t = 0, and λ_e ∈ {0,1} at optimality
    print("In standard max flow, the dual has an integral optimal solution")
    print("with μ ∈ {0,1} (cut indicator) and λ ∈ {0,1} (cut edge indicator).")
    print("In fairness flow, the dual variables are continuous — there is")
    print("no 0-1 integrality, revealing a richer economic structure.")

    return lam_opt, mu_opt, f_dual


# ── 4. Sensitivity: How dual variables change with α ────────────────────────
def dual_sensitivity(G, source="s", sink="t"):
    """
    For the simple parallel-path network, trace how dual variables
    evolve as α changes from 0 (linear) through 1 (log) toward ∞.
    """
    print("\n\n=== Dual Sensitivity Across α Values ===\n")

    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        opt_val, f_opt, edges, nodes, A, cap = solve_primal_fairness(
            G, source, sink, alpha
        )
        total_flow = sum(f_opt[i] for i, (u, v) in enumerate(edges) if u == source)

        label = ("utilitarian" if alpha == 0 else
                 "proportional" if alpha == 1 else f"α={alpha}")

        # Compute approximate dual variables from KKT
        # For α-fairness: U'(f) = f^{-α}, so stationarity gives f_e = (λ_e + a_e)^{-1/α}
        # We can estimate λ from f_opt and capacity
        lam_est = np.zeros(len(edges))
        for j, (u, v) in enumerate(edges):
            if f_opt[j] >= cap[j] * 0.999:
                lam_est[j] = 1.0 / (f_opt[j] ** alpha) if f_opt[j] > 1e-9 else 1e6
            # else λ ≈ 0 by complementary slackness

        print(f"{label:>20s} (α={alpha}): total flow={total_flow:.4f}")
        print(f"  Estimated λ (capacity shadow prices):")
        for j, (u, v) in enumerate(edges):
            if lam_est[j] > 0.001:
                print(f"    {u}→{v}: λ≈{lam_est[j]:.4f} "
                      f"(f={f_opt[j]:.4f}, cap={cap[j]:.4f})")

    print("\nKey insight: As α increases, the marginal utility of flow")
    print("diminishes more rapidly (U'(f) = 1/f^α). This makes the optimizer")
    print("more willing to sacrifice total throughput for equal distribution.")
    print("In the dual, larger α means capacity constraints on popular edges")
    print("have lower shadow prices (less 'pain' from restricting them).")


# ── 5. Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G = build_simple_network()

    # Solve primal with log utility
    opt_val, f_opt, edges, nodes, A, cap = solve_primal_fairness(G, alpha=1.0)

    print("Primal solution (log utility, proportional fairness):")
    for j, (u, v) in enumerate(edges):
        print(f"  f({u}→{v}) = {f_opt[j]:.4f}  (capacity = {cap[j]:.0f})")

    # Run full Lagrangian analysis
    lam_opt, mu_opt, f_dual = lagrangian_analysis(G, f_opt, edges, nodes, A, cap)

    # Sensitivity across α
    dual_sensitivity(G)
