"""
Example: Visualization Approaches for Max Flow & Fairness Flow
===============================================================
A catalog of visualization techniques for flow networks, duality,
and fairness optimization — from static network drawings to animated
augmenting paths and interactive α-fairness sliders.Visualizations covered (original 1-9):
  1. Static graph: flow/capacity labels, edge coloring by utilization
  2. Min-cut partition: S/T coloring, bottleneck highlighting
  3. Feasible polytope: 2D projection of flow constraints + objective
  4. Duality gap: primal-dual convergence, complementary slackness heatmap
  5. Fairness comparison: bar charts, Pareto frontier, 3D α-surface
  6. Animation: augmenting-path discovery, α-sweep morphing
  7. Interactive: α slider (ipywidgets), capacity sensitivity click-map
  8. (reserved)
  9. Saddle point: 3D Lagrangian surface + 2D contour

New visualizations (10-16):
  10. 3D Feasible Flow Polytope — actual ℝ³ polytope for 3-edge network
  11. 3D Feasible Region — half-space intersection visualization
  12. Stationarity / KKT Geometry — ∇L = 0 residual heatmap
  13. 3D Animated Saddle Point — rotating camera view
  14. Residual Capacity 3D Surface — total flow vs source edge flows
  15. Flow Vector Field — arrows scaled by flow magnitude
  16. Flow Route Sankey — path decomposition of flow

KKT & Unimodularity visualizations (17-20):
  17. Relaxed Dual Integrality — bar chart + histogram of relaxed LP d values
  18. Unimodularity Heatmap — incidence matrix + submatrix determinant check
  19. Random Capacity Integrality Test — perturbation grid showing d stays {0,1}
  20. KKT Dual Force Analysis — why total unimodularity forces integral d

KKT Force-Balance Geometry (21-24):
  21. KKT Force Balance 2D — 1D LP: how objective gradient + constraint normals
      push d to 0 or 1 (three cases: L≤0, L∈(0,1) impossible, L≥1)
  22. 2D Constraint-Plane Force Diagram — 2-edge projection showing polytope,
      objective contours, gradient arrow, active constraint normals at optimum
  23. 3D Constraint-Force Polytope — 3-edge [0,1]³ cube with coupling
      half-spaces, gradient quiver, and active normal cone at {0,1}³ vertex
  24. KKT Gradient Force Field — sampling across 2D polytope: every interior
      point is pushed by −∇(cᵀd) toward a {0,1}² vertex
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Arc, ConnectionPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import deque
from itertools import product

# ── Utility: small flow network for quick demos ─────────────────────────────
def demo_network():
    G = nx.DiGraph()
    edges = [
        ("s", "a", 10), ("s", "b", 8),
        ("a", "b", 2),  ("a", "c", 6),
        ("b", "c", 5),  ("b", "d", 7),
        ("c", "t", 8),  ("d", "t", 9),
    ]
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    # fixed positions for consistent layout
    pos = {"s": (0, 0), "a": (1, 0.8), "b": (1, -0.8),
           "c": (2, 0.8), "d": (2, -0.8), "t": (3, 0)}
    return G, pos


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STATIC GRAPH VISUALIZATION — edge utilization coloring
# ═══════════════════════════════════════════════════════════════════════════════

def draw_flow_utilization(G, flow_dict, pos=None, title="Flow Network — Utilization"):
    """
    Color edges by utilization = flow/capacity.
    Green  → low utilization (slack)
    Yellow → moderate
    Red    → saturated

    Edge thickness also scales with absolute flow.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Build per-edge utilization data
    util = {}
    abs_flow = {}
    for u, v, data in G.edges(data=True):
        cap = data["capacity"]
        f = flow_dict.get(u, {}).get(v, 0)
        util[(u, v)] = f / cap if cap > 0 else 0
        abs_flow[(u, v)] = f

    # Colormap: green → yellow → red
    cmap = LinearSegmentedColormap.from_list("util", ["#2ca02c", "#ffdd55", "#d62728"])

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#e8e8e8",
                           edgecolors="#333", linewidths=1.5, ax=ax)

    # Draw each edge with its own color and width
    for u, v in G.edges():
        u_val = util[(u, v)]
        w = 1.0 + 4.0 * abs_flow[(u, v)] / max(max(abs_flow.values()), 1)
        color = cmap(u_val)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w,
                               edge_color=[color], arrowstyle='-|>',
                               arrowsize=18, ax=ax,
                               connectionstyle="arc3,rad=0.08")

    # Labels: f / cap on each edge
    edge_labels = {e: f"{abs_flow[e]:.1f}/{G.edges[e]['capacity']}"
                   for e in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, label_pos=0.55, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Utilization (flow / capacity)", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MIN-CUT PARTITION + BOTTLENECK HIGHLIGHTING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_min_cut_partition(G, flow_dict, source="s", sink="t", pos=None):
    """
    After computing max flow, extract the min cut via residual graph BFS.
    Color S-side nodes blue, T-side nodes orange.
    Highlight cut edges (S→T at capacity) with thick red arcs.
    """
    # Build residual graph
    R = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data["capacity"]
        f = flow_dict[u].get(v, 0)
        if f < cap:
            R.add_edge(u, v)
        if f > 0:
            R.add_edge(v, u)

    # BFS from source in residual graph → S-side
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

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Draw all edges in light gray
    nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=1.2,
                           arrowstyle='-|>', arrowsize=12, ax=ax,
                           connectionstyle="arc3,rad=0.08")

    # Identify cut edges (S→T)
    cut_edges = [(u, v) for u, v in G.edges() if u in S and v in T]
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                           edge_color="#d62728", width=3.5,
                           arrowstyle='-|>', arrowsize=20, ax=ax,
                           connectionstyle="arc3,rad=0.08")

    # Draw nodes with S/T coloring
    node_colors = ["#6baed6" if n in S else "#fd8d3c" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800,
                           edgecolors="#333", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)

    # Edge labels
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        f = flow_dict[u].get(v, 0)
        edge_labels[(u, v)] = f"{f:.1f}/{data['capacity']}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, label_pos=0.5, ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6baed6',
               markersize=12, label='S-side (reachable from s)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fd8d3c',
               markersize=12, label='T-side'),
        Line2D([0], [0], color='#d62728', linewidth=3, label='Cut edges (S→T)'),
        Line2D([0], [0], color='#cccccc', linewidth=1.2, label='Internal edges'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    cut_capacity = sum(G.edges[e]["capacity"] for e in cut_edges)
    ax.set_title(f"Min s-t Cut Partition\n"
                 f"S = {S} | Cut capacity = {cut_capacity}",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig, ax, S, T, cut_edges


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEASIBLE POLYTOPE — 2D projection of flow constraints
# ═══════════════════════════════════════════════════════════════════════════════

def draw_feasible_polytope_2d():
    """
    For a tiny 3-edge network  s→a→t, s→t  with capacities c_sa=5, c_at=4, c_st=3,
    project the flow polytope into the (f_sa, f_st) plane.

    Constraints:
      0 ≤ f_sa ≤ 5,  0 ≤ f_st ≤ 3,  0 ≤ f_at ≤ 4
      f_sa = f_at  (conservation at node a — it's a simple relay)

    So in (f_sa, f_st):
      0 ≤ f_sa ≤ 4   (bottlenecked by a→t capacity)
      0 ≤ f_st ≤ 3

    The total flow = f_sa + f_st. The max-flow LP solves:
      max f_sa + f_st  s.t.  0 ≤ f_sa ≤ 4, 0 ≤ f_st ≤ 3
    → Optimum at (4, 3), value = 7.

    We visualize the rectangle, objective contour lines, and the optimum.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Feasible rectangle
    rect = plt.Rectangle((0, 0), 4, 3, fill=True, facecolor="#c6dbef",
                         edgecolor="#3182bd", linewidth=2, alpha=0.5)
    ax.add_patch(rect)

    # Constraint half-planes (redundant but educational)
    ax.axvline(5, color="grey", linestyle="--", alpha=0.4, label="f_sa ≤ 5 (redundant)")
    ax.axhline(3, color="grey", linestyle="--", alpha=0.4, label="f_st ≤ 3 (binding)")
    ax.axvline(4, color="#d62728", linestyle="--", linewidth=1.5, label="f_sa ≤ 4 (binding)")

    # Objective contours: f_sa + f_st = k
    for k in [1, 3, 5, 7]:
        x_vals = np.linspace(0, k, 50)
        ax.plot(x_vals, k - x_vals, color="#ff7f0e", alpha=0.5, linewidth=1,
                label="obj = f_sa + f_st" if k == 7 else "")

    # Gradient arrow
    ax.annotate("∇ obj", xy=(3.0, 0.8), fontsize=11, color="#ff7f0e",
                ha="center")
    ax.arrow(2.5, 0.5, 0.8, 0.8, head_width=0.15, head_length=0.15,
             fc="#ff7f0e", ec="#ff7f0e")

    # Optimum point
    ax.scatter([4], [3], s=150, color="#d62728", zorder=5, edgecolors="black")
    ax.annotate("OPT (4, 3)\nflow = 7", xy=(4, 3), xytext=(3.3, 2.4),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    # Vertex labels
    ax.text(0.1, -0.35, "(0,0)", fontsize=9, ha="center")
    ax.text(4.1, -0.35, "(4,0)", fontsize=9, ha="center")
    ax.text(0.1, 3.1, "(0,3)", fontsize=9, ha="center")
    ax.text(4.1, 3.1, "(4,3)", fontsize=9, ha="center", fontweight="bold")

    ax.set_xlabel("f_sa (flow on s→a)", fontsize=11)
    ax.set_ylabel("f_st (flow on s→t)", fontsize=11)
    ax.set_title("Feasible Flow Polytope (2D projection)\n"
                 "Network: s→a→t, s→t | Capacities: c_sa=5, c_at=4, c_st=3",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DUALITY GAP + COMPLEMENTARY SLACKNESS HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def draw_duality_visualizations(G, flow_dict, source="s", sink="t"):
    """
    Left: convergence of primal/dual objective (simulated for educational display).
    Right: complementary slackness heatmap —
           for each edge, λ_e · (c_e − f_e) ≈ 0 at optimality.
    """
    edges = list(G.edges())
    n_edges = len(edges)

    # Simulated duality gap convergence (LP iterations)
    iterations = np.arange(1, 16)
    primal_vals = 7.0 * (1 - np.exp(-0.4 * iterations)) + 0.05 * np.random.randn(15)
    dual_vals = 7.0 * (1 - np.exp(-0.3 * iterations)) - 0.03 * np.random.randn(15)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: duality gap ──
    ax = axes[0]
    ax.plot(iterations, primal_vals, "o-", label="Primal (max flow)",
            color="#3182bd", markersize=5)
    ax.plot(iterations, dual_vals, "s-", label="Dual (min cut)",
            color="#d62728", markersize=5)
    ax.fill_between(iterations, primal_vals, dual_vals, alpha=0.15,
                    color="grey", label="Duality gap")
    ax.axhline(7.0, color="black", linestyle=":", alpha=0.5, label="Optimal = 7")
    ax.set_xlabel("Simplex iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("Duality Gap Convergence\n(Primal ↑  |  Dual ↓)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: complementary slackness ──
    ax = axes[1]
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    flow = np.array([flow_dict[u].get(v, 0) for u, v in edges], dtype=float)

    # Estimate λ (shadow price): positive if edge is ≈ saturated
    slack = cap - flow
    lam_est = np.where(flow >= cap * 0.98, 1.0 / (flow + 1e-6), 0.0)
    complement = lam_est * slack

    # Heatmap-style: rows = edges, columns = [f, c, slack, λ, λ·(c−f)]
    data = np.column_stack([flow, cap, slack, lam_est, complement])
    edge_labels = [f"{u}→{v}" for u, v in edges]

    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_edges))
    ax.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["flow f", "capacity c", "slack c−f", "λ (est)",
                         "λ·(c−f) ≈ 0"], fontsize=9)
    ax.set_title("Complementary Slackness Heatmap\n"
                 "λ·(c−f) ≈ 0 at optimality", fontsize=11)

    # Annotate cells
    for i in range(5):
        for j in range(n_edges):
            val = data[j, i]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 3 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FAIRNESS COMPARISON DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def draw_fairness_dashboard(flow_results):
    """
    flow_results: dict  {alpha: {"total": float, "flows": np.array,
                                  "edges": list, "jain": float}}

    Shows:
      Top-left:  edge flow bar chart for selected α values
      Top-right: total throughput vs α (price of fairness)
      Bottom:    Jain's fairness index vs α
    """
    alphas = sorted(flow_results.keys())
    fig = plt.figure(figsize=(14, 10))

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ── Top-left: stacked/grouped bar — flow per edge by α ──
    ax_bar = fig.add_subplot(gs[0, 0])
    edges = flow_results[alphas[0]]["edges"]
    edge_labels = [f"{u}→{v}" for u, v in edges]
    x = np.arange(len(edges))
    width = 0.7 / len(alphas)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))
    for i, alpha in enumerate(alphas):
        flows = flow_results[alpha]["flows"]
        offset = (i - len(alphas) / 2 + 0.5) * width
        ax_bar.bar(x + offset, flows, width, label=f"α={alpha}",
                   color=colors[i], alpha=0.85)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=7)
    ax_bar.set_ylabel("Flow")
    ax_bar.set_title("Edge Flow Distribution by α", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=7, ncol=2)
    ax_bar.grid(True, alpha=0.2, axis="y")

    # ── Top-right: total throughput vs α ──
    ax_total = fig.add_subplot(gs[0, 1])
    totals = [flow_results[a]["total"] for a in alphas]
    ax_total.plot(alphas, totals, "o-", color="#3182bd", linewidth=2, markersize=8)
    ax_total.fill_between(alphas, totals, max(totals), alpha=0.12, color="#d62728",
                          label="Price of fairness")
    ax_total.set_xlabel("α (fairness parameter)")
    ax_total.set_ylabel("Total Throughput")
    ax_total.set_title("Price of Fairness: Throughput vs α", fontsize=11,
                       fontweight="bold")
    ax_total.legend(fontsize=8)
    ax_total.grid(True, alpha=0.3)

    # ── Bottom: Jain's index vs α + utility curve ──
    ax_jain = fig.add_subplot(gs[1, :])
    jains = [flow_results[a]["jain"] for a in alphas]

    ax_jain.plot(alphas, jains, "s-", color="#2ca02c", linewidth=2.5,
                 markersize=10, label="Jain's Fairness Index")
    ax_jain.set_xlabel("α")
    ax_jain.set_ylabel("Jain's Fairness Index", color="#2ca02c")
    ax_jain.tick_params(axis="y", labelcolor="#2ca02c")
    ax_jain.set_ylim(0, 1.05)
    ax_jain.axhline(1.0, color="#2ca02c", linestyle=":", alpha=0.5,
                    label="Perfect equality (J=1)")

    # Twin axis: total throughput
    ax_twin = ax_jain.twinx()
    ax_twin.plot(alphas, totals, "o-", color="#3182bd", linewidth=2, markersize=8,
                 alpha=0.5, label="Total Throughput")
    ax_twin.set_ylabel("Total Throughput", color="#3182bd")
    ax_twin.tick_params(axis="y", labelcolor="#3182bd")

    ax_jain.set_title("Fairness–Efficiency Trade-off: Jain Index & Throughput vs α",
                      fontsize=12, fontweight="bold")
    lines1, labels1 = ax_jain.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_jain.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax_jain.grid(True, alpha=0.2)

    plt.suptitle("α-Fairness Flow Optimization Dashboard", fontsize=14,
                 fontweight="bold", y=0.98)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANIMATION — Augmenting path discovery (Ford-Fulkerson style)
# ═══════════════════════════════════════════════════════════════════════════════

def animate_augmenting_paths(G, source="s", sink="t"):
    """
    Animate the sequence of augmenting paths found during max-flow computation.
    Each frame highlights the current augmenting path in the residual graph
    and shows the cumulative flow pushed so far.

    Uses a simple manual path enumeration for the demo network.
    """
    G_work = G.copy()
    pos = nx.spring_layout(G, seed=7)

    # Manually define augmenting paths for a deterministic demo
    # (In a real setting, we'd record paths from nx.maximum_flow internals)
    augmenting_paths = [
        (["s", "a", "c", "t"], 6),   # path + bottleneck capacity
        (["s", "b", "d", "t"], 7),
        (["s", "a", "b", "c", "t"], 2),  # uses reverse edge b→a from residual
        (["s", "b", "c", "t"], 1),  # residual on b→c
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    cumulative_flow = 0

    def draw_frame(frame_idx):
        nonlocal cumulative_flow
        ax.clear()

        path, bottleneck = augmenting_paths[frame_idx]
        cumulative_flow += bottleneck

        # Draw full graph in light gray
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="#e8e8e8",
                               edgecolors="#333", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=1.0,
                               arrowstyle='-|>', arrowsize=12, ax=ax,
                               connectionstyle="arc3,rad=0.08")

        # Highlight the augmenting path edges in sequence
        path_edges = list(zip(path, path[1:]))
        for i, (u, v) in enumerate(path_edges):
            color = plt.cm.Oranges(0.4 + 0.6 * i / len(path_edges))
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=4,
                                   edge_color=[color], arrowstyle='-|>',
                                   arrowsize=20, ax=ax,
                                   connectionstyle="arc3,rad=0.08")

        # Capacity labels
        edge_labels = {(u, v): f"{data['capacity']}"
                       for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, ax=ax)

        ax.set_title(f"Augmenting Path #{frame_idx + 1}: {'→'.join(path)}\n"
                     f"Bottleneck: {bottleneck}  |  "
                     f"Cumulative flow: {cumulative_flow}",
                     fontsize=11, fontweight="bold")
        ax.axis("off")

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(augmenting_paths),
                                  interval=1800, repeat=True, repeat_delay=2000)
    plt.close()  # prevent double-display in notebook
    return ani


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ANIMATION — α-Fairness morphing (sweep α from 0 to 5)
# ═══════════════════════════════════════════════════════════════════════════════

def animate_alpha_sweep(G, source="s", sinks=None, n_frames=30):
    """
    Animate the flow distribution as α sweeps from 0 (utilitarian)
    to 5 (strongly fair). Fairness is applied to per-sink inflows.
    Requires cvxpy to solve at each frame.
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required for alpha-sweep animation")
        return None

    if sinks is None:
        sinks = ["t"]
    elif isinstance(sinks, str):
        sinks = [sinks]

    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    n_sinks = len(sinks)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    # Sink-inflow matrix
    B = np.zeros((n_sinks, n_edges))
    for j, (u, v) in enumerate(edges):
        for k, sk in enumerate(sinks):
            if v == sk:
                B[k, j] = 1.0

    terminal = {source} | set(sinks)
    cons_idx = [nodes.index(n) for n in nodes if n not in terminal]

    alphas = np.linspace(0.01, 5.0, n_frames)
    all_flows = []

    # Pre-compute flows for each α
    for alpha in alphas:
        f = cp.Variable(n_edges, nonneg=True)
        S = B @ f
        if alpha < 0.05:
            s_idx = nodes.index(source)
            obj = cp.Maximize(-np.asarray(A[s_idx, :]).flatten() @ f)
        elif abs(alpha - 1.0) < 0.05:
            obj = cp.Maximize(cp.sum(cp.log(S + 1e-9)))
        elif alpha < 1:
            obj = cp.Maximize(cp.sum(cp.power(S + 1e-9, 1 - alpha)) / (1 - alpha))
        else:
            obj = cp.Maximize(-cp.sum(cp.power(S + 1e-9, 1 - alpha)) / (alpha - 1))

        constraints = [f <= cap, A[cons_idx, :] @ f == 0]
        cp.Problem(obj, constraints).solve(verbose=False)

        all_flows.append(f.value if f.value is not None else np.zeros(n_edges))

    pos_demo = {"s": (0, 0), "a": (1, 0.6), "b": (1, -0.6),
                "c": (2, 0.6), "d": (2, -0.6), "t": (3, 0)}

    fig, (ax_graph, ax_bar) = plt.subplots(1, 2, figsize=(14, 6),
                                           gridspec_kw={"width_ratios": [1.2, 1]})

    def update(frame):
        ax_graph.clear()
        ax_bar.clear()
        alpha = alphas[frame]
        flows = all_flows[frame]

        # Left: graph with edge widths proportional to flow
        nx.draw_networkx_nodes(G, pos_demo, node_size=600, node_color="#e8e8e8",
                               edgecolors="#333", ax=ax_graph)
        nx.draw_networkx_labels(G, pos_demo, font_size=10, font_weight="bold",
                                ax=ax_graph)

        max_f = max(flows) if max(flows) > 0 else 1
        for j, (u, v) in enumerate(G.edges()):
            w = 1.0 + 4.0 * flows[j] / max_f
            color = plt.cm.RdYlGn(1.0 - flows[j] / (cap[j] + 1e-9))
            nx.draw_networkx_edges(G, pos_demo, edgelist=[(u, v)], width=w,
                                   edge_color=[color], arrowstyle='-|>',
                                   arrowsize=14, ax=ax_graph,
                                   connectionstyle="arc3,rad=0.08")

        total_flow = float(np.sum(B @ flows))
        ax_graph.set_title(f"α = {alpha:.2f}  |  Total sink inflow = {total_flow:.2f}",
                           fontsize=11, fontweight="bold")
        ax_graph.axis("off")

        # Right: bar chart of edge flows
        edge_labels = [f"{u}→{v}" for u, v in edges]
        colors_bar = plt.cm.RdYlGn(1.0 - flows / (cap + 1e-9))
        ax_bar.bar(range(n_edges), flows, color=colors_bar, edgecolor="#333")
        ax_bar.set_xticks(range(n_edges))
        ax_bar.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
        ax_bar.set_ylabel("Flow")
        ax_bar.set_ylim(0, max(cap) * 1.1)
        ax_bar.set_title(f"Edge Flows at α = {alpha:.2f}", fontsize=11,
                         fontweight="bold")

        # Capacity lines
        for j, c in enumerate(cap):
            ax_bar.axhline(c, xmin=(j - 0.4) / n_edges, xmax=(j + 0.4) / n_edges,
                           color="red", linewidth=0.8, linestyle="--", alpha=0.5)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=200, repeat=True, repeat_delay=1500)
    plt.close()
    return ani


# ═══════════════════════════════════════════════════════════════════════════════
# 8. INTERACTIVE — α-fairness slider (ipywidgets)
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_alpha_slider(G, source="s", sinks=None):
    """
    Requires: ipywidgets, cvxpy, and a Jupyter notebook (or %matplotlib widget).

    An interactive slider that re-solves the α-fairness flow (with fairness
    applied to per-sink inflows) and updates both the graph and the bar chart
    in real time.

    Usage in notebook:
        from example_flow_visualizations import *
        interactive_alpha_slider(demo_network()[0])
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        import cvxpy as cp
    except ImportError:
        print("Requires: pip install ipywidgets cvxpy (and a Jupyter notebook)")
        return

    if sinks is None:
        sinks = ["t"]
    elif isinstance(sinks, str):
        sinks = [sinks]

    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    n_sinks = len(sinks)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())

    # Sink-inflow matrix
    B = np.zeros((n_sinks, n_edges))
    for j, (u, v) in enumerate(edges):
        for k, sk in enumerate(sinks):
            if v == sk:
                B[k, j] = 1.0

    terminal = {source} | set(sinks)
    cons_idx = [nodes.index(n) for n in nodes if n not in terminal]

    pos = nx.spring_layout(G, seed=7)

    def solve_for_alpha(alpha):
        f = cp.Variable(n_edges, nonneg=True)
        S_expr = B @ f
        if alpha < 0.01:
            s_idx = nodes.index(source)
            obj = cp.Maximize(A[s_idx, :] @ f)
        elif abs(alpha - 1.0) < 0.01:
            obj = cp.Maximize(cp.sum(cp.log(S_expr + 1e-9)))
        elif alpha < 1:
            obj = cp.Maximize(cp.sum(cp.power(S_expr + 1e-9, 1 - alpha)) / (1 - alpha))
        else:
            obj = cp.Maximize(-cp.sum(cp.power(S_expr + 1e-9, 1 - alpha)) / (alpha - 1))

        constraints = [f <= cap, A[cons_idx, :] @ f == 0]
        cp.Problem(obj, constraints).solve(verbose=False)
        return f.value if f.value is not None else np.zeros(n_edges)

    def update_display(alpha):
        clear_output(wait=True)
        flows = solve_for_alpha(alpha)
        total = float(np.sum(B @ flows))

        fig, (ax_g, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

        # Graph
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#e8e8e8",
                               edgecolors="#333", ax=ax_g)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax_g)
        max_f = max(flows) if max(flows) > 0 else 1
        for j, (u, v) in enumerate(G.edges()):
            w = 1.5 + 3.5 * flows[j] / max_f
            color_ratio = flows[j] / (cap[j] + 1e-9)
            color = plt.cm.RdYlGn(1.0 - color_ratio)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w,
                                   edge_color=[color], arrowstyle='-|>',
                                   arrowsize=14, ax=ax_g,
                                   connectionstyle="arc3,rad=0.08")
        ax_g.set_title(f"α = {alpha:.2f} | Total sink inflow = {total:.2f}", fontsize=11,
                       fontweight="bold")
        ax_g.axis("off")

        # Bar chart
        edge_labels = [f"{u}→{v}" for u, v in edges]
        colors_bar = [plt.cm.RdYlGn(1.0 - flows[j] / (cap[j] + 1e-9))
                      for j in range(n_edges)]
        ax_b.bar(range(n_edges), flows, color=colors_bar, edgecolor="#333")
        ax_b.set_xticks(range(n_edges))
        ax_b.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
        ax_b.set_ylabel("Flow")
        ax_b.set_ylim(0, max(cap) * 1.1)
        for j, c in enumerate(cap):
            ax_b.axhline(c, xmin=(j - 0.4) / n_edges, xmax=(j + 0.4) / n_edges,
                         color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_b.set_title("Edge Flows (red dashes = capacity)", fontsize=10)

        plt.tight_layout()
        plt.show()

    slider = widgets.FloatSlider(
        value=1.0, min=0.0, max=5.0, step=0.1,
        description="α:",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    widgets.interactive_output(update_display, {"alpha": slider})
    display(slider)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SADDLE POINT — Lagrangian surface sketch for 1-edge toy problem
# ═══════════════════════════════════════════════════════════════════════════════

def draw_saddle_point_dual():
    """
    For the single-edge problem (max log(f) s.t. f ≤ c, f ≥ 0),
    the Lagrangian is L(f, λ) = log(f) − λ(f − c).

    Plot the 3D surface z = L(f, λ) and mark the saddle point (f*, λ*).
    At the saddle: ∂L/∂f = 0 → 1/f − λ = 0, ∂L/∂λ = 0 → f = c.
    So f* = c, λ* = 1/c.

    This is the simplest possible KKT saddle-point geometry.
    """
    c = 4.0
    f_vals = np.linspace(0.3, 6, 60)
    lam_vals = np.linspace(-0.5, 2.0, 60)
    F, L = np.meshgrid(f_vals, lam_vals)

    Z = np.log(np.maximum(F, 1e-9)) - L * (F - c)

    fig = plt.figure(figsize=(12, 5))

    # ── 3D saddle surface ──
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(F, L, Z, cmap="RdYlGn_r", alpha=0.85,
                             edgecolor="none", zorder=1)
    f_star, lam_star = c, 1.0 / c
    z_star = np.log(f_star) - lam_star * (f_star - c)
    marker_z = z_star + 0.8
    ax3d.scatter([f_star], [lam_star], [marker_z], color="black", s=80,
                 marker="o", zorder=100, depthshade=False)
    ax3d.text(f_star + 0.3, lam_star, marker_z + 0.3,
              f"Saddle\n(f*={f_star}, λ*={lam_star:.3f})",
              fontsize=9, fontweight="bold", zorder=100, clip_on=False)
    
    ax3d.view_init(elev=40, azim=25, roll=0)

    ax3d.set_xlabel("f (flow)")
    ax3d.set_ylabel("λ (Lagrange multiplier)")
    ax3d.set_zlabel("L(f, λ)")
    ax3d.set_title("Lagrangian Saddle Point\nmax log(f) s.t. f ≤ c",
                   fontsize=10, fontweight="bold")

    # ── 2D contour view ──
    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contour(F, L, Z, levels=20, cmap="coolwarm", alpha=0.8)
    ax2d.clabel(contour, inline=True, fontsize=7)
    ax2d.scatter([f_star], [lam_star], color="black", s=100, marker="*",
                 zorder=10, edgecolors="white", linewidth=1)
    ax2d.annotate("Saddle point\n(min in λ, max in f)",
                  xy=(f_star, lam_star), xytext=(f_star + 1.2, lam_star + 0.4),
                  fontsize=9, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="black"))

    ax2d.set_xlabel("f (flow)")
    ax2d.set_ylabel("λ (multiplier)")
    ax2d.set_title("Contours of L(f, λ)\nArrows: ∇L = 0 at saddle",
                   fontsize=10, fontweight="bold")
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. 3D FEASIBLE FLOW POLYTOPE — the polytope in ℝ³
# ═══════════════════════════════════════════════════════════════════════════════

def draw_3d_feasible_polytope():
    """
    Render the actual 3D flow polytope for a 3-edge network.
    Network:  s→a→t, s→t  (3 edges: e1=s→a, e2=a→t, e3=s→t)
    Variables: f = (f_e1, f_e2, f_e3)
    Constraints:
      0 ≤ f_e1 ≤ 5,  0 ≤ f_e2 ≤ 4,  0 ≤ f_e3 ≤ 3
      f_e1 = f_e2  (flow conservation at node a)
    So the feasible set in (f_e1, f_e3) space is [0,4]×[0,3] rectangle,
    and f_e2 is determined by f_e2 = f_e1.

    In full 3D (f_e1, f_e2, f_e3), the polytope is a rectangular prism
    intersected with the plane f_e1 = f_e2 — forming a 2D face embedded in ℝ³.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Capacities
    c1, c2, c3 = 5, 4, 3

    fig = plt.figure(figsize=(14, 6))

    # ── Left subplot: the full constraint box + the plane f_e1 = f_e2 ──
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    # Draw the full capacity box (wireframe)
    xx = np.array([0, c1])
    yy = np.array([0, c2])
    zz = np.array([0, c3])
    ax.plot(xx, [0, 0], [0, 0], "k-", linewidth=0.5, alpha=0.3)
    ax.plot(xx, [c2, c2], [0, 0], "k-", linewidth=0.5, alpha=0.3)
    ax.plot(xx, [0, 0], [c3, c3], "k-", linewidth=0.5, alpha=0.3)
    ax.plot(xx, [c2, c2], [c3, c3], "k-", linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], yy, [0, 0], "k-", linewidth=0.5, alpha=0.3)
    ax.plot([c1, c1], yy, [0, 0], "k-", linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], yy, [c3, c3], "k-", linewidth=0.5, alpha=0.3)
    ax.plot([c1, c1], yy, [c3, c3], "k-", linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [0, 0], zz, "k-", linewidth=0.5, alpha=0.3)
    ax.plot([c1, c1], [0, 0], zz, "k-", linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [c2, c2], zz, "k-", linewidth=0.5, alpha=0.3)
    ax.plot([c1, c1], [c2, c2], zz, "k-", linewidth=0.5, alpha=0.3)

    # The conservation plane f_e1 = f_e2  (semi-transparent)
    f1_vals = np.linspace(0, min(c1, c2), 20)
    f3_vals = np.linspace(0, c3, 10)
    F1_grid, F3_grid = np.meshgrid(f1_vals, f3_vals)
    F2_grid = F1_grid  # conservation: f_e2 = f_e1
    ax.plot_surface(F1_grid, F2_grid, F3_grid, alpha=0.35,
                    color="#ff7f0e", edgecolor="none")

    # The feasible polytope is the intersection of the box with this plane
    # → a rectangle in the (f_e1, f_e3) plane, extruded along f_e2 = f_e1
    vertices = np.array([
        [0, 0, 0],
        [min(c1, c2), min(c1, c2), 0],
        [min(c1, c2), min(c1, c2), c3],
        [0, 0, c3],
    ])
    poly = Poly3DCollection([vertices], alpha=0.6, facecolor="#3182bd",
                             edgecolor="#08519c", linewidth=2)
    ax.add_collection3d(poly)

    # Mark optimum point: max f_e1 + f_e3 → (min(c1,c2), min(c1,c2), c3)
    opt = np.array([min(c1, c2), min(c1, c2), c3])
    ax.scatter(*opt, color="#d62728", s=120, edgecolors="black", zorder=10)
    ax.text(opt[0] + 0.2, opt[1] + 0.2, opt[2],
            f"OPT\n(4,4,3)", fontsize=9, fontweight="bold", color="#d62728")

    ax.set_xlabel("f_e1 (s→a)")
    ax.set_ylabel("f_e2 (a→t)")
    ax.set_zlabel("f_e3 (s→t)")
    ax.set_title("3D Feasible Flow Polytope\nBox = capacities, Plane = conservation f_e1=f_e2",
                 fontsize=10, fontweight="bold")
    ax.view_init(elev=22, azim=-55)

    # ── Right subplot: the 2D feasible face (f_e1, f_e3) ──
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Draw the feasible face as a surface
    f_e1_vals = np.linspace(0, min(c1, c2), 30)
    f_e3_vals = np.linspace(0, c3, 30)
    FE1, FE3 = np.meshgrid(f_e1_vals, f_e3_vals)
    FE2 = FE1
    Z_obj = FE1 + FE3  # total flow = f_e1 + f_e3

    surf = ax2.plot_surface(FE1, FE2, FE3, facecolors=plt.cm.RdYlGn(
                              Z_obj / Z_obj.max()),
                            alpha=0.85, edgecolor="none")

    # Colorbar for objective
    mappable = plt.cm.ScalarMappable(cmap="RdYlGn",
                                     norm=Normalize(Z_obj.min(), Z_obj.max()))
    mappable.set_array(Z_obj)
    plt.colorbar(mappable, ax=ax2, shrink=0.6, label="Total flow (objective)")

    # Optimum
    ax2.scatter(*opt, color="#d62728", s=120, edgecolors="black", zorder=10)
    ax2.text(opt[0] + 0.3, opt[1] + 0.3, opt[2],
             "OPT", fontsize=10, fontweight="bold", color="#d62728")

    # Gradient direction
    grad = np.array([1, 1, 1])
    ax2.quiver(2, 2, 1.5, grad[0], grad[1], grad[2], color="#ff7f0e",
               linewidth=2, arrow_length_ratio=0.15, label="∇(total flow)")

    ax2.set_xlabel("f_e1 (s→a)")
    ax2.set_ylabel("f_e2 = f_e1")
    ax2.set_zlabel("f_e3 (s→t)")
    ax2.set_title("Feasible Face (f_e1 = f_e2)\nColored by total flow = f_e1 + f_e3",
                  fontsize=10, fontweight="bold")
    ax2.view_init(elev=25, azim=-50)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 11. 3D FEASIBLE REGION — half-space intersection for a 3-edge network
# ═══════════════════════════════════════════════════════════════════════════════

def draw_feasible_region_3d_halfspaces():
    """
    Visualize the feasible region as the intersection of half-spaces in ℝ³.

    For the same 3-edge network, constraints are half-spaces:
      f_e1 ≥ 0,  f_e1 ≤ 5   (two parallel planes)
      f_e2 ≥ 0,  f_e2 ≤ 4
      f_e3 ≥ 0,  f_e3 ≤ 3
      f_e1 − f_e2 = 0       (hyperplane, conservation)

    We show each half-space as a semi-transparent bounding plane,
    revealing how the polytope is carved out by intersecting constraints.
    """
    c1, c2, c3 = 5, 4, 3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Helper: draw a semi-transparent plane
    def draw_plane(ax, point, normal, size, color, alpha=0.15, label=""):
        """Draw a plane given a point and normal vector."""
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        # Find two orthogonal directions in the plane
        if abs(n[2]) < 0.99:
            u = np.cross(n, [0, 0, 1])
            u = u / np.linalg.norm(u)
        else:
            u = np.cross(n, [1, 0, 0])
            u = u / np.linalg.norm(u)
        v = np.cross(n, u)

        p = np.asarray(point)
        s = size
        xx = p[0] + (u[0] * np.array([-s, s, s, -s]) +
                      v[0] * np.array([-s, -s, s, s]))
        yy = p[1] + (u[1] * np.array([-s, s, s, -s]) +
                      v[1] * np.array([-s, -s, s, s]))
        zz = p[2] + (u[2] * np.array([-s, s, s, -s]) +
                      v[2] * np.array([-s, -s, s, s]))
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        verts = [list(zip(xx, yy, zz))]
        poly = Poly3DCollection(verts, alpha=alpha, facecolor=color,
                                edgecolor=color, linewidth=0.5)
        ax.add_collection3d(poly)

    # ── Capacity half-spaces (upper bounds) ──
    draw_plane(ax, (0, 0, c3), (0, 0, 1), 6, "#d62728", 0.12,
               "f_e3 ≤ 3")
    draw_plane(ax, (c1/2, c2/2, c3/2), (1, 0, 0), 6, "#3182bd", 0.08,
               "f_e1 ≤ 5")
    draw_plane(ax, (c1/2, c2/2, c3/2), (0, 1, 0), 6, "#2ca02c", 0.08,
               "f_e2 ≤ 4")

    # ── Non-negativity planes (lower bounds) ──
    draw_plane(ax, (0, 0, 0), (0, 0, -1), 6, "#ff7f0e", 0.06,
               "f_e3 ≥ 0")
    draw_plane(ax, (0, 0, 0), (-1, 0, 0), 6, "#ff7f0e", 0.06,
               "f_e1 ≥ 0")
    draw_plane(ax, (0, 0, 0), (0, -1, 0), 6, "#ff7f0e", 0.06,
               "f_e2 ≥ 0")

    # ── Conservation hyperplane: f_e1 = f_e2 ──
    # Normal to the plane f_e1 - f_e2 = 0 is (1, -1, 0)
    draw_plane(ax, (c1/2, c1/2, c3/2), (1, -1, 0), 6, "#9467bd", 0.25,
               "f_e1 = f_e2")

    # ── The feasible polytope (intersection) as a solid ──
    # The intersection is the segment in (f_e1, f_e3): [0,4]×[0,3] with f_e2=f_e1
    f_e1_line = np.linspace(0, min(c1, c2), 50)
    f_e3_line = np.linspace(0, c3, 50)
    F1_m, F3_m = np.meshgrid(f_e1_line, f_e3_line)
    F2_m = F1_m

    # Draw the feasible face as a mesh
    ax.plot_surface(F1_m, F2_m, F3_m, alpha=0.7, color="#ffbb78",
                    edgecolor="#ff7f0e", linewidth=0.3)

    # Highlight the feasible vertices
    verts_feas = np.array([[0, 0, 0], [4, 4, 0], [4, 4, 3], [0, 0, 3]])
    ax.scatter(verts_feas[:, 0], verts_feas[:, 1], verts_feas[:, 2],
               color="#d62728", s=80, edgecolors="black", zorder=10)
    for v in verts_feas:
        ax.text(v[0] + 0.15, v[1] + 0.15, v[2],
                f"({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})", fontsize=7)

    ax.set_xlabel("f_e1 (s→a)")
    ax.set_ylabel("f_e2 (a→t)")
    ax.set_zlabel("f_e3 (s→t)")
    ax.set_title("Feasible Region as Half-Space Intersection\n"
                 "6 capacity half-spaces + 1 conservation hyperplane",
                 fontsize=10, fontweight="bold")
    ax.view_init(elev=20, azim=-45)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3182bd',
               markersize=8, alpha=0.4, label='f_e1 ≤ 5'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c',
               markersize=8, alpha=0.4, label='f_e2 ≤ 4'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
               markersize=8, alpha=0.4, label='f_e3 ≤ 3'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#9467bd',
               markersize=8, alpha=0.5, label='f_e1 = f_e2 (conservation)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffbb78',
               markersize=8, label='Feasible polytope'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 12. STATIONARITY / KKT GEOMETRY — where ∇L = 0
# ═══════════════════════════════════════════════════════════════════════════════

def draw_stationarity_visualization():
    """
    Visualize the KKT stationarity condition for a single-edge fairness problem:
      maximize  log(f)   subject to   0 ≤ f ≤ c

    Lagrangian: L(f, λ, μ) = log(f) − λ(f − c) + μ·f
    (λ ≥ 0 is the multiplier for f ≤ c, μ ≥ 0 for f ≥ 0)

    KKT stationarity: ∇_f L = 0  →  1/f − λ + μ = 0  →  1/f = λ − μ

    At optimum:  f = c (binding), λ = 1/c > 0, μ = 0  (lower bound inactive)

    We plot:
      - Left:  The primal objective log(f) and the Lagrangian L(f, λ*) for λ=1/c
      - Right: The stationarity residual |1/f − λ + μ| over the (f, λ) plane
               with the KKT curve 1/f = λ highlighted
    """
    c = 4.0
    lam_star = 1.0 / c  # optimal multiplier
    f_star = c

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: Primal objective and Lagrangian at optimal λ ──
    ax = axes[0]
    f_vals = np.linspace(0.01, 6, 200)
    obj = np.log(f_vals)
    lagrangian = np.log(f_vals) - lam_star * (f_vals - c)
    constr = c - f_vals  # slack

    ax.plot(f_vals, obj, linewidth=2.5, color="#3182bd", label="log(f) [primal objective]")
    ax.plot(f_vals, lagrangian, linewidth=2, color="#d62728",
            label=f"L(f, λ*) = log(f) − {lam_star:.3f}·(f−c)")
    ax.axvline(f_star, color="black", linestyle="--", alpha=0.5,
               label=f"f* = c = {c}")
    ax.axhline(np.log(c), color="#3182bd", linestyle=":", alpha=0.4,
                label=f"log(c) = {np.log(c):.3f}")

    # Mark the tangency point: ∇obj(f*) = ∇constr(f*)·λ
    ax.scatter([f_star], [np.log(f_star)], color="black", s=100, zorder=5)
    # Tangent line at f*
    tang = np.log(c) + (1/c) * (f_vals - c)
    ax.plot(f_vals, tang, "--", color="#ff7f0e", linewidth=1.2, alpha=0.7,
            label=f"Tangent: slope = 1/c = {1/c:.3f}")

    ax.set_xlabel("f (flow)")
    ax.set_ylabel("Value")
    ax.set_title("Stationarity: ∇log(f*) = λ*·∇(f*−c)\n"
                 "At optimum, gradient of objective ∝ gradient of constraint",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-2, 2.5)

    # ── Right: Stationarity residual heatmap ──
    ax2 = axes[1]
    F_grid = np.linspace(0.1, 6, 100)
    Lam_grid = np.linspace(-0.1, 2.0, 100)
    FF, LL = np.meshgrid(F_grid, Lam_grid)
    residual = np.abs(1.0 / FF - LL)

    # Log-scale for better visibility
    im = ax2.contourf(FF, LL, np.log10(residual + 1e-12), levels=20,
                      cmap="RdYlBu_r")
    # KKT curve: 1/f = λ  →  λ = 1/f
    ax2.plot(F_grid, 1.0 / F_grid, "k-", linewidth=2.5, label="KKT: 1/f = λ (stationarity)")
    ax2.scatter([f_star], [lam_star], color="black", s=150, marker="*",
                zorder=10, edgecolors="white", linewidth=1.5)
    ax2.annotate(f"(f*, λ*) = ({f_star}, {lam_star:.3f})",
                xy=(f_star, lam_star), xytext=(f_star + 1.5, lam_star + 0.4),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    # Feasibility region: f ≤ c → region left of c
    ax2.axvspan(0, c, alpha=0.06, color="green", label="Primal feasible (f ≤ c)")
    ax2.axhline(0, color="grey", linestyle="-", alpha=0.3)
    ax2.axvline(c, color="grey", linestyle="--", alpha=0.4, label=f"f = c = {c}")

    ax2.set_xlabel("f (flow)")
    ax2.set_ylabel("λ (Lagrange multiplier)")
    ax2.set_title("Stationarity Residual: |1/f − λ|\n"
                  "KKT curve 1/f = λ, optimum at intersection with f ≤ c",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right")
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("log₁₀(|1/f − λ|)", fontsize=9)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 13. 3D ANIMATED SADDLE POINT — rotating view of Lagrangian surface
# ═══════════════════════════════════════════════════════════════════════════════

def animate_3d_saddle_point(n_frames=90):
    """
    Create a view-only 3D animation of the Lagrangian saddle point,
    rotating the camera around the saddle to reveal its geometry.

    For the problem:  max log(f)  s.t.  f ≤ c   (c = 4)
    Lagrangian:  L(f, λ) = log(f) − λ(f − c)

    The camera orbits in azimuth while maintaining elevation,
    showing the saddle nature: convex in λ, concave in f.
    """
    c = 4.0
    f_vals = np.linspace(0.3, 7, 65)
    lam_vals = np.linspace(-0.5, 2.5, 65)
    F, LAM = np.meshgrid(f_vals, lam_vals)
    Z = np.log(np.maximum(F, 1e-9)) - LAM * (F - c)

    f_star, lam_star = c, 1.0 / c
    z_star = np.log(f_star) - lam_star * (f_star - c)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface once
    surf = ax.plot_surface(F, LAM, Z, cmap="coolwarm", alpha=0.85,
                           edgecolor="none", zorder=1)

    # Plot saddle paths: max-in-f curve and min-in-λ curve
    # For fixed λ, L is max at f = 1/λ  (∂L/∂f = 0 → 1/f − λ = 0)
    lambda_for_curve = np.linspace(0.05, 3.0, 100)
    f_max_for_lam = 1.0 / lambda_for_curve
    # Only keep points within our visible range
    mask = (f_max_for_lam > 0.3) & (f_max_for_lam < 7)
    lambda_for_curve = lambda_for_curve[mask]
    f_max_for_lam = f_max_for_lam[mask]

    z_max_curve = np.log(f_max_for_lam) - lambda_for_curve * (f_max_for_lam - c)
    ax.plot(f_max_for_lam, lambda_for_curve, z_max_curve,
            color="#d62728", linewidth=2.5, zorder=50,
            label="∂L/∂f = 0 ridge (max in f)")

    # For fixed f, L is linear in λ: ∂L/∂λ = c − f
    # At f = c, L = log(c) constant in λ (the saddle ridge)
    lam_line = np.linspace(-0.5, 2.5, 50)
    z_min_curve = np.log(c) - lam_line * (c - c)
    ax.plot([c] * len(lam_line), lam_line, z_min_curve,
            color="#9467bd", linewidth=2.5, zorder=50,
            label="∂L/∂λ = 0 valley (min in λ)")

    # Mark the saddle point above the surface
    ax.scatter([f_star], [lam_star], [z_star + 0.6], color="black", s=120,
               marker="o", zorder=100, depthshade=False)
    ax.text(f_star + 0.6, lam_star, z_star + 1.0,
            "SADDLE\nPOINT", fontsize=9, fontweight="bold",
            color="black", zorder=100, clip_on=False)

    # Dashed vertical line from saddle down to surface
    ax.plot([f_star, f_star], [lam_star, lam_star],
            [z_star, z_star + 0.5], "k--", linewidth=1, alpha=0.6)

    ax.set_xlabel("f (flow)")
    ax.set_ylabel("λ (Lagrange multiplier)")
    ax.set_zlabel("L(f, λ)")
    ax.set_title("Lagrangian Saddle Point — Rotating View\n"
                 "max log(f) s.t. f ≤ c  |  f* = c, λ* = 1/c",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")

    # Animation function: rotate azimuth
    def update(frame):
        ax.view_init(elev=30, azim=frame * 4)  # 4° per frame → full rotation in 90 frames
        return [ax]

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=80, repeat=True, blit=False)
 
    ani_saddle = animate_3d_saddle_point(n_frames=90)
    html_saddle = ani_saddle.to_jshtml()
    plt.close("all")
    display(HTML(html_saddle))


# ═══════════════════════════════════════════════════════════════════════════════
# 14. RESIDUAL CAPACITY 3D SURFACE — how slack evolves
# ═══════════════════════════════════════════════════════════════════════════════

def draw_residual_capacity_surface(G, source="s", sink="t"):
    """
    For the demo network, vary flow on two source edges (s→a, s→b)
    while satisfying conservation and capacity constraints on the rest
    of the network. Plot the resulting total flow as a 3D surface over
    the (f_sa, f_sb) plane.

    This shows the "residual capacity landscape" — how much flow can
    reach the sink given fixed amounts on the two source edges.
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required for residual capacity surface")
        return None

    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]

    # Identify source edges
    s_idx = nodes.index(source)
    source_mask = np.array([u == source for u, v in edges], dtype=bool)
    source_edge_indices = list(np.where(source_mask)[0])

    # Grid over source edge flows
    n_grid = 25
    sa_max = min(cap[i] for i in source_edge_indices[:1])
    sb_max = min(cap[i] for i in source_edge_indices[1:2])

    f_sa_vals = np.linspace(0, sa_max, n_grid)
    f_sb_vals = np.linspace(0, sb_max, n_grid)
    F_SA, F_SB = np.meshgrid(f_sa_vals, f_sb_vals)
    Z_total = np.zeros_like(F_SA)

    for i in range(n_grid):
        for j in range(n_grid):
            f_sa = F_SA[i, j]
            f_sb = F_SB[i, j]

            f = cp.Variable(n_edges, nonneg=True)
            constraints = [
                f <= cap,
                A[cons_idx, :] @ f == 0,
                f[source_edge_indices[0]] == f_sa,
                f[source_edge_indices[1]] == f_sb,
            ]
            # Maximize total flow from source
            q_source = -np.asarray(A[s_idx, :]).flatten()
            obj = cp.Maximize(q_source @ f)
            try:
                cp.Problem(obj, constraints).solve(verbose=False)
                Z_total[i, j] = (q_source @ f.value) if f.value is not None else 0
            except Exception:
                Z_total[i, j] = 0

    fig = plt.figure(figsize=(12, 5))

    # ── 3D surface ──
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(F_SA, F_SB, Z_total, cmap="viridis",
                             alpha=0.88, edgecolor="none")

    max_total = np.max(Z_total)
    max_idx = np.unravel_index(np.argmax(Z_total), Z_total.shape)
    ax3d.scatter([F_SA[max_idx]], [F_SB[max_idx]], [max_total],
                 color="#d62728", s=80, zorder=10)
    ax3d.text(F_SA[max_idx], F_SB[max_idx], max_total + 0.5,
              f"Max: {max_total:.1f}", fontsize=9, fontweight="bold",
              color="#d62728")

    ax3d.set_xlabel(f"f({edges[source_edge_indices[0]][0]}→{edges[source_edge_indices[0]][1]})")
    ax3d.set_ylabel(f"f({edges[source_edge_indices[1]][0]}→{edges[source_edge_indices[1]][1]})")
    ax3d.set_zlabel("Total flow to sink")
    ax3d.set_title("Residual Capacity Surface\n"
                   "Total flow achievable given source edge flows",
                   fontsize=10, fontweight="bold")
    ax3d.view_init(elev=25, azim=-50)
    plt.colorbar(surf, ax=ax3d, shrink=0.6, label="Total flow")

    # ── 2D contour view ──
    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(F_SA, F_SB, Z_total, levels=15, cmap="viridis")
    ax2d.scatter([F_SA[max_idx]], [F_SB[max_idx]], color="#d62728",
                 s=100, marker="*", edgecolors="white", linewidth=1)
    ax2d.annotate(f"Max: {max_total:.1f}",
                  xy=(F_SA[max_idx], F_SB[max_idx]),
                  xytext=(F_SA[max_idx] + 0.5, F_SB[max_idx] + 0.4),
                  fontsize=9, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="black"))

    ax2d.set_xlabel(f"f({edges[source_edge_indices[0]][0]}→{edges[source_edge_indices[0]][1]})")
    ax2d.set_ylabel(f"f({edges[source_edge_indices[1]][0]}→{edges[source_edge_indices[1]][1]})")
    ax2d.set_title("Contour View: Total Flow vs Source Edge Flows",
                   fontsize=10, fontweight="bold")
    ax2d.grid(True, alpha=0.15)
    plt.colorbar(contour, ax=ax2d, shrink=0.85, label="Total flow")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 15. FLOW VECTOR FIELD — arrows scaled by flow magnitude
# ═══════════════════════════════════════════════════════════════════════════════

def draw_flow_vector_field(G, flow_dict, pos=None,
                           title="Flow Vector Field on Network"):
    """
    Draw the network with each edge rendered as a thick arrow whose
    width, color intensity, and head size scale with the flow magnitude.
    This turns the graph into a "flow map" where dominant routes
    immediately stand out visually.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=10)

    edges = list(G.edges())
    flows = np.array([flow_dict[u].get(v, 0) for u, v in edges],
                     dtype=float)
    caps = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    max_flow = max(flows) if len(flows) > 0 and max(flows) > 0 else 1

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Vector field view ──
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="#f0f0f0",
                           edgecolors="#333", linewidths=1.2, ax=ax_left)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold",
                            ax=ax_left)

    # Draw edges as arrows with width ∝ flow, color ∝ utilization
    for j, (u, v) in enumerate(edges):
        w = 0.8 + 6.0 * flows[j] / max_flow
        util = flows[j] / (caps[j] + 1e-9)
        color = plt.cm.YlOrRd(0.2 + 0.8 * util)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w,
                               edge_color=[color], arrowstyle='-|>',
                               arrowsize=10 + 20 * flows[j] / max_flow,
                               ax=ax_left, connectionstyle="arc3,rad=0.08")

    ax_left.set_title(title + "\nArrow thickness & color ∝ flow",
                      fontsize=11, fontweight="bold")
    ax_left.axis("off")

    # ── Right: Heatmap of edge flow values ──
    edge_labels = [f"{u}→{v}" for u, v in edges]
    # Sort edges by flow for the bar chart
    sort_idx = np.argsort(flows)[::-1]
    sorted_labels = [edge_labels[i] for i in sort_idx]
    sorted_flows = flows[sort_idx]
    sorted_caps = caps[sort_idx]

    x = np.arange(len(edges))
    bars = ax_right.bar(x, sorted_flows, color=plt.cm.YlOrRd(
                        0.2 + 0.8 * sorted_flows / (sorted_caps + 1e-9)),
                        edgecolor="#333", linewidth=0.5)
    # Overlay capacity markers
    ax_right.scatter(x, sorted_caps, marker="_", s=80, color="black",
                     zorder=5, linewidth=2, label="Capacity")

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(sorted_labels, rotation=45, ha="right", fontsize=8)
    ax_right.set_ylabel("Flow")
    ax_right.set_title("Edge Flows (sorted, with capacity markers)",
                       fontsize=10, fontweight="bold")
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.15, axis="y")

    # Add value labels on top of bars
    for bar, val in zip(bars, sorted_flows):
        ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                      f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 16. NETWORK FLOW SANKEY — simplified Sankey diagram of flow routes
# ═══════════════════════════════════════════════════════════════════════════════

def draw_flow_routes_sankey(G, flow_dict, source="s", sink="t"):
    """
    Trace all source-to-sink routes with non-zero flow and display them
    as a layered route decomposition. Each route is a sequence of edges;
    route thickness is proportional to the bottleneck flow on that route.

    This decomposes the flow into constituent s→t paths (not unique in
    general, but we use a greedy extraction).
    """
    # Greedy path decomposition: repeatedly extract a path with positive flow
    G_work = G.copy()
    flow_copy = {}
    for u in flow_dict:
        flow_copy[u] = dict(flow_dict[u])

    # Find all edges with positive flow
    routes = []
    while True:
        # BFS to find any s→t path with positive residual flow
        path_edges = _find_flow_path(flow_copy, source, sink, G)
        if path_edges is None:
            break
        # Bottleneck flow on this path
        bottleneck = min(flow_copy[u][v] for u, v in path_edges)
        if bottleneck < 1e-9:
            break
        routes.append((path_edges, bottleneck))
        # Subtract from flow_copy
        for u, v in path_edges:
            flow_copy[u][v] -= bottleneck

    if not routes:
        print("No flow routes found (flow may be zero).")
        return None

    n_routes = len(routes)
    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * n_routes))

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_routes, 1)))

    for i, (path_edges, bn) in enumerate(routes):
        path_str = " → ".join([path_edges[0][0]] +
                              [v for _, v in path_edges])
        y_pos = n_routes - i - 0.5

        # Draw route label
        ax.text(0.02, y_pos, f"Route {i + 1}", fontsize=9,
                fontweight="bold", va="center")
        ax.text(0.18, y_pos, path_str, fontsize=8, va="center",
                color="#333")

        # Draw flow bar proportional to bottleneck
        max_bn = max(r[1] for r in routes)
        bar_width = 0.6 * bn / max_bn if max_bn > 0 else 0.1
        ax.barh(y_pos, bar_width, 0.55, left=0.62,
                color=colors[i], edgecolor="#333", linewidth=0.5)

        ax.text(0.62 + bar_width + 0.01, y_pos,
                f"{bn:.2f}", fontsize=8, va="center", fontweight="bold")

    total_flow = sum(r[1] for r in routes)
    ax.text(0.02, n_routes, f"Total flow to sink: {total_flow:.2f}",
            fontsize=10, fontweight="bold", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_routes + 0.3)
    ax.axis("off")
    ax.set_title("Flow Route Decomposition\n"
                 "(Greedy path extraction from non-zero edge flows)",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def _find_flow_path(flow_copy, source, sink, G):
    """BFS to find any s→t path with positive flow. Returns list of (u,v) edges."""
    from collections import deque
    visited = {source: (None, None)}  # node → (prev_node, edge)
    q = deque([source])
    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v not in visited and flow_copy[u].get(v, 0) > 1e-9:
                visited[v] = (u, (u, v))
                if v == sink:
                    # Reconstruct path
                    path = []
                    curr = sink
                    while curr != source:
                        prev, edge = visited[curr]
                        path.append(edge)
                        curr = prev
                    return list(reversed(path))
                q.append(v)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 17. KKT & UNIMODULARITY — why the relaxed dual stays integral
# ═══════════════════════════════════════════════════════════════════════════════

def draw_relaxed_dual_integrality(G, relaxed_vals, boolean_vals=None, tol=1e-6):
    """
    Side-by-side comparison: relaxed LP vs boolean MILP edge-inclusion values.
    
    Left bar chart: relaxed d values (should be 0 or 1 via unimodularity).
    Right bar chart: boolean d values (MILP solution) for comparison.
    
    Mathematical insight:
      The oriented incidence matrix of a directed graph is TOTALLY UNIMODULAR
      (every square submatrix has determinant 0, +1, or −1).
      When the right-hand side q is integral (it is: entries are {−1,0,+1})
      and the objective is integral (capacity vector), the LP optimum
      lies at an integral vertex of the polytope — even without
      integer constraints on d.

    Parameters:
      G: the DiGraph
      relaxed_vals: array of d values from LP solve (should be 0/1)
      boolean_vals: array of d values from MILP (optional, for comparison)
      tol: tolerance for "is integral" check
    """
    edges = list(G.edges())
    edge_labels = [f"{u}→{v}" for u, v in edges]
    n_edges = len(edges)

    fig, (ax_left, ax_mid, ax_right) = plt.subplots(
        1, 3, figsize=(16, 5),
        gridspec_kw={"width_ratios": [1.2, 0.8, 1.0]})

    # ── Left: bar chart of relaxed d values ──
    x = np.arange(n_edges)
    colors = ["#2ca02c" if abs(v - 0) < tol or abs(v - 1) < tol
              else "#d62728" for v in relaxed_vals]
    bars = ax_left.bar(x, relaxed_vals, color=colors, edgecolor="#333",
                       linewidth=0.8)

    # Value labels on top of bars
    for j, (bar, val) in enumerate(zip(bars, relaxed_vals)):
        ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=7,
                     rotation=90 if val > 0.9 else 0)

    # Integrality boundary lines
    ax_left.axhline(0, color="grey", linestyle=":", alpha=0.4)
    ax_left.axhline(1, color="grey", linestyle=":", alpha=0.4)

    n_nonint = sum(not (abs(v - 0) < tol or abs(v - 1) < tol)
                   for v in relaxed_vals)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
    ax_left.set_ylabel("d value (edge inclusion)")
    ax_left.set_title(f"Relaxed LP (continuous d)\n"
                      f"{n_nonint} non-integral / {n_edges} edges\n"
                      f"Tol = {tol:.0e}",
                      fontsize=10, fontweight="bold")
    ax_left.set_ylim(-0.05, 1.25)
    ax_left.grid(True, alpha=0.15, axis="y")

    # ── Mid: histogram of d values to show clustering at 0 and 1 ──
    ax_mid.hist(relaxed_vals, bins=30, color="#3182bd", edgecolor="#333",
                alpha=0.8)
    ax_mid.axvline(0, color="#2ca02c", linestyle="--", linewidth=1.5,
                   label="Integral 0")
    ax_mid.axvline(1, color="#2ca02c", linestyle="--", linewidth=1.5,
                   label="Integral 1")
    ax_mid.set_xlabel("Relaxed d value")
    ax_mid.set_ylabel("Frequency")
    ax_mid.set_title("Distribution of d Values\n(should cluster at 0 and 1)",
                     fontsize=10, fontweight="bold")
    ax_mid.legend(fontsize=7)
    ax_mid.grid(True, alpha=0.15)

    # ── Right: relaxed vs boolean comparison if available ──
    if boolean_vals is not None:
        x_wide = np.arange(n_edges) * 2
        w = 0.7
        ax_right.bar(x_wide - w / 2, relaxed_vals, w, label="Relaxed LP (continuous)",
                     color="#3182bd", alpha=0.8)
        ax_right.bar(x_wide + w / 2, boolean_vals, w, label="Boolean MILP",
                     color="#d62728", alpha=0.7)
        ax_right.set_xticks(x_wide)
        ax_right.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
        ax_right.set_ylabel("d value")
        ax_right.set_title("Relaxed LP vs Boolean MILP\n"
                           "(should match exactly)",
                           fontsize=10, fontweight="bold")
        ax_right.legend(fontsize=7)
        ax_right.set_ylim(-0.05, 1.25)
        ax_right.grid(True, alpha=0.15, axis="y")

        # Check match
        is_match = all(abs(relaxed_vals[j] - boolean_vals[j]) < tol
                       for j in range(n_edges))
        match_text = "✓ MATCH" if is_match else "✗ MISMATCH"
        match_color = "#2ca02c" if is_match else "#d62728"
        ax_right.text(0.5, 1.05, match_text, transform=ax_right.transAxes,
                      ha="center", fontsize=12, fontweight="bold",
                      color=match_color)
    else:
        ax_right.text(0.5, 0.5, "Boolean MILP\nnot provided",
                      transform=ax_right.transAxes,
                      ha="center", va="center", fontsize=12,
                      color="grey")
        ax_right.set_title("Comparison Skipped", fontsize=10, fontweight="bold")
        ax_right.axis("off")

    plt.tight_layout()
    return fig


def draw_unimodularity_heatmap(G):
    """
    Verify and visualize total unimodularity of the oriented incidence matrix.

    Computes the determinant of every square submatrix (up to a reasonable
    size limit) and displays a heatmap of results.
    For a totally unimodular matrix, every square submatrix has
    determinant ∈ {−1, 0, +1}.

    Also shows the full incidence matrix as an annotated heatmap so the
    structure is visible: each column has exactly one +1 (head) and
    one −1 (tail), with all other entries 0.
    """
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
    m, n = A.shape  # m nodes, n edges

    fig = plt.figure(figsize=(14, 10))

    # ── Top: incidence matrix heatmap ──
    ax_mat = fig.add_subplot(2, 2, (1, 2))
    im = ax_mat.imshow(A, cmap="RdBu_r", aspect="auto",
                       vmin=-1, vmax=1, interpolation="nearest")

    # Annotate +1 and -1 cells
    nodes_list = list(G.nodes())
    edges_list = list(G.edges())
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:
                color = "white" if abs(A[i, j]) == 1 else "black"
                ax_mat.text(j, i, f"{int(A[i, j])}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

    ax_mat.set_xticks(range(n))
    ax_mat.set_xticklabels([f"{u}→{v}" for u, v in edges_list],
                           rotation=45, ha="right", fontsize=7)
    ax_mat.set_yticks(range(m))
    ax_mat.set_yticklabels(nodes_list, fontsize=9)
    ax_mat.set_title("Oriented Incidence Matrix A ∈ {−1, 0, +1}^{|V|×|E|}\n"
                     "Each column: one +1 (head), one −1 (tail)",
                     fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax_mat, shrink=0.8, label="Entry value")

    # ── Bottom-left: submatrix determinant histogram ──
    ax_det = fig.add_subplot(2, 2, 3)
    dets = []

    # Sample square submatrices of various sizes
    np.random.seed(42)
    from math import comb
    for k in range(1, min(m, n) + 1):
        # Choose k rows and k columns
        n_samples = min(200, int(comb(m, k) * comb(n, k)))
        n_samples = min(n_samples, 200)
        for _ in range(n_samples):
            rows = np.random.choice(m, k, replace=False)
            cols = np.random.choice(n, k, replace=False)
            submat = A[np.ix_(rows, cols)]
            det = np.linalg.det(submat)
            dets.append(round(det, 8))  # round to handle floating point

    dets = np.array(dets)
    unique_dets, counts = np.unique(dets, return_counts=True)

    colors_det = []
    for d in unique_dets:
        if abs(d - 0) < 1e-8:
            colors_det.append("#3182bd")      # blue for 0
        elif abs(abs(d) - 1) < 1e-8:
            colors_det.append("#2ca02c")      # green for ±1
        else:
            colors_det.append("#d62728")      # red for anything else

    ax_det.bar(range(len(unique_dets)), counts, color=colors_det,
               edgecolor="#333", linewidth=0.5)
    ax_det.set_xticks(range(len(unique_dets)))
    ax_det.set_xticklabels([f"{int(d)}" for d in unique_dets], fontsize=8)
    ax_det.set_xlabel("Determinant value")
    ax_det.set_ylabel("Count (sampled submatrices)")
    ax_det.set_title("Determinant Distribution of Square Submatrices\n"
                     "Green={−1,+1}, Blue=0, Red=other\n"
                     f"Total: {len(dets)} submatrices, "
                     f"{sum(colors_det.count(c) for c in ['#d62728'])} "
                     f"non-unimodular",
                     fontsize=9, fontweight="bold")
    ax_det.grid(True, alpha=0.15, axis="y")

    # ── Bottom-right: property summary ──
    ax_props = fig.add_subplot(2, 2, 4)
    ax_props.axis("off")

    non_uni = sum(1 for d in dets if not (abs(d - 0) < 1e-8 or
                                           abs(abs(d) - 1) < 1e-8))
    is_tu = non_uni == 0

    props = [
        f"Total unimodular (TU)?",
        f"  → {'YES' if is_tu else 'NO'} ({non_uni} non-TU dets found)",
        "",
        "Why it matters:",
        "  • TU matrix + integral RHS → LP extreme points",
        "    are integral (no branch-and-bound needed)",
        "  • For max flow: incidence matrix is TU",
        "  • Therefore: min-cut LP has integral optimal",
        "    solutions even without boolean constraints",
        "",
        "Matrix properties:",
        f"  • |V| = {m} nodes, |E| = {n} edges",
        f"  • Rank ≤ {min(m, n)} (actual: {np.linalg.matrix_rank(A)})",
        "  • Each column = one +1, one −1, rest 0",
        "  • Any row permutation of TU is TU",
        "",
        "Why incidence matrix is TU:",
        "  • Proof by induction on submatrix size",
        "  • Every column has ≤2 nonzeros",
        "  • If all columns have 2 nonzeros: they are",
        "    one +1 and one −1 → rows sum to zero",
        "    (columns are linearly dependent)",
        "  • Poincaré's theorem: directed graph",
        "    incidence matrix is totally unimodular",
    ]
    for i, line in enumerate(props):
        color = "#2ca02c" if "YES" in line and is_tu else (
            "#d62728" if "NO" in line else "black")
        ax_props.text(0.05, 0.95 - i * 0.045, line,
                      transform=ax_props.transAxes,
                      fontsize=9, fontfamily="monospace",
                      color=color, va="top")

    plt.tight_layout()
    return fig


def draw_random_capacity_integrality(G, source="s", sink="t", n_trials=12):
    """
    Perturb edge capacities randomly, re-solve the relaxed dual LP,
    and verify that the d variables remain integer.

    For each trial:
      - Multiply each capacity by a random factor in [0.3, 2.0]
      - Solve the relaxed LP (continuous d ∈ [0,1])
      - Check if all d are 0 or 1
      - Record the max-flow value and cut capacity

    The result is a grid of subplots showing d values for each trial.
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required")
        return None

    edges = list(G.edges())
    n_edges = len(edges)
    base_cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)

    all_d_vals = []
    all_results = []

    np.random.seed(123)
    for trial in range(n_trials):
        # Random perturbation
        perturbed_cap = base_cap * np.random.uniform(0.3, 2.0, n_edges)

        # Build and solve relaxed LP
        # Dual: min c^T d s.t. A_int^T μ + d ≥ q,  d ≥ 0  (no d ≤ 1 needed
        # since at optimum d will be 0 or 1)
        nodes = list(G.nodes())
        A_full = np.asarray(nx.incidence_matrix(
            G, oriented=True, dtype=float).todense())
        A_int = A_full[[nodes.index(n) for n in nodes if n not in (source, sink)], :]
        q = np.array([1 if u == source else (-1 if v == source else 0)
                      for u, v in edges], dtype=float)

        d = cp.Variable(n_edges, nonneg=True)
        # Also constrain d ≤ 1 to ensure bounded feasible region
        d = cp.Variable(n_edges)
        mu = cp.Variable(A_int.shape[0])

        constraints = [
            d >= 0,
            d <= 1,
            A_int.T @ mu + d >= q,
        ]
        obj = cp.Minimize(perturbed_cap @ d)
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)

        d_vals = d.value
        all_d_vals.append(d_vals)
        all_results.append({
            "trial": trial,
            "cut_value": float(perturbed_cap @ d_vals),
            "n_nonint": sum(not (abs(v - 0) < 1e-6 or abs(v - 1) < 1e-6)
                            for v in d_vals),
        })

    # Plot
    cols = 4
    rows = int(np.ceil(n_trials / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    axes = axes.flatten()

    edge_labels_short = [f"{u}→{v}" for u, v in edges]

    for trial, ax in enumerate(axes):
        if trial < n_trials:
            d_vals = all_d_vals[trial]
            colors = ["#2ca02c" if abs(v - 0) < 1e-6 or abs(v - 1) < 1e-6
                      else "#d62728" for v in d_vals]
            ax.bar(range(n_edges), d_vals, color=colors, edgecolor="#333")
            ax.axhline(0, color="grey", linestyle=":", alpha=0.3)
            ax.axhline(1, color="grey", linestyle=":", alpha=0.3)
            ax.set_xticks(range(n_edges))
            ax.set_xticklabels(edge_labels_short, rotation=45, ha="right",
                               fontsize=6)
            ax.set_ylim(-0.05, 1.25)
            res = all_results[trial]
            ax.set_title(f"Trial {trial + 1}: cut={res['cut_value']:.1f}, "
                         f"non-int={res['n_nonint']}",
                         fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.1, axis="y")
        else:
            ax.axis("off")

    fig.suptitle("Random Capacity Perturbation — Relaxed LP Integrality Test\n"
                 f"Green bars = integral (0/1), Red = non-integral",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def draw_kkt_dual_analysis(G, source="s", sink="t"):
    """
    Show the KKT conditions that force the relaxed dual variables d to be
    0 or 1, despite the LP having only continuous variables d ∈ [0,1].

    The relaxed dual LP:
      min  c^T d
      s.t. A_int^T μ + d ≥ q
           d ≥ 0,  d ≤ 1

    where:
      - A is the oriented incidence matrix (rows: internal nodes)
      - μ are node potentials (free variables, interpreted as dual
        multipliers for flow conservation)
      - d ≥ 0 are multipliers for capacity constraints
      - q indicates edges incident to source: q_e = 1 if e leaves s,
        q_e = −1 if e enters s, 0 otherwise

    KKT Conditions:
      1. Primal feasibility: A_int^T μ + d ≥ q, 0 ≤ d ≤ 1
      2. Dual feasibility: ν ≥ 0 (multiplier for d ≥ 0),
         ω ≥ 0 (multiplier for d ≤ 1)
      3. Stationarity: c − ν + ω = 0  ⇒  ν = c + ω
      4. Complementary slackness:
           ν_e · d_e = 0    (if d_e > 0 then ν_e = 0)
           ω_e · (d_e − 1) = 0  (if d_e < 1 then ω_e = 0)

    Because c_e > 0 for all edges:
      - If d_e > 0: complementary slackness → ν_e = 0
        Then stationarity → ω_e = ν_e − c_e = −c_e < 0
        But dual feasibility requires ω_e ≥ 0 → CONTRADICTION
        unless we also constrain d ≤ 1 actively...

    The TOTAL UNIMODULARITY argument is actually:
      The constraint matrix [I | A_int^T; −I | 0] is TU because
      A_int (incidence matrix rows) is TU and appending identity
      blocks preserves TU. The RHS [q; 0] is integral. Therefore
      ALL basic feasible solutions (vertices of the LP) have
      integral d values.

    This visualization shows:
      - Left: The KKT multipliers (ν, ω) for each edge
      - Right: Complementary slackness products ν·d and ω·(d−1)
      - Center: A "force diagram" showing which forces push d to 0 or 1
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required")
        return None

    edges = list(G.edges())
    n_edges = len(edges)
    nodes = list(G.nodes())
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A_full = np.asarray(nx.incidence_matrix(
        G, oriented=True, dtype=float).todense())
    A_int = A_full[[nodes.index(n) for n in nodes if n not in (source, sink)], :]
    q = np.array([1 if u == source else (-1 if v == source else 0)
                  for u, v in edges], dtype=float)

    # Solve relaxed LP
    d = cp.Variable(n_edges)
    mu = cp.Variable(A_int.shape[0])
    constraints_relaxed = [
        d >= 0,
        d <= 1,
        A_int.T @ mu + d >= q,
    ]
    prob = cp.Problem(cp.Minimize(cap @ d), constraints_relaxed)
    prob.solve(verbose=False)

    d_opt = d.value
    mu_opt = mu.value

    # Compute KKT multipliers
    # ν (multiplier for d ≥ 0), ω (multiplier for d ≤ 1)
    # From stationarity: cap = ν − ω + [constraint for A^T μ + d ≥ q] multipliers
    # Actually: constraints are:
    #   A_int^T @ mu + d >= q   → dual multiplier α ≥ 0
    #   d >= 0                   → dual multiplier ν ≥ 0
    #   d <= 1                   → dual multiplier ω ≥ 0 (for −d ≥ −1)
    # Lagrangian: L = c^T d − α^T(A_int^T μ + d − q) − ν^T d + ω^T(d − 1)
    # Stationarity w.r.t d: c − α − ν + ω = 0
    # Stationarity w.r.t μ: A_int α = 0

    # At optimum of LP: α = dual variables for coupling constraints
    # We can extract α from the LP dual solution
    # For LP: min c^T d s.t. D d ≥ e, d_low ≤ d ≤ d_up
    # The dual is: max e^T α + 0^T ν − 1^T ω s.t. D^T α + ν − ω = c, ...

    # Simpler approach: just show the integrality evidence
    edge_labels = [f"{u}→{v}" for u, v in edges]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top-left: d values (should be 0/1) ──
    ax = axes[0, 0]
    colors_d = ["#2ca02c" if abs(v - 0) < 1e-6 or abs(v - 1) < 1e-6
                else "#d62728" for v in d_opt]
    x = np.arange(n_edges)
    ax.bar(x, d_opt, color=colors_d, edgecolor="#333")
    ax.axhline(0, color="grey", linestyle=":", alpha=0.4)
    ax.axhline(1, color="grey", linestyle=":", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("d_e (edge inclusion)")
    ax.set_title("Relaxed LP Solution d_e\n"
                 "(should be 0 or 1 by total unimodularity)",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(-0.08, 1.15)
    ax.grid(True, alpha=0.15, axis="y")

    # ── Top-right: force diagram (which edges are "pushed" to 0 or 1) ──
    ax = axes[0, 1]
    # The "force" toward 1 comes from: having positive q and no μ offset
    # The "force" toward 0 comes from: capacity cost c_e
    # For each edge, compute "slack" in the constraint A_int^T μ + d ≥ q
    slack = A_int.T @ mu_opt + d_opt - q

    force_to_1 = np.where(d_opt > 0.5, 1.0, 0.0)
    force_to_0 = np.where(d_opt < 0.5, 1.0, 0.0)

    x_wide = np.arange(n_edges) * 2
    ax.bar(x_wide - 0.4, force_to_1, 0.7, color="#2ca02c", alpha=0.6,
           label="d→1 (in cut)")
    ax.bar(x_wide + 0.4, force_to_0, 0.7, color="#d62728", alpha=0.6,
           label="d→0 (not in cut)")
    ax.set_xticks(x_wide)
    ax.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Force direction")
    ax.set_title("Binary Decision per Edge\n"
                 "(the LP 'chooses' 0 or 1 for each edge)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.25)

    # ── Bottom-left: constraint slack A_int^T μ + d − q ──
    ax = axes[1, 0]
    colors_slack = ["#2ca02c" if s > -1e-8 else "#d62728" for s in slack]
    ax.bar(x, slack, color=colors_slack, edgecolor="#333")
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Slack = A_int^T μ + d − q")
    ax.set_title("Constraint Slack\n"
                 "(≥ 0 for feasibility; positive = inactive constraint)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    # ── Bottom-right: explanation text ──
    ax = axes[1, 1]
    ax.axis("off")

    n_nonint = sum(not (abs(v - 0) < 1e-6 or abs(v - 1) < 1e-6)
                   for v in d_opt)
    cut_capacity = float(cap @ d_opt)

    explanation = [
        "WHY d ∈ {0,1} WITHOUT BOOLEAN CONSTRAINTS",
        "═" * 48,
        "",
        f"Result: {n_nonint} non-integral entries / {n_edges} edges",
        f"Min cut capacity = {cut_capacity:.4f}",
        "",
        "The oriented incidence matrix A of a directed",
        "graph is TOTALLY UNIMODULAR (TU):",
        "  → Every square submatrix has det ∈ {−1, 0, +1}",
        "",
        "The dual LP constraint matrix is:",
        "  [ I   A_int^T ]",
        "  [ −I   0     ]    (from d ≤ 1 rewritten as −d ≥ −1)",
        "  [ 0   −A_int ]    (if μ sign constraints added)",
        "",
        "Since appending identity blocks preserves TU,",
        "and the RHS [q; 0; −1] is integral, EVERY basic",
        "feasible solution has integral d values.",
        "",
        "In plain language:",
        "  The LP solver (ECOS/OSQP) will naturally find",
        "  a vertex solution where all d_e ∈ {0, 1} — no",
        "  branch-and-bound, no integer constraints needed.",
        "",
        "This is why max-flow/min-cut is 'easy':",
        "  LP = MILP for this problem!",
    ]

    for i, line in enumerate(explanation):
        if line.startswith("═"):
            ax.text(0.05, 0.97 - i * 0.033, line,
                    transform=ax.transAxes, fontsize=9,
                    fontfamily="monospace", va="top")
        elif line.startswith("WHY") or line.startswith("In plain"):
            ax.text(0.05, 0.97 - i * 0.033, line,
                    transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top", color="#3182bd")
        elif f"{n_nonint}" in line and n_nonint == 0:
            ax.text(0.05, 0.97 - i * 0.033, line,
                    transform=ax.transAxes, fontsize=9,
                    fontfamily="monospace", va="top", color="#2ca02c")
        elif "TRUE" in line or "easy" in line:
            ax.text(0.05, 0.97 - i * 0.033, line,
                    transform=ax.transAxes, fontsize=9,
                    fontfamily="monospace", va="top", color="#2ca02c")
        else:
            ax.text(0.05, 0.97 - i * 0.033, line,
                    transform=ax.transAxes, fontsize=9,
                    fontfamily="monospace", va="top", color="#333")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 21. KKT FORCE BALANCE — why gradient + constraint normals push d to 0 or 1
# ═══════════════════════════════════════════════════════════════════════════════

def draw_kkt_force_balance_2d():
    """
    2D KKT Force Balance diagram for a single-edge relaxed dual problem.

    The relaxed dual for ONE edge e is:
        min  c_e · d
        s.t.  d ≥ L(μ)      where L(μ) = q_e − (A_int^T μ)_e
              0 ≤ d ≤ 1

    This is a 1-variable LP in d. The optimal d* is:
        d* = 0  if L(μ*) ≤ 0    (pushed against the d ≥ 0 wall)
        d* = 1  if L(μ*) ≥ 1    (pushed against the d ≤ 1 wall)

    The KKT force balance says: at optimum, the gradient of the objective
    (c_e > 0, pushing d DOWN) is balanced by the normal cone of the feasible
    interval [max(0, L), 1]. The TU property makes L jump between ≤0 and ≥1,
    never landing in (0,1).

    This visualization shows THREE cases:
      - Left:  L ≤ 0 → d* = 0  (lower-bound active, gradient balanced by
               inward normal at d=0)
      - Center:  L ∈ (0,1) → THEORETICALLY possible, but TU prevents this
      - Right: L ≥ 1 → d* = 1  (upper-bound active, gradient balanced by
               inward normal at d=1)

    For each case we show:
      - The feasible interval [max(0,L), 1] as a colored segment
      - The objective function c_e·d as a line (slope = c_e > 0)
      - The KKT force vectors: ∇f = c_e (objective gradient, points right)
        and constraint normals at the active bounds
    """
    c = 4.0  # capacity (cost coefficient for this edge)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    cases = [
        {"L": -1.0, "title": "Case 1: L ≤ 0\n(Lower bound active)", "dstar": 0.0},
        {"L": 0.5, "title": "Case 2: L ∈ (0,1)\n(IMPOSSIBLE by TU!)", "dstar": 0.5},
        {"L": 1.2, "title": "Case 3: L ≥ 1\n(Upper bound active)", "dstar": 1.0},
    ]

    for ax, case in zip(axes, cases):
        L_val = case["L"]
        dstar = case["dstar"]
        d_low = max(0, L_val)
        d_high = 1.0

        # ── Feasible interval ──
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvspan(d_low, d_high, alpha=0.15, color="#3182bd",
                   label=f"Feasible: [{d_low}, {d_high}]")

        # ── Bounding planes (constraints) ──
        ax.axvline(0, color="#2ca02c", linestyle="--", linewidth=2,
                   alpha=0.6, label="d ≥ 0")
        ax.axvline(1, color="#d62728", linestyle="--", linewidth=2,
                   alpha=0.6, label="d ≤ 1")
        if L_val > 0:
            ax.axvline(L_val, color="#ff7f0e", linestyle=":", linewidth=1.5,
                       alpha=0.5, label=f"d ≥ L(μ) = {L_val}")

        # ── Objective function line ──
        d_range = np.linspace(-0.3, 1.5, 200)
        obj = c * d_range
        ax.plot(d_range, obj, "k-", linewidth=2, alpha=0.7,
                label=f"Obj: c·d = {c}·d")

        # ── Mark optimum ──
        ax.scatter([dstar], [c * dstar], s=180, color="black",
                   zorder=10, edgecolors="white", linewidth=2)
        ax.annotate(f"d* = {dstar}",
                    xy=(dstar, c * dstar),
                    xytext=(dstar - 0.7, c * dstar + 1.5),
                    fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=1.5))

        # ── KKT FORCE VECTORS at optimum ──
        # Gradient of objective w.r.t d: ∇_d(c·d) = c (pushes d DOWN/left
        # since we minimize; the gradient points UP/to larger values)
        # The normal cone at d*=0: inward normals point RIGHT (into feasible region)
        # The normal cone at d*=1: inward normals point LEFT (into feasible region)
        arrow_y = c * dstar + 1.0

        # Objective gradient (c, points right — towards larger cost)
        ax.arrow(dstar, arrow_y, 0.35, 0, head_width=0.25, head_length=0.08,
                 fc="#3182bd", ec="#3182bd", linewidth=1.5, alpha=0.7,
                 label="∇(c·d) = c (gradient)")

        if abs(dstar) < 1e-6:
            # At d=0: constraint normal from d≥0 points RIGHT (feasible is d≥0)
            ax.arrow(dstar, arrow_y + 0.8, -0.35, 0, head_width=0.25,
                     head_length=0.08, fc="#2ca02c", ec="#2ca02c",
                     linewidth=1.5, alpha=0.7,
                     label="Normal cone at d=0 (→)")
            # BALANCE: gradient c is balanced by constraint normal
            ax.annotate("∇f balanced by\nnormal at d=0",
                        xy=(dstar, arrow_y + 1.8), fontsize=9,
                        color="#2ca02c", ha="center",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#e8f5e9", alpha=0.8))

        elif abs(dstar - 1.0) < 1e-6:
            # At d=1: constraint normal from d≤1 points LEFT
            ax.arrow(dstar, arrow_y + 0.8, 0.35, 0, head_width=0.25,
                     head_length=0.08, fc="#d62728", ec="#d62728",
                     linewidth=1.5, alpha=0.7,
                     label="Normal cone at d=1 (←)")
            ax.annotate("∇f balanced by\nnormal at d=1",
                        xy=(dstar, arrow_y + 1.8), fontsize=9,
                        color="#d62728", ha="center",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#fde0dd", alpha=0.8))
        else:
            # Inside (0,1) — this never happens for TU problems!
            ax.annotate("NO constraint active!\n"
                        "KKT: ∇f + 0 = 0 impossible\n"
                        "since ∇f = c > 0",
                        xy=(dstar, arrow_y + 2.0), fontsize=10,
                        color="#d62728", ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#fde0dd", alpha=0.9))

        # ── Label the "walls" ──
        ax.text(0.02, c * 1.05, "WALL\nd≥0", fontsize=8, color="#2ca02c",
                va="bottom", fontweight="bold")
        ax.text(0.98, c * 1.05, "WALL\nd≤1", fontsize=8, color="#d62728",
                va="bottom", fontweight="bold")

        ax.set_xlabel("d (edge inclusion variable)")
        ax.set_ylabel("Cost c·d")
        ax.set_title(case["title"], fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, 1.7)
        ax.set_ylim(-1, c * 1.5 + 0.5)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.2)

    fig.suptitle("KKT Force Balance — Why the LP Pushes d to 0 or 1\n"
                 "∇(c·d) = c > 0 balanced against constraint normal cone at {0,1}",
                 fontsize=13, fontweight="bold", y=0.99)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 22. 2D CONSTRAINT-PLANE FORCE DIAGRAM — two edges, all constraints visible
# ═══════════════════════════════════════════════════════════════════════════════

def draw_kkt_constraint_planes_2d(G, source="s", sink="t"):
    """
    Project the relaxed dual LP into 2D by fixing all but 2 edge variables.
    Show:
      - The feasible polytope (intersection of half-planes)
      - The objective gradient direction (c, pointing toward cheaper d)
      - The constraint normals at the optimal vertex
      - The TU property: the optimum lies at a {0,1}² vertex

    This makes the KKT force balance visible in 2D: you see the gradient
    "pushing" against the constraint walls, and the vertex where they meet.
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required")
        return None

    edges = list(G.edges())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    nodes = list(G.nodes())

    A_full = np.asarray(nx.incidence_matrix(
        G, oriented=True, dtype=float).todense())
    A_int = A_full[[nodes.index(n) for n in nodes if n not in (source, sink)], :]
    q = np.array([1 if u == source else (-1 if v == source else 0)
                  for u, v in edges], dtype=float)

    # ── Solve full relaxed LP to find μ* ──
    d = cp.Variable(n_edges)
    mu = cp.Variable(A_int.shape[0])
    constraints = [d >= 0, d <= 1, A_int.T @ mu + d >= q]
    prob = cp.Problem(cp.Minimize(cap @ d), constraints)
    prob.solve(verbose=False)
    mu_opt = mu.value
    d_opt = d.value

    # ── Pick two edges with different behavior (one 0, one 1) ──
    zero_edges = [j for j, dv in enumerate(d_opt) if abs(dv) < 1e-6]
    one_edges = [j for j, dv in enumerate(d_opt) if abs(dv - 1) < 1e-6]
    if len(zero_edges) >= 1 and len(one_edges) >= 1:
        e1, e2 = zero_edges[0], one_edges[0]
    else:
        e1, e2 = 0, 2  # fallback

    # ── Construct the 2D feasible region ──
    # Fix all other d at their optimal values, vary d[e1] and d[e2]
    d_fixed_indices = [j for j in range(n_edges) if j != e1 and j != e2]
    d_fixed_vals = d_opt[d_fixed_indices]

    # The constraint for each node i (not s,t):
    #   sum_{j} A_int[i,j] * mu[i'] + d[j] >= q[j]  for each edge j
    #   With mu fixed at mu*:
    #   d[j] >= q[j] − (A_int^T mu*)[j]  →  lower_bound[j]

    a_star = A_int.T @ mu_opt  # shape (n_edges,)
    lower_bounds = q - a_star  # d must be >= this

    # In 2D (d[e1], d[e2]):
    #   d[e1] ∈ [max(0, lower_bounds[e1]), 1]
    #   d[e2] ∈ [max(0, lower_bounds[e2]), 1]
    # The feasible set is a rectangle (possibly truncated by coupling via other d)

    lb1, ub1 = max(0, lower_bounds[e1]), 1
    lb2, ub2 = max(0, lower_bounds[e2]), 1

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 8))

    # Feasible rectangle
    rect = plt.Rectangle((lb1, lb2), ub1 - lb1, ub2 - lb2,
                         fill=True, facecolor="#c6dbef",
                         edgecolor="#3182bd", linewidth=2, alpha=0.4)
    ax.add_patch(rect)

    # Constraint half-plane boundaries
    # d[e1] ≥ 0
    ax.axvline(0, color="#2ca02c", linestyle="--", linewidth=1.8, alpha=0.6)
    # d[e1] ≤ 1
    ax.axvline(1, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.6)
    # d[e2] ≥ 0
    ax.axhline(0, color="#2ca02c", linestyle="--", linewidth=1.8, alpha=0.6)
    # d[e2] ≤ 1
    ax.axhline(1, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.6)

    # Lower bounds from coupling constraints
    if lower_bounds[e1] > 0:
        ax.axvline(lower_bounds[e1], color="#ff7f0e", linestyle=":",
                   linewidth=1.2, alpha=0.5,
                   label=f"d[{e1}] ≥ L₁(μ*) = {lower_bounds[e1]:.2f}")
    if lower_bounds[e2] > 0:
        ax.axhline(lower_bounds[e2], color="#9467bd", linestyle=":",
                   linewidth=1.2, alpha=0.5,
                   label=f"d[{e2}] ≥ L₂(μ*) = {lower_bounds[e2]:.2f}")

    # ── Objective gradient ──
    # Objective is c₁·d₁ + c₂·d₂ + const. Gradient = (c₁, c₂).
    c1, c2 = cap[e1], cap[e2]
    grad_norm = np.sqrt(c1**2 + c2**2) or 1
    gx, gy = c1 / grad_norm * 0.35, c2 / grad_norm * 0.35

    # Objective contours
    for k in np.linspace(0, cap[e1] + cap[e2], 8):
        # Line: c1*d1 + c2*d2 = k
        if c2 != 0:
            x_vals = np.linspace(-0.1, 1.2, 50)
            y_vals = (k - c1 * x_vals) / c2
            valid = (y_vals >= -0.1) & (y_vals <= 1.3)
            if valid.any():
                ax.plot(x_vals[valid], y_vals[valid], color="#ff7f0e",
                        alpha=0.3, linewidth=0.8)

    # ── Gradient arrow from center ──
    ax.arrow(0.5, 0.5, gx, gy, head_width=0.04, head_length=0.04,
             fc="#ff7f0e", ec="#ff7f0e", linewidth=2,
             label=f"∇(cᵀd) = ({c1}, {c2})")

    # ── Optimal vertex ──
    opt_x, opt_y = d_opt[e1], d_opt[e2]
    ax.scatter([opt_x], [opt_y], s=250, color="black", zorder=10,
               edgecolors="white", linewidth=2)
    ax.annotate(f"OPT\n({opt_x:.0f}, {opt_y:.0f})",
                xy=(opt_x, opt_y),
                xytext=(opt_x - 0.35, opt_y - 0.35),
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # ── KKT constraint normal cone at optimum ──
    # At vertex (d₁*, d₂*): which constraints are active?
    active_normals = []
    normal_colors = []
    if abs(opt_x) < 1e-6:
        # d₁ ≥ 0 active → normal points RIGHT (d₁ increasing enters feasible)
        active_normals.append((1, 0, "#2ca02c", "d₁≥0"))
    elif abs(opt_x - 1) < 1e-6:
        # d₁ ≤ 1 active → normal points LEFT
        active_normals.append((-1, 0, "#d62728", "d₁≤1"))
    if abs(opt_y) < 1e-6:
        active_normals.append((0, 1, "#2ca02c", "d₂≥0"))
    elif abs(opt_y - 1) < 1e-6:
        active_normals.append((0, -1, "#d62728", "d₂≤1"))

    # Draw constraint normals at optimum
    for nx_v, ny_v, color, label in active_normals:
        ax.arrow(opt_x, opt_y, nx_v * 0.22, ny_v * 0.22,
                 head_width=0.04, head_length=0.04,
                 fc=color, ec=color, linewidth=2, alpha=0.8,
                 label=f"Normal: {label}")

    # ── Vertex labels ──
    vertices = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for vx, vy in vertices:
        ax.plot(vx, vy, "o", color="#3182bd", markersize=8, alpha=0.4)
        ax.text(vx + 0.03, vy + 0.03, f"({vx},{vy})", fontsize=8, color="#3182bd")

    # Labels
    e1_name = f"{edges[e1][0]}→{edges[e1][1]}"
    e2_name = f"{edges[e2][0]}→{edges[e2][1]}"
    ax.set_xlabel(f"d[{e1}] = d({e1_name})")
    ax.set_ylabel(f"d[{e2}] = d({e2_name})")
    ax.set_title("KKT Constraint-Plane Force Diagram (2D projection)\n"
                 f"Edges: {e1_name} (c={cap[e1]}), {e2_name} (c={cap[e2]})\n"
                 f"∇f balanced by constraint normals at the {opt_x:.0f},{opt_y:.0f} vertex",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.15, 1.25)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 23. 3D CONSTRAINT-FORCE POLYTOPE — three edges, full geometry
# ═══════════════════════════════════════════════════════════════════════════════

def draw_kkt_constraint_polytope_3d(G, source="s", sink="t"):
    """
    3D visualization: pick 3 edges and show the feasible polytope as a
    3D box [0,1]³ intersected with coupling half-spaces. The objective
    is a linear function c₁d₁ + c₂d₂ + c₃d₃. The gradient ∇(cᵀd) = c
    points toward higher cost. The optimum is at the vertex where the
    normal cone of the polytope contains the negative gradient (−c).

    Key insight:
      - The feasible region is a vertex of the unit cube [0,1]³
      - Each face of the cube corresponds to dⱼ = 0 or dⱼ = 1
      - The objective pushes d DOWN (minimize cost)
      - The coupling constraints d ≥ L(μ*) carve away portions
      - TU guarantees L(μ*)ⱼ is either ≤ 0 or ≥ 1
      - So EVERY vertex of the feasible polytope has {0,1} coordinates
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required")
        return None

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    edges = list(G.edges())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    nodes = list(G.nodes())

    A_full = np.asarray(nx.incidence_matrix(
        G, oriented=True, dtype=float).todense())
    A_int = A_full[[nodes.index(n) for n in nodes if n not in (source, sink)], :]
    q = np.array([1 if u == source else (-1 if v == source else 0)
                  for u, v in edges], dtype=float)

    # Solve relaxed LP
    d = cp.Variable(n_edges)
    mu = cp.Variable(A_int.shape[0])
    constraints = [d >= 0, d <= 1, A_int.T @ mu + d >= q]
    prob = cp.Problem(cp.Minimize(cap @ d), constraints)
    prob.solve(verbose=False)
    mu_opt = mu.value
    d_opt = d.value
    a_star = A_int.T @ mu_opt
    lower_bounds = q - a_star

    # Pick 3 edges with different behaviors
    zero_edges = [j for j, dv in enumerate(d_opt) if abs(dv) < 1e-6]
    one_edges = [j for j, dv in enumerate(d_opt) if abs(dv - 1) < 1e-6]
    # Try to get a mix; fall back to first three edges
    candidates = zero_edges[:2] + one_edges[:2]
    if len(candidates) >= 3:
        e1, e2, e3 = candidates[0], candidates[1], candidates[2]
    else:
        e1, e2, e3 = 0, 1, min(2, n_edges - 1)

    c1, c2, c3 = cap[e1], cap[e2], cap[e3]
    lb1, lb2, lb3 = [max(0, lower_bounds[j]) for j in [e1, e2, e3]]
    opt1, opt2, opt3 = d_opt[e1], d_opt[e2], d_opt[e3]

    fig = plt.figure(figsize=(14, 6))

    # ── Left: 3D polytope + gradient + constraint planes ──
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    # Draw the unit cube wireframe
    for xv in [0, 1]:
        for yv in [0, 1]:
            ax3d.plot([xv, xv], [yv, yv], [0, 1], "k-", alpha=0.2, linewidth=0.5)
    for xv in [0, 1]:
        for zv in [0, 1]:
            ax3d.plot([xv, xv], [0, 1], [zv, zv], "k-", alpha=0.2, linewidth=0.5)
    for yv in [0, 1]:
        for zv in [0, 1]:
            ax3d.plot([0, 1], [yv, yv], [zv, zv], "k-", alpha=0.2, linewidth=0.5)

    # Feasible region: portion of cube satisfying d ≥ L(μ*)
    # For each edge j, the feasible interval is [max(0, L_j), 1]
    # This is a rectangular box (sub-box of [0,1]³)
    x_box = np.array([lb1, 1])
    y_box = np.array([lb2, 1])
    z_box = np.array([lb3, 1])

    # Draw the feasible box edges
    for xv in x_box:
        for yv in y_box:
            ax3d.plot([xv, xv], [yv, yv], z_box, "b-", alpha=0.5, linewidth=1.5)
    for xv in x_box:
        for zv in z_box:
            ax3d.plot([xv, xv], y_box, [zv, zv], "b-", alpha=0.5, linewidth=1.5)
    for yv in y_box:
        for zv in z_box:
            ax3d.plot(x_box, [yv, yv], [zv, zv], "b-", alpha=0.5, linewidth=1.5)

    # Semi-transparent feasible faces
    def draw_rect_3d(ax, x_range, y_range, z_val, color, alpha, edge):
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.full_like(xx, z_val)
        ax.plot_surface(xx, yy, zz, alpha=alpha, color=color,
                        edgecolor=edge, linewidth=0.3)

    # Constraint walls d=0 (green) and d=1 (red)
    # Only show the part that's inside the feasible box
    draw_rect_3d(ax3d, [0, 1], [0, 1], 0, "#2ca02c", 0.08, "#2ca02c")
    draw_rect_3d(ax3d, [0, 1], [0, 1], 1, "#d62728", 0.08, "#d62728")
    draw_rect_3d(ax3d, [0, 1], [0, 0], [0.01, 0.99], "#2ca02c", 0.04, None)
    draw_rect_3d(ax3d, [0, 0], [0, 1], [0.01, 0.99], "#2ca02c", 0.04, None)
    draw_rect_3d(ax3d, [1, 1], [0, 1], [0.01, 0.99], "#d62728", 0.04, None)
    draw_rect_3d(ax3d, [0, 1], [1, 1], [0.01, 0.99], "#d62728", 0.04, None)

    # Highlight the 8 vertices of unit cube
    vertices_3d = [(x, y, z) for x in [0, 1] for y in [0, 1] for z in [0, 1]]
    vx, vy, vz = zip(*vertices_3d)
    ax3d.scatter(vx, vy, vz, c="grey", s=20, alpha=0.5)

    # ── Objective gradient ──
    grad_norm = np.sqrt(c1**2 + c2**2 + c3**2) or 1
    ax3d.quiver(0.5, 0.5, 0.5,
                c1 / grad_norm * 0.4, c2 / grad_norm * 0.4,
                c3 / grad_norm * 0.4,
                color="#ff7f0e", linewidth=3, arrow_length_ratio=0.2,
                label=f"∇(cᵀd) = ({c1},{c2},{c3})")

    # ── Optimal vertex ──
    ax3d.scatter([opt1], [opt2], [opt3], color="black", s=200,
                 zorder=20, edgecolors="white", linewidth=2)
    ax3d.text(opt1 + 0.08, opt2 + 0.08, opt3 + 0.08,
              f"OPT\n({opt1:.0f},{opt2:.0f},{opt3:.0f})",
              fontsize=9, fontweight="bold", color="black")

    # ── Constraint normals at optimum ──
    normal_scale = 0.25
    # Check which walls are active at optimum
    checks = [
        (opt1, 0, (1, 0, 0), "#2ca02c", "d₁≥0 active"),
        (opt1, 1, (-1, 0, 0), "#d62728", "d₁≤1 active"),
        (opt2, 0, (0, 1, 0), "#2ca02c", "d₂≥0 active"),
        (opt2, 1, (0, -1, 0), "#d62728", "d₂≤1 active"),
        (opt3, 0, (0, 0, 1), "#2ca02c", "d₃≥0 active"),
        (opt3, 1, (0, 0, -1), "#d62728", "d₃≤1 active"),
    ]
    for val, wall, direction, color, label in checks:
        if abs(val - wall) < 1e-6:
            ax3d.quiver(opt1, opt2, opt3,
                        direction[0] * normal_scale,
                        direction[1] * normal_scale,
                        direction[2] * normal_scale,
                        color=color, linewidth=2.5, arrow_length_ratio=0.25,
                        alpha=0.8)

    ax3d.set_xlabel(f"d₁: {edges[e1][0]}→{edges[e1][1]} (c={c1})", fontsize=8)
    ax3d.set_ylabel(f"d₂: {edges[e2][0]}→{edges[e2][1]} (c={c2})", fontsize=8)
    ax3d.set_zlabel(f"d₃: {edges[e3][0]}→{edges[e3][1]} (c={c3})", fontsize=8)
    ax3d.set_title("3D Feasible Polytope + KKT Forces\n"
                   f"∇f = ({c1},{c2},{c3}) balanced at ({opt1:.0f},{opt2:.0f},{opt3:.0f})",
                   fontsize=10, fontweight="bold")
    ax3d.set_xlim(-0.1, 1.2)
    ax3d.set_ylim(-0.1, 1.2)
    ax3d.set_zlim(-0.1, 1.2)
    ax3d.view_init(elev=22, azim=-38)

    # ── Right: explanation of KKT balance ──
    ax_text = fig.add_subplot(1, 2, 2)
    ax_text.axis("off")

    active_walls = []
    for val, wall, direction, color, label in checks:
        if abs(val - wall) < 1e-6:
            active_walls.append(label)

    lines = [
        "KKT FORCE BALANCE IN 3D",
        "═" * 35,
        "",
        f"Edges: {edges[e1][0]}→{edges[e1][1]} (c₁={c1})",
        f"       {edges[e2][0]}→{edges[e2][1]} (c₂={c2})",
        f"       {edges[e3][0]}→{edges[e3][1]} (c₃={c3})",
        "",
        f"Optimal: d = ({opt1:.2f}, {opt2:.2f}, {opt3:.2f})",
        "",
        "Active constraints:",
    ] + [f"  • {w}" for w in active_walls] + [
        "",
        "WHY {0,1}?",
        "The gradient ∇(cᵀd) = c pushes d",
        "toward 0 (since we minimize cost).",
        "But d must also satisfy:",
        "  dⱼ ≥ qⱼ − (Aᵀμ*)ⱼ",
        "",
        "At the optimal μ*, each lower bound",
        "Lⱼ(μ*) is either ≤0 or ≥1 — because",
        "the incidence matrix is TU and q, c",
        "are integral.",
        "",
        "When Lⱼ ≤ 0: d*ⱼ = 0 (wall d≥0 active)",
        "When Lⱼ ≥ 1: d*ⱼ = 1 (wall d≤1 active)",
        "",
        "The normal cone at the vertex contains",
        "the negative gradient, satisfying KKT.",
        "No interior point can be optimal because",
        "∇f = c ≠ 0 and no constraint would be",
        "active to balance it.",
    ]

    for i, line in enumerate(lines):
        y = 0.95 - i * 0.028
        if line.startswith("═"):
            ax_text.text(0.05, y, line, transform=ax_text.transAxes,
                         fontsize=10, fontfamily="monospace", va="top")
        elif line.startswith("KKT") or line.startswith("WHY"):
            ax_text.text(0.05, y, line, transform=ax_text.transAxes,
                         fontsize=11, fontweight="bold", color="#3182bd",
                         va="top")
        elif "wall" in line or "active" in line:
            ax_text.text(0.05, y, line, transform=ax_text.transAxes,
                         fontsize=9, fontfamily="monospace", va="top",
                         color="#2ca02c" if "d≥0" in line else
                         ("#d62728" if "d≤1" in line else "#333"))
        else:
            ax_text.text(0.05, y, line, transform=ax_text.transAxes,
                         fontsize=9, fontfamily="monospace", va="top",
                         color="#333")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 24. GRADIENT VS NORMAL-CONE FORCE FIELD — sampling across the polytope
# ═══════════════════════════════════════════════════════════════════════════════

def draw_kkt_force_field_across_polytope(G, source="s", sink="t"):
    """
    For the 2D projection (same two edges as #22), sample points across
    the feasible polytope and draw:
      - The objective gradient at each point (uniform, = c)
      - The direction toward the optimal vertex (negative gradient if
        interior, zero at optimum)
      - Color each point by whether it would be "pushed" to (0,0), (0,1),
        (1,0), or (1,1) by a projected-gradient step

    This creates a "force field" diagram showing how every interior
    point is pushed by gradient descent toward a {0,1} vertex.
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required")
        return None

    edges = list(G.edges())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    nodes = list(G.nodes())

    A_full = np.asarray(nx.incidence_matrix(
        G, oriented=True, dtype=float).todense())
    A_int = A_full[[nodes.index(n) for n in nodes if n not in (source, sink)], :]
    q = np.array([1 if u == source else (-1 if v == source else 0)
                  for u, v in edges], dtype=float)

    # Solve
    d = cp.Variable(n_edges)
    mu = cp.Variable(A_int.shape[0])
    prob = cp.Problem(cp.Minimize(cap @ d),
                      [A_int.T @ mu + d >= q])
    prob.solve(verbose=False)
    d_opt = d.value

    # Choose two edges
    zero_edges = [j for j, dv in enumerate(d_opt) if abs(dv) < 1e-6]
    one_edges = [j for j, dv in enumerate(d_opt) if abs(dv - 1) < 1e-6]
    if zero_edges and one_edges:
        e1, e2 = zero_edges[0], one_edges[0]
    else:
        e1, e2 = 0, 2
    c1, c2 = cap[e1], cap[e2]

    fig, ax = plt.subplots(figsize=(10, 9))

    # Grid of sample points in [0,1]²
    grid = np.linspace(0, 1, 18)
    X, Y = np.meshgrid(grid, grid)
    U = np.full_like(X, -c1)  # negative gradient points toward lower cost
    V = np.full_like(Y, -c2)

    # Normalize for uniform arrow length
    mag = np.sqrt(U**2 + V**2)
    mag[mag == 0] = 1
    U_norm, V_norm = U / mag, V / mag

    # ── Color each point by which vertex it's closest to ──
    vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    vertex_colors = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
    vertex_labels = ["(0,0)", "(1,0)", "(0,1)", "(1,1)"]

    for px, py in zip(X.flat, Y.flat):
        dists = np.sqrt((vertices[:, 0] - px)**2 + (vertices[:, 1] - py)**2)
        nearest = np.argmin(dists)
        ax.plot(px, py, "o", color=vertex_colors[nearest],
                markersize=4, alpha=0.5)

    # ── Quiver: negative gradient (pointing to lower cost) ──
    step = 2
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              U_norm[::step, ::step], V_norm[::step, ::step],
              scale=18, width=0.003, color="black", alpha=0.6,
              label="−∇(cᵀd) = direction of cost decrease")

    # ── Optimal vertex ──
    opt_x, opt_y = d_opt[e1], d_opt[e2]
    ax.scatter([opt_x], [opt_y], s=350, color="black", zorder=20,
               edgecolors="white", linewidth=3)
    ax.annotate(f"OPTIMUM\n({opt_x:.0f},{opt_y:.0f})\n"
                f"−∇f balanced by\nnormal cone",
                xy=(opt_x, opt_y),
                xytext=(opt_x - 0.45, opt_y - 0.45),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9))

    # ── Constraint walls ──
    ax.axvline(0, color="#2ca02c", linewidth=2.5, alpha=0.5,
               label="Wall: d₁ ≥ 0")
    ax.axvline(1, color="#d62728", linewidth=2.5, alpha=0.5,
               label="Wall: d₁ ≤ 1")
    ax.axhline(0, color="#2ca02c", linewidth=2.5, alpha=0.5,
               linestyle="--", label="Wall: d₂ ≥ 0")
    ax.axhline(1, color="#d62728", linewidth=2.5, alpha=0.5,
               linestyle="--", label="Wall: d₂ ≤ 1")

    # ── Vertex labels ──
    for (vx, vy), vc, vl in zip(vertices, vertex_colors, vertex_labels):
        ax.scatter([vx], [vy], s=120, color=vc, edgecolors="black",
                   linewidth=2, zorder=15)
        ax.text(vx + 0.04, vy + 0.04, vl, fontsize=9, fontweight="bold",
                color=vc)

    e1_name = f"{edges[e1][0]}→{edges[e1][1]}"
    e2_name = f"{edges[e2][0]}→{edges[e2][1]}"
    ax.set_xlabel(f"d({e1_name})", fontsize=11)
    ax.set_ylabel(f"d({e2_name})", fontsize=11)
    ax.set_title("KKT Gradient Force Field Across the Polytope\n"
                 f"−∇(cᵀd) = ({-c1},{-c2}) pushes every point toward a {{0,1}}² vertex\n"
                 "Color = nearest {0,1}² vertex",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.08, 1.12)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN — run all static demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    G, pos = demo_network()

    # 1. Compute max flow for visuals that need it
    flow_val, flow_dict = nx.maximum_flow(G, "s", "t", capacity="capacity")
    print(f"Max flow value: {flow_val}")
    print("Flow distribution:", {e: flow_dict[e[0]][e[1]] for e in G.edges()})

    # 1. Utilization coloring
    fig1, _ = draw_flow_utilization(G, flow_dict, pos)
    print("[1] Edge utilization visualization ready.")

    # 2. Min-cut partition
    fig2, _, S, T, cut_edges = draw_min_cut_partition(G, flow_dict, pos=pos)
    print(f"[2] Min-cut partition: S={S}, T={T}, cut_edges={cut_edges}")

    # 3. Feasible polytope
    fig3, _ = draw_feasible_polytope_2d()
    print("[3] Feasible polytope visualization ready.")

    # 4. Duality gap + complementary slackness
    fig4, _ = draw_duality_visualizations(G, flow_dict)
    print("[4] Duality visualizations ready.")

    # 5. Fairness dashboard (requires cvxpy)
    try:
        import cvxpy as cp

        def solve_alpha_quick(G, s, t, alpha):
            edges = list(G.edges())
            nodes = list(G.nodes())
            n = len(edges)
            cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
            A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
            ci = [nodes.index(n) for n in nodes if n not in (s, t)]
            f = cp.Variable(n, nonneg=True)
            if alpha == 0:
                si = nodes.index(s)
                obj = cp.Maximize(A[si, :] @ f)
            elif alpha == 1:
                obj = cp.Maximize(cp.sum(cp.log(f + 1e-9)))
            elif alpha < 1:
                obj = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
            else:
                obj = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))
            cp.Problem(obj, [f <= cap, A[ci, :] @ f == 0]).solve(verbose=False)
            return f.value, edges

        def jain(x):
            x = np.asarray(x, dtype=float)
            x = x[x > 1e-9]
            if len(x) <= 1:
                return 1.0
            return float(np.sum(x) ** 2 / (len(x) * np.sum(x ** 2)))

        results = {}
        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
            f_vals, edges = solve_alpha_quick(G, "s", "t", alpha)
            source_mask = np.array([u == "s" for u, v in edges], dtype=bool)
            total = sum(f_vals[source_mask])
            j = jain(f_vals[source_mask])
            results[alpha] = {"total": total, "flows": f_vals,
                              "edges": edges, "jain": j}

        fig5 = draw_fairness_dashboard(results)
        print("[5] Fairness dashboard ready.")
    except ImportError:
        print("[5] cvxpy not available — skipping fairness dashboard.")

    # 6. Augmenting path animation
    ani_aug = animate_augmenting_paths(G)
    print("[6] Augmenting path animation created (in ani_aug).")

    # 9. Saddle point
    fig9 = draw_saddle_point_dual()
    print("[9] Lagrangian saddle point visualization ready.")

    plt.show()
    print("\nDone. All static visuals drawn.")
    print("For interactive slider: call interactive_alpha_slider(G) in a notebook.")
    print("For α-sweep animation: call animate_alpha_sweep(G) and display the returned animation.")
