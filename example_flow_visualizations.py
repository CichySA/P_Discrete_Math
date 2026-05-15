"""
Example: Visualization Approaches for Max Flow & Fairness Flow
===============================================================
A catalog of visualization techniques for flow networks, duality,
and fairness optimization — from static network drawings to animated
augmenting paths and interactive α-fairness sliders.

Visualizations covered:
  1. Static graph: flow/capacity labels, edge coloring by utilization
  2. Min-cut partition: S/T coloring, bottleneck highlighting
  3. Feasible polytope: 2D projection of flow constraints + objective
  4. Duality gap: primal-dual convergence, complementary slackness heatmap
  5. Fairness comparison: bar charts, Pareto frontier, 3D α-surface
  6. Animation: augmenting-path discovery, α-sweep morphing
  7. Interactive: α slider (ipywidgets), capacity sensitivity click-map
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

def animate_alpha_sweep(G, source="s", sink="t", n_frames=30):
    """
    Animate the flow distribution as α sweeps from 0 (utilitarian)
    to 5 (strongly fair). Requires cvxpy to solve at each frame.
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy required for alpha-sweep animation")
        return None

    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]

    alphas = np.linspace(0.01, 5.0, n_frames)
    all_flows = []

    # Pre-compute flows for each α
    for alpha in alphas:
        f = cp.Variable(n_edges, nonneg=True)
        if alpha < 0.05:
            s_idx = nodes.index(source)
            obj = cp.Maximize(A[s_idx, :] @ f)
        elif abs(alpha - 1.0) < 0.05:
            obj = cp.Maximize(cp.sum(cp.log(f + 1e-9)))
        elif alpha < 1:
            obj = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
        else:
            obj = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))

        constraints = [f <= cap, A[cons_idx, :] @ f == 0]
        cp.Problem(obj, constraints).solve(solver=cp.ECOS, verbose=False)

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

        total_flow = sum(flows[i] for i, (u, v) in enumerate(edges) if u == source)
        ax_graph.set_title(f"α = {alpha:.2f}  |  Total flow = {total_flow:.2f}",
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

def interactive_alpha_slider(G, source="s", sink="t"):
    """
    Requires: ipywidgets, cvxpy, and a Jupyter notebook (or %matplotlib widget).

    An interactive slider that re-solves the α-fairness flow and updates
    both the graph and the bar chart in real time.

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

    edges = list(G.edges())
    nodes = list(G.nodes())
    n_edges = len(edges)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)
    A = np.asarray(nx.incidence_matrix(G, oriented=True, dtype=float).todense())
    cons_idx = [nodes.index(n) for n in nodes if n not in (source, sink)]

    pos = nx.spring_layout(G, seed=7)

    def solve_for_alpha(alpha):
        f = cp.Variable(n_edges, nonneg=True)
        if alpha < 0.01:
            s_idx = nodes.index(source)
            obj = cp.Maximize(A[s_idx, :] @ f)
        elif abs(alpha - 1.0) < 0.01:
            obj = cp.Maximize(cp.sum(cp.log(f + 1e-9)))
        elif alpha < 1:
            obj = cp.Maximize(cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (1 - alpha))
        else:
            obj = cp.Maximize(-cp.sum(cp.power(f + 1e-9, 1 - alpha)) / (alpha - 1))

        constraints = [f <= cap, A[cons_idx, :] @ f == 0]
        cp.Problem(obj, constraints).solve(solver=cp.ECOS, verbose=False)
        return f.value if f.value is not None else np.zeros(n_edges)

    def update_display(alpha):
        clear_output(wait=True)
        flows = solve_for_alpha(alpha)
        total = sum(flows[i] for i, (u, v) in enumerate(edges) if u == source)

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
        ax_g.set_title(f"α = {alpha:.2f} | Total flow = {total:.2f}", fontsize=11,
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
    surf = ax3d.plot_surface(F, L, Z, cmap="coolwarm", alpha=0.85,
                             edgecolor="none")
    f_star, lam_star = c, 1.0 / c
    z_star = np.log(f_star) - lam_star * (f_star - c)
    ax3d.scatter([f_star], [lam_star], [z_star], color="black", s=80,
                 marker="o", zorder=10)
    ax3d.text(f_star + 0.3, lam_star, z_star + 0.3,
              f"Saddle\n(f*={f_star}, λ*={lam_star:.3f})",
              fontsize=9, fontweight="bold")

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
            cp.Problem(obj, [f <= cap, A[ci, :] @ f == 0]).solve(solver=cp.ECOS, verbose=False)
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
