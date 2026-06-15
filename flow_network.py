import os

import cvxpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

FIGURE_DIR = os.path.join("Reports", "Project", "img")
os.makedirs(FIGURE_DIR, exist_ok=True)

_SAT_CMAP = plt.cm.YlOrRd
_TAB = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def fig_path(name):
    return os.path.join(FIGURE_DIR, name)


def jain(x):
    x = np.asarray(x, dtype=float)
    d = np.sum(x ** 2)
    return 0.0 if d <= 1e-9 else float(np.sum(x) ** 2 / (len(x) * d))


# ── graph drawing helpers ──────────────────────────────────────────────────

def _draw_nodes(ax, G, pos, source, sinks):
    internal = set(G.nodes()) - {source} - set(sinks)
    groups = [([source], "#9ecae1"), (sinks, "#fdae6b"), (list(internal), "#e0e0e0")]
    for nodelist, color in groups:
        if nodelist:
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=color,
                                   node_size=700, edgecolors="black",
                                   linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)


def _draw_edges(ax, G, pos, edgelist, widths, colors, labels=None):
    for i, (u, v) in enumerate(edgelist):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=widths[i],
                               edge_color=[colors[i]], ax=ax,
                               arrowstyle="-|>", arrowsize=18,
                               connectionstyle="arc3,rad=0.12")
    if labels:
        for (u, v), lbl in labels.items():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            dx, dy = x1 - x0, y1 - y0
            n = np.hypot(dx, dy) or 1.0
            ax.text(0.5 * (x0 + x1) - 0.06 * dy / n,
                    0.5 * (y0 + y1) + 0.06 * dx / n,
                    lbl, fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))


# ── main flow-network plot ─────────────────────────────────────────────────

def plot_flow_graph(
    G, edge_values, pos=None, title="Flow Network",
    S=None, T=None, cut_edges=None,
    show_saturation=False, save_path=None,
):
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    source = "s"
    sinks = [n for n in G.nodes() if n.startswith("t")]
    _draw_nodes(ax, G, pos, source, sinks)

    max_val = max(edge_values.values()) if edge_values else 1.0
    cut_set = set(cut_edges or [])
    widths, colors, labels = [], [], {}

    for u, v, d in G.edges(data=True):
        val = edge_values[(u, v)]
        widths.append(1 + 4 * val / max_val if max_val > 0 else 1)
        if show_saturation:
            sat = val / d["capacity"] if d["capacity"] > 0 else 0
            colors.append(_SAT_CMAP(min(sat, 1.0)))
        elif (u, v) in cut_set:
            colors.append("#d62728")
        else:
            colors.append("#999999")
        labels[(u, v)] = f"{val:.1f}/{d['capacity']}"

    _draw_edges(ax, G, pos, list(G.edges()), widths, colors, labels)

    if show_saturation:
        sm = plt.cm.ScalarMappable(cmap=_SAT_CMAP, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("flow / capacity", rotation=270, labelpad=15)

    if S is not None and T is not None:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label="Source side (S)",
                        markerfacecolor="#9ecae1", markersize=10, markeredgecolor="black"),
            plt.Line2D([0], [0], marker="o", color="w", label="Sink side (T)",
                        markerfacecolor="#fdae6b", markersize=10, markeredgecolor="black"),
            plt.Line2D([0], [0], color="#d62728", lw=3, label="Min-cut edge"),
            plt.Line2D([0], [0], color="#999999", lw=1, label="Other edge"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=9)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── fairness network ───────────────────────────────────────────────────────

def fairness_network():
    G = nx.DiGraph()
    arcs = [
        ("s", "v1", 11), ("s", "v2", 11),
        ("v1", "v3", 6), ("v1", "v4", 6),
        ("v2", "v4", 6), ("v2", "v5", 6),
        ("v4", "v3", 2),
        ("v3", "t1", 4), ("v4", "t1", 8),
        ("v4", "t2", 4), ("v5", "t2", 4),
        ("v5", "t3", 2), ("v4", "t3", 3),
    ]
    for u, v, c in arcs:
        G.add_edge(u, v, capacity=c)

    pos = {
        "s": (-2, 0), "v1": (-1, 1), "v2": (-1, -1),
        "v3": (0, 1),  "v4": (0, 0),  "v5": (0, -1),
        "t1": (1, 1),  "t2": (1, 0),  "t3": (1, -1),
    }

    source, sinks = "s", ["t1", "t2", "t3"]
    edges = list(G.edges())
    nodes = list(G.nodes())
    inc = np.asarray(nx.incidence_matrix(G, oriented=True).todense(), dtype=float)
    cap = np.array([G[u][v]["capacity"] for u, v in edges], dtype=float)

    internal = ~np.isin(nodes, [source, *sinks])
    A = inc[internal]
    B = inc[np.isin(nodes, sinks)]
    D = B @ cap
    return G, pos, source, sinks, edges, nodes, A, B, D, cap


# ── fairness drawing helpers ───────────────────────────────────────────────

def draw_saturation_graph(ax, G, pos, flows, cap_vec, source, sinks, edges,
                          title="", costs=None, show_edge_labels=True):
    _draw_nodes(ax, G, pos, source, sinks)

    max_flow = float(np.max(flows)) if np.max(flows) > 0 else 1.0
    widths = [1 + 4 * flows[j] / max_flow for j in range(len(edges))]
    colors = [_SAT_CMAP(min(flows[j] / cap_vec[j], 1.0) if cap_vec[j] > 0 else 0)
              for j in range(len(edges))]
    _draw_edges(ax, G, pos, edges, widths, colors)

    if show_edge_labels:
        for j, (u, v) in enumerate(edges):
            lbl = f"{flows[j]:.1f}/{cap_vec[j]:.0f}"
            if costs is not None:
                lbl += f"\nw={costs[j]:.1f}"
            x0, y0 = pos[u]; x1, y1 = pos[v]
            dx, dy = x1 - x0, y1 - y0
            n = np.hypot(dx, dy) or 1.0
            ax.text(0.5 * (x0 + x1) - 0.06 * dy / n,
                    0.5 * (y0 + y1) + 0.06 * dx / n,
                    lbl, fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))

    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_fairness_bars(ax, ax2, sink_x, sinks, s_sum, util, demand,
                       max_flow, max_util, title=""):
    bw = 0.35
    ax.bar(sink_x - bw / 2, s_sum, width=bw, color=_TAB[0],
           edgecolor="black", linewidth=0.5, label="sink flow")
    ax2.bar(sink_x + bw / 2, util, width=bw, color=_TAB[1],
            edgecolor="black", linewidth=0.5, label="utilization")
    ax.scatter(sink_x, demand, color=_TAB[3], marker="_", s=1200,
               linewidths=2, label="demand")
    ax.set_xticks(sink_x)
    ax.set_xticklabels(sinks)
    ax.set_ylim(0, max_flow * 1.1)
    ax2.set_ylim(0, max_util * 1.1)
    ax.set_ylabel("sink flow")
    ax2.set_ylabel("utilization")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def plot_fairness_summary(x, jain_arr, util_arr, flow_arr, sink_arr,
                          sinks, xlabel="alpha", best_flow=None,
                          fill_pof=False, suptitle=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, jain_arr, "s-", color=_TAB[2], lw=2, ms=6)
    axes[0, 0].set(xlabel=xlabel, ylabel="Jain index")
    axes[0, 0].grid(True, alpha=0.3)

    for k, sink in enumerate(sinks):
        axes[0, 1].plot(x, util_arr[:, k], "o-", lw=1.5, ms=4,
                        color=_TAB[k % 10], label=sink)
    axes[0, 1].set(xlabel=xlabel, ylabel="sink utilization")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x, flow_arr, "o-", color=_TAB[0], lw=2, ms=6)
    if fill_pof and best_flow is not None:
        axes[1, 0].fill_between(x, flow_arr, best_flow,
                                alpha=0.12, color=_TAB[3], label="PoF")
    axes[1, 0].set(xlabel=xlabel, ylabel="total throughput")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    for k, sink in enumerate(sinks):
        axes[1, 1].plot(x, sink_arr[:, k], "o-", lw=1.5, ms=4,
                        color=_TAB[k % 10], label=sink)
    axes[1, 1].set(xlabel=xlabel, ylabel="sink flow")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    plt.show()


# ── optimiser ──────────────────────────────────────────────────────────────

def solve_soft_cost(gamma, alpha=1.0, with_duals=False,
                    edges=None, A=None, B=None, D=None,
                    capacity_vector=None, costs=None,
                    solver=cvxpy.CLARABEL, verbose=False):
    f = cvxpy.Variable(len(edges))
    S = B @ f
    eps = 1e-9

    if alpha == 0:
        U = cvxpy.sum(D @ S)
    elif alpha == 1:
        U = cvxpy.sum(D @ cvxpy.log(S + eps))
    elif alpha < 1:
        U = cvxpy.sum(D @ cvxpy.power(S + eps, 1 - alpha) / (1 - alpha))
    else:
        U = cvxpy.sum(D @ (-cvxpy.power(S + eps, 1 - alpha) / (alpha - 1)))

    obj = cvxpy.Maximize(U - gamma * (costs @ f))
    c_nonneg = f >= 0
    c_cap = f <= capacity_vector
    c_demand = S <= D
    c_cons = A @ f == 0
    prob = cvxpy.Problem(obj, [c_nonneg, c_cap, c_demand, c_cons])
    prob.solve(solver=solver, verbose=verbose)

    flows = np.asarray(f.value, dtype=float) if f.value is not None else np.zeros(len(edges))
    s_sum = B @ flows
    row = {
        "gamma": float(gamma), "alpha": float(alpha),
        "flows": flows, "s_sum": s_sum, "util": s_sum / D,
        "total_flow": float(np.sum(s_sum)),
        "transport_cost": float(costs @ flows),
        "jain": jain(s_sum / D),
        "objective": float(prob.value),
    }
    if with_duals:
        row["rho"] = np.asarray(c_nonneg.dual_value, dtype=float)
        row["lambda"] = np.asarray(c_cap.dual_value, dtype=float)
        row["eta"] = np.asarray(c_demand.dual_value, dtype=float)
        row["mu"] = np.asarray(c_cons.dual_value, dtype=float)
    return row