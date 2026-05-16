from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import networkx as nx


def plot_flow_graph(
    G: nx.DiGraph,
    edge_values: dict,
    pos: dict = None,
    title: str = "Flow Network",
    S: set = None,
    T: set = None,
    cut_edges: list = None,
    show_saturation: bool = False,
):
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    cut_edges = set(cut_edges or [])

    if S is None or T is None:
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=800,
            node_color="white",
            node_shape="o",
            alpha=1.0,
            cmap=None,
            vmin=None,
            vmax=None,
            ax=ax,
            linewidths=1.0,
            edgecolors="black",
        )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=S,
            node_size=800,
            node_color="#97cdff",
            node_shape="o",
            alpha=1.0,
            cmap=None,
            vmin=None,
            vmax=None,
            ax=ax,
            linewidths=1.0,
            edgecolors="black",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=T,
            node_size=800,
            node_color="#ffa0ff",
            node_shape="o",
            alpha=1.0,
            cmap=None,
            vmin=None,
            vmax=None,
            ax=ax,
            linewidths=1.0,
            edgecolors="black",
        )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_color="black",
        font_family="sans-serif",
        font_weight="normal",
        alpha=1.0,
        bbox=None,
        ax=ax,
        verticalalignment="center",
        horizontalalignment="center",
        clip_on=True,
    )

    max_edge_value = max(edge_values.values()) if edge_values else 0
    saturation_cmap = LinearSegmentedColormap.from_list(
        "saturation",
        ["#00ff00", "#fffb00", "#FF0000"],
    )

    for u, v, d in G.edges(data=True):
        edge_value = edge_values[(u, v)]
        line_width = 1 + 4 * edge_value / max_edge_value if max_edge_value > 0 else 1

        if show_saturation:
            saturation = edge_value / d["capacity"] if d["capacity"] > 0 else 0
            arc_color = saturation_cmap(saturation)
        elif (u, v) in cut_edges:
            arc_color = "red"
        else:
            arc_color = "gray"

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=line_width,
            edge_color=[arc_color],
            ax=ax,
            alpha=1,
            arrowstyle="-|>",
            arrowsize=20,
            connectionstyle="arc3,rad=0.15",
        )

        arc_label = f"{edge_value:.1f}/{d['capacity']}"
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): arc_label},
            label_pos=0.5,
            font_size=10,
            font_color="black",
            font_family="sans-serif",
            font_weight="normal",
            alpha=1.0,
            bbox=None,
            ax=ax,
            rotate=False,
            clip_on=True,
            verticalalignment="center",
            horizontalalignment="center",
            connectionstyle="arc3",
            hide_ticks=True,
        )

    if show_saturation:
        sm = plt.cm.ScalarMappable(cmap=saturation_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("Arc Saturation (Flow/Capacity)", rotation=270, labelpad=15)

    if S is not None and T is not None:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="S set (source side)",
                markerfacecolor="#97cdff",
                markersize=10,
                markeredgecolor="black",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="T set (sink side)",
                markerfacecolor="#ffa0ff",
                markersize=10,
                markeredgecolor="black",
            ),
            plt.Line2D([0], [0], color="red", lw=3, label="Min-cut Edge"),
            plt.Line2D([0], [0], color="gray", lw=1, label="Non Min-cut Edge"),
        ]
        plt.legend(handles=legend_elements, loc="upper left")

    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_network(G: nx.DiGraph, flow_dict: dict, pos: dict = None, title: str = "Flow Network"):
    plot_flow_graph(
        G,
        flow_dict,
        pos=pos,
        title=title,
        show_saturation=True,
    )


def plot_min_cut(G, S, T, min_cut, flow_dict, pos=None, title="Max-flow Min-cut"):
    plot_flow_graph(
        G,
        flow_dict,
        pos=pos,
        title=title,
        S=S,
        T=T,
        cut_edges=min_cut,
    )


def plot_cvxpy_min_cut(G, edge_dict, pos=None, source="s", title="CVXPY Min-cut"):
    cvxpy_min_cut = [
        (u, v)
        for (u, v), value in edge_dict.items()
        if value is not None and value > 0.5
    ]
    graph_without_cut = G.copy()
    graph_without_cut.remove_edges_from(cvxpy_min_cut)
    S = nx.descendants(graph_without_cut, source) | {source}
    T = set(G.nodes()) - S

    plot_flow_graph(
        G,
        edge_dict,
        pos=pos,
        title=title,
        S=S,
        T=T,
        cut_edges=cvxpy_min_cut,
    )
    return S, T, cvxpy_min_cut
