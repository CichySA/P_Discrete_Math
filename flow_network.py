import networkx as nx
import matplotlib.pyplot as plt


def draw_network(G, flow_dict=None, source="s", sink="t"):
    pos = {
        "s": (0, 0),
        "a": (1, 1),
        "b": (1, -1),
        "c": (2, 1),
        "d": (2, -1),
        "t": (3, 0),
    }

    plt.figure(figsize=(8, 4))
    nx.draw(G, pos, with_labels=True, node_size=900, node_color="#b3cde3")

    # Draw separate labels for flow (red) and capacity (black) near the edge midpoint
    for u, v, data in G.edges(data=True):
        cap = data.get("capacity", 0)
        flow = 0
        if flow_dict:
            flow = flow_dict.get(u, {}).get(v, 0)

        # Draw flow (red) slightly before midpoint
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): str(flow)},
            label_pos=0.42,
            font_color="red",
        )

        # Draw capacity (black) slightly after midpoint
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): str(cap)},
            label_pos=0.58,
            font_color="black",
        )

    plt.title("Wikipedia flow network — flow (red) / capacity (black)")
    plt.axis("off")
    plt.show()