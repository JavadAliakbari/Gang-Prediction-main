import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from PIL import Image
import io
import os


def create_coarsening_gif(
    Gall,
    iCs,
    save_path="coarsening_evolution.gif",
    duration=2000,
    node_size_base=50,
    alpha=0.8,
    figsize=(14, 10),
):
    """
    Create a GIF showing the evolution of graph coarsening levels.

    This improved version:
    - Clears the background between frames
    - Shows smooth transitions where nodes gradually merge
    - Highlights which nodes are being clustered together
    - Makes the graph progressively smaller while maintaining visual continuity

    Parameters:
    -----------
    Gall : list
        List of graphs at each coarsening level
    iCs : list
        List of coarsening matrices (mapping from level i to i+1)
    save_path : str
        Path to save the GIF file
    duration : int
        Duration of each frame in milliseconds
    node_size_base : int
        Base size for nodes
    alpha : float
        Transparency level
    figsize : tuple
        Figure size
    """

    print(f"Creating GIF with {len(Gall)} levels...")

    # Enhanced color palette for cluster visualization
    cluster_colors = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#f1c40f",
        "#e91e63",
        "#ff9800",
        "#4caf50",
        "#2196f3",
        "#673ab7",
        "#795548",
    ]

    frames = []

    # Determine consistent layout using the original graph
    print("Computing consistent layout...")
    global_pos = get_consistent_layout(Gall[0])

    for level in range(len(Gall)):
        print(f"Processing level {level}/{len(Gall)-1}")

        # Create the main frame for this level
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")
        ax.clear()  # Clear any previous plots
        ax.set_aspect("equal")
        ax.axis("off")

        G = Gall[level]

        # Get positions for this level
        if level == 0:
            pos = global_pos
        else:
            pos = compute_coarsened_positions(Gall, iCs, level, global_pos)

        # Get networkx graph for plotting
        nx_graph = get_networkx_graph(G)

        # Draw edges with subtle styling
        if nx_graph.number_of_edges() > 0:
            nx.draw_networkx_edges(
                nx_graph, pos, ax=ax, alpha=0.2, width=0.5, edge_color="#7f8c8d"
            )

        # Get node styling based on clustering
        node_colors, node_sizes = get_cluster_styling(
            level, iCs, pos, cluster_colors, node_size_base
        )

        # Draw nodes
        if pos:
            nodes_list = list(pos.keys())
            node_pos = [pos[n] for n in nodes_list]
            node_colors_list = [node_colors.get(n, "#74b9ff") for n in nodes_list]
            node_sizes_list = [node_sizes.get(n, node_size_base) for n in nodes_list]

            ax.scatter(
                [p[0] for p in node_pos],
                [p[1] for p in node_pos],
                c=node_colors_list,
                s=node_sizes_list,
                alpha=alpha,
                edgecolors="black",
                linewidths=1.2,
                zorder=3,
            )

        # Add title with level information
        num_nodes = len(pos)
        num_edges = nx_graph.number_of_edges()

        if level == 0:
            title = f"Original Graph\nNodes: {num_nodes:,}, Edges: {num_edges:,}"
        else:
            reduction_ratio = len(global_pos) / num_nodes if num_nodes > 0 else 1
            title = f"Coarsening Level {level}\nNodes: {num_nodes:,}, Edges: {num_edges:,}\nReduction: {reduction_ratio:.1f}x"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Add legend for the first few levels
        if level > 0 and level <= 3:
            add_simple_legend(ax)

        # Set consistent axis limits with margin
        if pos:
            x_coords = [pos[i][0] for i in pos.keys()]
            y_coords = [pos[i][1] for i in pos.keys()]
            margin = 0.15
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            ax.set_xlim(
                min(x_coords) - margin * x_range, max(x_coords) + margin * x_range
            )
            ax.set_ylim(
                min(y_coords) - margin * y_range, max(y_coords) + margin * y_range
            )

        plt.tight_layout()

        # Save frame to memory
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=120,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    # Create and save GIF
    print(f"Saving GIF to {save_path}...")
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        print(f"GIF saved successfully! {len(frames)} frames, {duration}ms per frame")
    else:
        print("Error: No frames created!")

    return save_path


def get_consistent_layout(G0):
    """Get a consistent layout for the original graph"""
    try:
        # Try to use existing positions
        if hasattr(G0, "pos") and G0.pos is not None:
            if isinstance(G0.pos, torch.Tensor):
                pos_array = G0.pos.cpu().numpy()
            else:
                pos_array = G0.pos
            return {i: pos_array[i] for i in range(len(pos_array))}

        # Create NetworkX graph and compute layout
        nx_graph = get_networkx_graph(G0)
        pos = nx.spring_layout(nx_graph, k=3, iterations=100, seed=42)
        return pos

    except Exception as e:
        print(f"Warning: Could not get consistent layout: {e}")
        # Fallback to random layout
        num_nodes = (
            G0.num_nodes
            if hasattr(G0, "num_nodes")
            else (G0.N if hasattr(G0, "N") else 100)
        )
        return {i: np.random.randn(2) for i in range(num_nodes)}


def get_networkx_graph(G):
    """Convert graph to NetworkX format"""
    try:
        if hasattr(G, "edge_index"):
            # PyTorch Geometric graph
            nx_graph = nx.Graph()
            num_nodes = G.num_nodes if hasattr(G, "num_nodes") else G.x.shape[0]
            nx_graph.add_nodes_from(range(num_nodes))

            if G.edge_index.numel() > 0:
                edge_index = G.edge_index.cpu().numpy()
                edges = [
                    (edge_index[0, i], edge_index[1, i])
                    for i in range(edge_index.shape[1])
                ]
                nx_graph.add_edges_from(edges)

            return nx_graph
        else:
            # PyGSP or similar graph
            if hasattr(G, "W"):
                return nx.Graph(G.W.toarray())
            else:
                return nx.Graph()
    except Exception as e:
        print(f"Warning: Could not convert to NetworkX: {e}")
        return nx.Graph()


def compute_coarsened_positions(Gall, iCs, level, global_pos):
    """Compute positions for coarsened nodes based on their constituent fine nodes"""
    if level == 0 or level > len(iCs):
        return global_pos

    pos = {}

    try:
        # Get the immediate coarsening matrix for this level
        iC = iCs[level - 1]

        # Convert to numpy array
        if isinstance(iC, torch.Tensor):
            if hasattr(iC, "is_sparse") and iC.is_sparse:
                iC_array = iC.to_dense().cpu().numpy()
            else:
                iC_array = iC.cpu().numpy()
        else:
            iC_array = iC.toarray() if hasattr(iC, "toarray") else iC

        # For each coarsened node, compute position as weighted average
        num_coarse_nodes = iC_array.shape[0]
        for coarse_idx in range(num_coarse_nodes):
            fine_nodes = np.where(iC_array[coarse_idx, :] > 0)[0]

            if len(fine_nodes) > 0:
                # Get positions of contributing fine nodes
                contributing_positions = []
                weights = []

                for fine_node in fine_nodes:
                    if level == 1:
                        # Direct mapping from original graph
                        if fine_node in global_pos:
                            contributing_positions.append(global_pos[fine_node])
                            weights.append(iC_array[coarse_idx, fine_node])
                    else:
                        # Need to trace back through previous levels
                        # For simplicity, use the position from previous level
                        prev_pos = compute_coarsened_positions(
                            Gall, iCs, level - 1, global_pos
                        )
                        if fine_node in prev_pos:
                            contributing_positions.append(prev_pos[fine_node])
                            weights.append(iC_array[coarse_idx, fine_node])

                if contributing_positions:
                    # Weighted average of positions
                    weights = np.array(weights)
                    positions = np.array(contributing_positions)
                    pos[coarse_idx] = np.average(positions, axis=0, weights=weights)
                else:
                    # Fallback position
                    pos[coarse_idx] = np.random.randn(2) * 0.1
            else:
                pos[coarse_idx] = np.random.randn(2) * 0.1

    except Exception as e:
        print(f"Warning: Could not compute coarsened positions for level {level}: {e}")
        # Fallback: create spring layout for current graph
        nx_graph = get_networkx_graph(Gall[level])
        pos = nx.spring_layout(nx_graph, seed=42)

    return pos


def get_cluster_styling(level, iCs, pos, cluster_colors, node_size_base):
    """Get node colors and sizes based on clustering information"""
    node_colors = {}
    node_sizes = {}

    # Default styling
    for node in pos.keys():
        node_colors[node] = "#74b9ff"  # Light blue
        node_sizes[node] = node_size_base

    # Apply clustering-based styling
    if level > 0 and level - 1 < len(iCs):
        try:
            iC = iCs[level - 1]

            # Convert to numpy array
            if isinstance(iC, torch.Tensor):
                if hasattr(iC, "is_sparse") and iC.is_sparse:
                    iC_array = iC.to_dense().cpu().numpy()
                else:
                    iC_array = iC.cpu().numpy()
            else:
                iC_array = iC.toarray() if hasattr(iC, "toarray") else iC

            # Color nodes based on which fine nodes they represent
            for coarse_idx in range(iC_array.shape[0]):
                if coarse_idx in pos:  # Only style nodes that exist in current level
                    fine_nodes = np.where(iC_array[coarse_idx, :] > 0)[0]
                    cluster_size = len(fine_nodes)

                    if cluster_size > 1:
                        # Use cluster color for nodes that represent multiple fine nodes
                        color = cluster_colors[coarse_idx % len(cluster_colors)]
                        size = node_size_base * (1.2 + 0.4 * min(cluster_size, 5))

                        node_colors[coarse_idx] = color
                        node_sizes[coarse_idx] = size

        except Exception as e:
            print(f"Warning: Could not apply cluster styling for level {level}: {e}")

    return node_colors, node_sizes


def add_simple_legend(ax):
    """Add a simple legend for the visualization"""
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#74b9ff",
            markersize=8,
            label="Single nodes",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=12,
            label="Clustered nodes",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=10)


def create_coarsening_gif_efficient(
    Gall,
    iCs,
    save_path="coarsening_evolution_efficient.gif",
    duration=1500,
    max_nodes_display=300,
    node_size_base=20,
):
    """
    Efficient version for very large graphs that samples nodes for display
    """
    print(
        f"Creating efficient GIF with {len(Gall)} levels (max {max_nodes_display} nodes per frame)..."
    )

    frames = []
    cluster_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for level in range(len(Gall)):
        print(f"Processing level {level}/{len(Gall)-1}")

        G = Gall[level]

        # Determine how many nodes to show
        if hasattr(G, "num_nodes"):
            total_nodes = G.num_nodes
        elif hasattr(G, "N"):
            total_nodes = G.N
        else:
            total_nodes = len(G.pos) if hasattr(G, "pos") else 100

        # Sample nodes if graph is too large
        if total_nodes > max_nodes_display:
            node_indices = np.random.choice(
                total_nodes, max_nodes_display, replace=False
            )
            node_indices = np.sort(node_indices)
        else:
            node_indices = np.arange(total_nodes)

        fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")

        # Get positions
        if hasattr(G, "pos") and G.pos is not None:
            if isinstance(G.pos, torch.Tensor):
                pos_array = G.pos.cpu().numpy()
            else:
                pos_array = G.pos

            pos_sampled = pos_array[node_indices]

            # Simple node coloring
            colors = ["#74b9ff"] * len(node_indices)
            sizes = [node_size_base] * len(node_indices)

            # Highlight clustered nodes if we have coarsening information
            if level > 0 and level - 1 < len(iCs):
                try:
                    iC = iCs[level - 1]
                    if isinstance(iC, torch.Tensor):
                        iC_array = (
                            iC.to_dense().cpu().numpy()
                            if hasattr(iC, "is_sparse") and iC.is_sparse
                            else iC.cpu().numpy()
                        )
                    else:
                        iC_array = iC.toarray() if hasattr(iC, "toarray") else iC

                    for i, node_idx in enumerate(node_indices):
                        if node_idx < iC_array.shape[1]:
                            coarse_assignments = np.where(iC_array[:, node_idx] > 0)[0]
                            if len(coarse_assignments) > 0:
                                coarse_idx = coarse_assignments[0]
                                cluster_size = np.sum(iC_array[coarse_idx, :] > 0)
                                if cluster_size > 1:
                                    colors[i] = cluster_colors[
                                        coarse_idx % len(cluster_colors)
                                    ]
                                    sizes[i] = node_size_base * 1.5
                except Exception as e:
                    print(f"Warning: Could not apply clustering colors: {e}")

            # Plot nodes
            ax.scatter(
                pos_sampled[:, 0],
                pos_sampled[:, 1],
                c=colors,
                s=sizes,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

        # Add title
        if total_nodes > max_nodes_display:
            title = f"Level {level} (showing {len(node_indices)}/{total_nodes} nodes)\nTotal nodes: {total_nodes:,}"
        else:
            title = f"Level {level}\nNodes: {total_nodes:,}"

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save frame
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    # Save GIF
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        print(f"Efficient GIF saved: {save_path}")

    return save_path


if __name__ == "__main__":
    print("Coarsening GIF creation utility ready!")
    print("Main functions:")
    print("- create_coarsening_gif(Gall, iCs): Full visualization with clustering")
    print(
        "- create_coarsening_gif_efficient(Gall, iCs): Memory-efficient for large graphs"
    )
