import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from PIL import Image
import io
import os


def create_coarsening_gif_fast(
    Gall,
    iCs,
    save_path="coarsening_evolution.gif",
    duration=1500,
    node_size_base=30,
    alpha=0.8,
    figsize=(10, 8),
    max_nodes_display=1000,
    dpi=80,
):
    """
    Fast version of GIF creation with optimizations for speed.

    Optimizations:
    - Lower DPI for faster rendering
    - Skip NetworkX for edge drawing
    - Cache computations
    - Vectorized operations
    - Simplified styling
    """

    print(f"Creating FAST GIF with {len(Gall)} levels...")

    # Simplified color palette
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
    ]

    frames = []

    # Fast layout computation
    print("Computing layout (fast mode)...")
    global_pos = get_fast_layout(Gall[0])

    # Cache axis limits
    if global_pos:
        x_coords = [global_pos[i][0] for i in global_pos.keys()]
        y_coords = [global_pos[i][1] for i in global_pos.keys()]
        margin = 0.1
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        xlim = (min(x_coords) - margin * x_range, max(x_coords) + margin * x_range)
        ylim = (min(y_coords) - margin * y_range, max(y_coords) + margin * y_range)
    else:
        xlim, ylim = (-1, 1), (-1, 1)

    for level in range(len(Gall)):
        print(f"Processing level {level}/{len(Gall)-1}")

        G = Gall[level]

        # Sample nodes for large graphs
        if hasattr(G, "num_nodes"):
            total_nodes = G.num_nodes
        elif hasattr(G, "N"):
            total_nodes = G.N
        else:
            total_nodes = len(global_pos)

        if total_nodes > max_nodes_display:
            # Sample nodes
            node_indices = np.random.choice(
                total_nodes, max_nodes_display, replace=False
            )
            node_indices = np.sort(node_indices)
        else:
            node_indices = np.arange(total_nodes)

        # Create frame quickly
        fig, ax = plt.subplots(figsize=figsize, facecolor="white", dpi=dpi)
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Get positions for this level (fast)
        if level == 0:
            pos = {i: global_pos[i] for i in node_indices if i in global_pos}
        else:
            pos = compute_fast_positions(G, iCs, level, global_pos, node_indices)

        # Fast edge drawing (skip for speed if too many)
        if (
            hasattr(G, "edge_index")
            and G.edge_index.numel() > 0
            and G.edge_index.shape[1] < 5000
        ):
            draw_edges_fast(G.edge_index, pos, ax, node_indices)

        # Fast node styling and drawing
        if pos:
            x_pos = np.array([pos[i][0] for i in pos.keys()])
            y_pos = np.array([pos[i][1] for i in pos.keys()])

            # Simple coloring based on clustering
            colors, sizes = get_fast_styling(
                level,
                iCs,
                list(pos.keys()),
                cluster_colors,
                node_size_base,
                total_nodes,
            )

            ax.scatter(
                x_pos,
                y_pos,
                c=colors,
                s=sizes,
                alpha=alpha,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

        # Simple title
        reduction_ratio = len(global_pos) / total_nodes if total_nodes > 0 else 1
        title = f"Level {level} | Nodes: {total_nodes:,} | Reduction: {reduction_ratio:.1f}x"
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Fast save
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.1,
        )
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    # Save GIF with optimization
    print(f"Saving GIF...")
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        print(f"FAST GIF saved! {len(frames)} frames")

    return save_path


def get_fast_layout(G0):
    """Fast layout computation with caching"""
    try:
        # Use existing positions if available
        if hasattr(G0, "pos") and G0.pos is not None:
            if isinstance(G0.pos, torch.Tensor):
                pos_array = G0.pos.cpu().numpy()
            else:
                pos_array = G0.pos
            return {i: pos_array[i] for i in range(len(pos_array))}

        # Fast layout using reduced iterations
        if hasattr(G0, "edge_index"):
            num_nodes = G0.num_nodes
            edge_index = G0.edge_index.cpu().numpy()

            # Create simple spring layout with fewer iterations
            if num_nodes > 500:
                # For large graphs, use random layout
                pos = {i: np.random.randn(2) * 2 for i in range(num_nodes)}
            else:
                # Quick NetworkX layout
                nx_graph = nx.Graph()
                nx_graph.add_nodes_from(range(num_nodes))
                edges = [
                    (edge_index[0, i], edge_index[1, i])
                    for i in range(edge_index.shape[1])
                ]
                nx_graph.add_edges_from(edges)
                pos = nx.spring_layout(
                    nx_graph, k=1, iterations=20, seed=42
                )  # Reduced iterations

            return pos
        else:
            # Fallback
            num_nodes = G0.N if hasattr(G0, "N") else 100
            return {i: np.random.randn(2) for i in range(num_nodes)}

    except Exception as e:
        print(f"Warning: Using random layout: {e}")
        num_nodes = G0.num_nodes if hasattr(G0, "num_nodes") else 100
        return {i: np.random.randn(2) for i in range(num_nodes)}


def draw_edges_fast(edge_index, pos, ax, node_indices=None):
    """Fast edge drawing without NetworkX"""
    try:
        edge_index = edge_index.cpu().numpy()

        # Filter edges to only those between visible nodes
        if node_indices is not None:
            node_set = set(node_indices)
            valid_edges = []
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src in node_set and dst in node_set and src in pos and dst in pos:
                    valid_edges.append((src, dst))
        else:
            valid_edges = [
                (edge_index[0, i], edge_index[1, i])
                for i in range(edge_index.shape[1])
                if edge_index[0, i] in pos and edge_index[1, i] in pos
            ]

        # Limit edges for performance
        if len(valid_edges) > 2000:
            valid_edges = valid_edges[:: len(valid_edges) // 2000]

        # Draw edges as line collection (faster)
        lines = []
        for src, dst in valid_edges:
            lines.append([pos[src], pos[dst]])

        if lines:
            from matplotlib.collections import LineCollection

            lc = LineCollection(lines, colors="gray", alpha=0.2, linewidths=0.3)
            ax.add_collection(lc)

    except Exception as e:
        print(f"Warning: Could not draw edges: {e}")


def compute_fast_positions(G, iCs, level, global_pos, node_indices):
    """Fast position computation without recursion"""
    try:
        if level == 0 or level > len(iCs):
            return {i: global_pos[i] for i in node_indices if i in global_pos}

        pos = {}
        iC = iCs[level - 1]

        # Convert to numpy
        if isinstance(iC, torch.Tensor):
            iC_array = (
                iC.to_dense().cpu().numpy()
                if hasattr(iC, "is_sparse") and iC.is_sparse
                else iC.cpu().numpy()
            )
        else:
            iC_array = iC.toarray() if hasattr(iC, "toarray") else iC

        # Simple position computation
        for node_idx in node_indices:
            if node_idx < iC_array.shape[0]:
                # Find contributing fine nodes
                fine_nodes = np.where(iC_array[node_idx, :] > 0)[0]

                if len(fine_nodes) > 0:
                    # Simple average (no complex weighting for speed)
                    positions = []
                    for fine_node in fine_nodes:
                        if fine_node in global_pos:
                            positions.append(global_pos[fine_node])

                    if positions:
                        pos[node_idx] = np.mean(positions, axis=0)
                    else:
                        pos[node_idx] = np.random.randn(2) * 0.1
                else:
                    pos[node_idx] = np.random.randn(2) * 0.1

        return pos

    except Exception as e:
        print(f"Warning: Using simple layout for level {level}: {e}")
        return {i: np.random.randn(2) for i in node_indices}


def get_fast_styling(level, iCs, node_list, colors, base_size, total_nodes):
    """Fast node styling without complex computations"""
    n_nodes = len(node_list)
    node_colors = ["#74b9ff"] * n_nodes  # Default blue
    node_sizes = [base_size] * n_nodes

    try:
        if level > 0 and level - 1 < len(iCs):
            iC = iCs[level - 1]

            # Convert to numpy
            if isinstance(iC, torch.Tensor):
                iC_array = (
                    iC.to_dense().cpu().numpy()
                    if hasattr(iC, "is_sparse") and iC.is_sparse
                    else iC.cpu().numpy()
                )
            else:
                iC_array = iC.toarray() if hasattr(iC, "toarray") else iC

            # Simple coloring
            for i, node_idx in enumerate(node_list):
                if node_idx < iC_array.shape[0]:
                    fine_nodes = np.where(iC_array[node_idx, :] > 0)[0]
                    cluster_size = len(fine_nodes)

                    if cluster_size > 1:
                        color_idx = node_idx % len(colors)
                        node_colors[i] = colors[color_idx]
                        node_sizes[i] = base_size * min(1.5 + 0.1 * cluster_size, 3)

    except Exception as e:
        print(f"Warning: Using default styling: {e}")

    return node_colors, node_sizes


def create_coarsening_gif(
    Gall,
    iCs,
    save_path="coarsening_evolution.gif",
    duration=1500,
    node_size_base=30,
    alpha=0.8,
    figsize=(10, 8),
    fast_mode="auto",
):
    """
    Create a GIF showing the evolution of graph coarsening levels.

    Automatically chooses between fast and high-quality mode based on graph size.

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
    fast_mode : str or bool
        "auto": automatically choose based on graph size
        True: always use fast mode
        False: use high-quality mode
    """

    if not Gall:
        print("Error: No graphs provided")
        return None

    # Determine mode
    if fast_mode == "auto":
        total_nodes = (
            Gall[0].num_nodes
            if hasattr(Gall[0], "num_nodes")
            else (Gall[0].N if hasattr(Gall[0], "N") else 100)
        )
        use_fast = total_nodes > 500 or len(Gall) > 10
    else:
        use_fast = bool(fast_mode)

    if use_fast:
        print("Using FAST mode for better performance")
        return create_coarsening_gif_fast(
            Gall, iCs, save_path, duration, node_size_base, alpha, figsize
        )
    else:
        print("Using HIGH-QUALITY mode")
        return create_coarsening_gif_quality(
            Gall, iCs, save_path, duration, node_size_base, alpha, figsize
        )


def create_coarsening_gif_quality(
    Gall,
    iCs,
    save_path="coarsening_evolution.gif",
    duration=2000,
    node_size_base=50,
    alpha=0.8,
    figsize=(12, 10),
):
    """
    High-quality version with full features (slower but prettier)
    """

    print(f"Creating HIGH-QUALITY GIF with {len(Gall)} levels...")

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
    global_pos = get_consistent_layout(Gall[0])

    for level in range(len(Gall)):
        print(f"Processing level {level}/{len(Gall)-1}")

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

        G = Gall[level]

        if level == 0:
            pos = global_pos
        else:
            pos = compute_coarsened_positions(Gall, iCs, level, global_pos)

        nx_graph = get_networkx_graph(G)

        if nx_graph.number_of_edges() > 0:
            nx.draw_networkx_edges(
                nx_graph, pos, ax=ax, alpha=0.2, width=0.5, edge_color="#7f8c8d"
            )

        node_colors, node_sizes = get_cluster_styling(
            level, iCs, pos, cluster_colors, node_size_base
        )

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

        num_nodes = len(pos)
        num_edges = nx_graph.number_of_edges()

        if level == 0:
            title = f"Original Graph\nNodes: {num_nodes:,}, Edges: {num_edges:,}"
        else:
            reduction_ratio = len(global_pos) / num_nodes if num_nodes > 0 else 1
            title = f"Coarsening Level {level}\nNodes: {num_nodes:,}, Edges: {num_edges:,}\nReduction: {reduction_ratio:.1f}x"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        if level > 0 and level <= 3:
            add_simple_legend(ax, G)

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

    print(f"Saving GIF...")
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        print(f"HIGH-QUALITY GIF saved! {len(frames)} frames")

    return save_path


def get_consistent_layout(G):
    """Get a consistent layout for the original graph"""
    try:
        # Try to use existing positions
        if hasattr(G, "pos") and G.pos is not None:
            if isinstance(G.pos, torch.Tensor):
                pos_array = G.pos.cpu().numpy()
            else:
                pos_array = G.pos
            return {i: pos_array[i] for i in range(len(pos_array))}

        # Create NetworkX graph and compute layout
        nx_graph = get_networkx_graph(G)
        pos = nx.spring_layout(nx_graph, k=3, iterations=100, seed=42)
        return pos

    except Exception as e:
        print(f"Warning: Could not get consistent layout: {e}")
        # Fallback to random layout
        num_nodes = (
            G.num_nodes
            if hasattr(G, "num_nodes")
            else (G.N if hasattr(G, "N") else 100)
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
