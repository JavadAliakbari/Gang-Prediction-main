import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor
from sortedcontainers import SortedList

from utils.utils import degree, graph_params, sparse_eye
import graph_utils
import maxWeightMatching


def coarsen(
    # G,
    # X=None, # WHT: node features
    Gc: Data,
    K=10,
    r=0.5,
    max_levels=10,
    method="variation_neighborhood",
    algorithm="greedy",
    Uk=None,
    lk=None,
    max_level_r=0.99,
    similarity_threshold=0.5,
):
    """
    This function provides a common interface for coarsening algorithms that contract subgraphs

    Parameters
    ----------
    G : pygsp Graph
    K : int
        The size of the subspace we are interested in preserving.
    r : float between (0,1)
        The desired reduction defined as 1 - n/N.
    method : String
        ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']

    Returns
    -------
    C : np.array of size n x N
        The coarsening matrix.
    Gc : pygsp Graph
        The smaller graph.
    Call : list of np.arrays
        Coarsening matrices for each level
    Gall : list of (n_levels+1) pygsp Graphs
        All graphs involved in the multilevel coarsening

    Example
    -------
    C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
    """
    r = np.clip(r, 0.0, max_level_r)
    N = Gc.num_nodes

    C = sparse_eye(N)
    Gc.W, Gc.L, Gc.dw = graph_params(Gc)

    B = calc_B(Gc, K, Uk, lk)
    iC = None

    Call, Gall = [], []
    Gall.append(Gc)

    for level in range(1, max_levels + 1):
        iC, Gc, B = coarse_one_level(
            Gc,
            iC,
            B,
            K=K,
            method=method,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            level=level,
            r_cur=r,
        )

        C = torch.sparse.mm(iC, C)
        Call.append(iC)
        Gall.append(Gc)

        if iC.shape[1] - iC.shape[0] <= 2:
            break  # avoid too many levels for so few nodes
        if Gc.num_nodes <= (1 - r) * N:
            break

    return C, Gc, Call, Gall


def coarse_one_level(
    G,
    iC,
    B,
    K=10,
    method="variation_neighborhoods",
    algorithm="greedy",
    similarity_threshold=0.5,
    level=1,
    r_cur=None,
):

    if "variation" in method:
        if level == 1:
            A = B
        else:
            B = torch.sparse.mm(iC, B)
            d, V = torch.linalg.eigh(B.T @ G.L @ B)
            mask = d == 0
            d[mask] = 1
            dinvsqrt = d ** (-1 / 2)
            dinvsqrt[mask] = 0
            A = B @ torch.diag(dinvsqrt) @ V

        if method == "variation_edges":
            coarsening_list = contract_variation_edges(
                G,
                K=K,
                A=A,
                r=r_cur,
                algorithm=algorithm,
                similarity_threshold=similarity_threshold,
            )
        else:
            coarsening_list = contract_variation_linear(
                G,
                A=A,
                r=r_cur,
                mode=method,
                similarity_threshold=similarity_threshold,
            )

    else:
        weights = get_proximity_measure(G, method, K=K)

        if algorithm == "optimal":
            # the edge-weight should be light at proximal edges
            weights = -weights
            if "rss" not in method:
                weights -= min(weights)
            coarsening_list = matching_optimal(G, weights=weights, r=r_cur)

        elif algorithm == "greedy":
            coarsening_list = matching_greedy(G, weights=weights, r=r_cur)

    iC = get_coarsening_matrix(coarsening_list, G.num_nodes)

    Gc = construct_G(G, iC)

    return iC, Gc, B


def calc_B(Gc, K, Uk=None, lk=None):
    if (Uk is not None) and (lk is not None) and (len(lk) >= K):
        mask = lk < 1e-10
        lk[mask] = 1
        lsinv = lk ** (-0.5)
        lsinv[mask] = 0
        B = Uk[:, :K] @ torch.diag(lsinv[:K])
    else:
        offset = 2 * max(Gc.dw)
        T = offset * sparse_eye(Gc.num_nodes) - Gc.L
        # lk, Uk = torch.linalg.eigh(T, k=K, which="LM", tol=1e-5)
        lk, Uk = torch.lobpcg(T, k=K, largest=True, tol=1e-5)
        lk = torch.flip(offset - lk, [0])
        Uk = torch.flip(Uk, [1])
        mask = lk < 1e-10
        lk[mask] = 1
        lsinv = lk ** (-0.5)
        lsinv[mask] = 0
        B = Uk @ torch.diag(lsinv)

    return B


################################################################################
# General coarsening utility functions
################################################################################


def construct_G(G: Data, iC: SparseTensor):
    Wc = graph_utils.zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
    Wc = (
        Wc + Wc.T
    ) / 2  # this is only needed to avoid pygsp complaining for tiny errors
    Wc = Wc.coalesce()  # Ensure Wc is in COO format
    indices = Wc.indices()
    values = Wc.values()
    num_nodes = Wc.size(0)
    x = torch.sparse.mm(iC, G.x) if G.x is not None else None
    Gc = Data(x=x, edge_index=indices, edge_weight=values, num_nodes=num_nodes)
    Gc.soft_y = torch.sparse.mm(iC, G.soft_y) if G.soft_y is not None else None
    Gc.W, Gc.L, Gc.dw = graph_params(Gc)

    embeddings = (
        torch.sparse.mm(iC, G.embeddings)
        if hasattr(G, "embeddings") and G.embeddings is not None
        else None
    )
    Gc.embeddings = embeddings
    Gc.embeddings = (
        F.normalize(Gc.embeddings, p=2, dim=1) if embeddings is not None else None
    )

    if hasattr(G, "pos") and G.pos is not None:
        # pos = torch.sparse.mm(iC, G.pos)
        Gc.pos = coarsen_vector(G.pos, iC)

    return Gc


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    C_sum = torch.sparse.sum(C, dim=0).to_dense()
    inv_C_sum = 1.0 / C_sum
    # Create sparse diagonal matrix
    indices = torch.arange(len(inv_C_sum)).repeat(2, 1)
    D = torch.sparse_coo_tensor(indices, inv_C_sum, (len(inv_C_sum), len(inv_C_sum)))
    Pinv = torch.sparse.mm(C, D).t()
    return torch.sparse.mm(torch.sparse.mm(Pinv.T, W), Pinv)
    # return (Pinv.T).dot(W.dot(Pinv))


def similarity(nodes_features):
    s = nodes_features @ nodes_features.T
    n = s.shape[0]
    return torch.sum(torch.triu(s, diagonal=1)) / ((n - 1) * n / 2)


def coarsen_vector(x, C):
    return (C * C) @ x


def lift_vector(x, C):
    # Pinv = C.T; Pinv[Pinv>0] = 1
    C_sum = torch.sparse.sum(C, dim=0).to_dense()
    inv_C_sum = 1.0 / C_sum
    # Create sparse diagonal matrix
    indices = torch.arange(len(inv_C_sum)).repeat(2, 1)
    D = torch.sparse_coo_tensor(indices, inv_C_sum, (len(inv_C_sum), len(inv_C_sum)))
    Pinv = torch.sparse.mm(C, D).t()
    return torch.sparse.mm(Pinv, x)


def lift_matrix(W, C):
    # Get the squared version of C (equivalent to C.power(2) in scipy)
    P = C * C  # Element-wise multiplication for sparse matrices in PyTorch

    # Perform matrix multiplications with sparse matrices
    return torch.sparse.mm(torch.sparse.mm(P.T, W), P)


def get_coarsening_matrix(partitioning, N):
    """
    Create coarsening matrix C using sparse tensor operations.

    Parameters:
    -----------
    G : PyTorch Geometric Data object
        The graph to be coarsened
    partitioning : list
        List of subgraphs to be contracted

    Returns:
    --------
    C : torch.sparse.Tensor
        The coarsening matrix
    """
    # Create sparse identity matrix
    C = sparse_eye(N)

    # Keep track of which rows to preserve
    rows_to_keep = torch.ones(N, dtype=torch.bool)

    # Process each subgraph
    for candidate in partitioning:
        subgraph = candidate["list"]
        nc = len(subgraph)
        if nc <= 1:
            continue

        # Convert subgraph to tensor if needed
        if not isinstance(subgraph, torch.Tensor):
            subgraph = torch.tensor(subgraph)

        # Mark rows to remove
        rows_to_keep[subgraph[1:]] = False

        # Get representative node
        rep_node = subgraph[0].item()

        # Create new entries for the representative node
        rep_indices = torch.stack(
            [
                torch.full((nc,), rep_node),  # Row indices (all rep_node)
                subgraph,  # Column indices
            ]
        )

        rep_values = torch.full(
            (nc,), 1.0 / torch.sqrt(torch.tensor(nc, dtype=torch.float32))
        )

        # Create sparse tensor for this update
        update = torch.sparse_coo_tensor(rep_indices, rep_values, (N, N))

        # Update the representative row
        # First zero out any existing entries in the row
        mask_indices = torch.tensor([[rep_node], [rep_node]])
        mask_values = torch.ones(1)
        mask = torch.sparse_coo_tensor(mask_indices, mask_values, (N, N))

        # Add the update to C
        C = C + update - mask

    # Keep only the rows that weren't contracted
    keep_indices = torch.nonzero(rows_to_keep).squeeze()

    # Extract the values and indices from C
    C = C.coalesce()  # Ensure C is in COO format
    indices = C.indices()
    values = C.values()

    # Filter rows that need to be kept
    mask = torch.isin(indices[0], keep_indices)
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]

    # Map old row indices to new indices
    row_map = torch.full((N,), -1, dtype=torch.long)
    row_map[keep_indices] = torch.arange(len(keep_indices))
    filtered_indices[0] = row_map[filtered_indices[0]]

    # Create the final sparse tensor
    C_final = torch.sparse_coo_tensor(
        filtered_indices, filtered_values, (len(keep_indices), N)
    )

    return C_final


def plot_coarsening(
    Gall, Call, size=3, edge_width=0.8, node_size=20, alpha=0.55, title=""
):
    """
    Plot a (hierarchical) coarsening evolution efficiently

    Parameters
    ----------
    Gall : list of torch_geometric.data.Data
        All graphs involved in the multilevel coarsening, from original to final
    Call : list of torch.sparse tensors
        Coarsening matrices for each level
    size : float
        Size multiplier for figure dimensions
    edge_width : float
        Width of edges in the plot
    node_size : int
        Base size of nodes
    alpha : float
        Transparency value
    title : str
        Title prefix for the plots

    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Colors signify the size of a coarsened subgraph
    colors = ["black", "green", "blue", "red", "orange", "purple", "brown", "pink"]

    n_levels = len(Gall) - 1
    if n_levels == 0:
        return None

    # Check if graphs have positions
    if not hasattr(Gall[0], "pos") or Gall[0].pos is None:
        print("Warning: No node positions found. Cannot plot graph coarsening.")
        return None

    # Determine if 2D or 3D
    pos_dim = Gall[0].pos.shape[1]
    if pos_dim not in [2, 3]:
        print(
            f"Warning: Unsupported position dimension {pos_dim}. Only 2D and 3D are supported."
        )
        return None

    # Create figure
    fig = plt.figure(figsize=(n_levels * size * 3, size * 2))

    # Plot each level
    for level in range(n_levels):
        G = Gall[level]
        Gc = Gall[level + 1]
        C = Call[level]

        # Create subplot
        if pos_dim == 2:
            ax = fig.add_subplot(1, n_levels + 1, level + 1)
        else:
            ax = fig.add_subplot(1, n_levels + 1, level + 1, projection="3d")

        ax.axis("off")
        ax.set_title(f"{title} | level = {level}, N = {G.num_nodes}")

        # Get positions
        pos = G.pos.cpu().numpy()

        # Plot edges efficiently
        if G.edge_index.shape[1] > 0:
            edge_coords = pos[G.edge_index.cpu().numpy()]

            if pos_dim == 2:
                # Plot all edges at once using LineCollection for efficiency
                from matplotlib.collections import LineCollection

                lines = LineCollection(
                    edge_coords.transpose(1, 0, 2),
                    colors="black",
                    alpha=alpha,
                    linewidths=edge_width,
                )
                ax.add_collection(lines)
            else:
                # For 3D, we need to plot edges one by one (matplotlib limitation)
                for edge_coord in edge_coords.transpose(1, 0, 2):
                    ax.plot(
                        *edge_coord.T, color="black", alpha=alpha, linewidth=edge_width
                    )

        # Plot nodes colored by coarsening groups efficiently
        C_dense = C.coalesce()
        C_indices = C_dense.indices().cpu().numpy()
        C_values = C_dense.values().cpu().numpy()

        # Group nodes by which coarse node they belong to
        node_groups = {}
        for coarse_idx, fine_idx in zip(C_indices[0], C_indices[1]):
            if coarse_idx not in node_groups:
                node_groups[coarse_idx] = []
            node_groups[coarse_idx].append(fine_idx)

        # Plot each group with appropriate color and size
        for coarse_idx, fine_nodes in node_groups.items():
            fine_nodes = np.array(fine_nodes)
            group_size = len(fine_nodes)
            color_idx = min(group_size - 1, len(colors) - 1)
            color = colors[color_idx]
            size = node_size * group_size

            node_pos = pos[fine_nodes]
            if pos_dim == 2:
                ax.scatter(node_pos[:, 0], node_pos[:, 1], c=color, s=size, alpha=alpha)
            else:
                ax.scatter(
                    node_pos[:, 0],
                    node_pos[:, 1],
                    node_pos[:, 2],
                    c=color,
                    s=size,
                    alpha=alpha,
                )

        # Set equal aspect ratio
        if pos_dim == 2:
            ax.set_aspect("equal")
            # Set axis limits based on node positions
            margin = 0.1
            pos_range = pos.max(axis=0) - pos.min(axis=0)
            ax.set_xlim(
                pos[:, 0].min() - margin * pos_range[0],
                pos[:, 0].max() + margin * pos_range[0],
            )
            ax.set_ylim(
                pos[:, 1].min() - margin * pos_range[1],
                pos[:, 1].max() + margin * pos_range[1],
            )

    # Plot the final coarsened graph
    Gc = Gall[-1]
    if pos_dim == 2:
        ax = fig.add_subplot(1, n_levels + 1, n_levels + 1)
    else:
        ax = fig.add_subplot(1, n_levels + 1, n_levels + 1, projection="3d")

    ax.axis("off")
    ax.set_title(f"{title} | level = {n_levels}, n = {Gc.num_nodes}")

    # Get final positions
    final_pos = Gc.pos.cpu().numpy()

    # Plot final edges
    if Gc.edge_index.shape[1] > 0:
        final_edge_coords = final_pos[Gc.edge_index.cpu().numpy()]

        if pos_dim == 2:
            from matplotlib.collections import LineCollection

            lines = LineCollection(
                final_edge_coords.transpose(1, 0, 2),
                colors="black",
                alpha=alpha,
                linewidths=edge_width,
            )
            ax.add_collection(lines)
        else:
            for edge_coord in final_edge_coords.transpose(1, 0, 2):
                ax.plot(*edge_coord.T, color="black", alpha=alpha, linewidth=edge_width)

    # Plot final nodes
    if pos_dim == 2:
        ax.scatter(
            final_pos[:, 0], final_pos[:, 1], c="black", s=node_size, alpha=alpha
        )
        ax.set_aspect("equal")
        # Set axis limits
        margin = 0.1
        pos_range = final_pos.max(axis=0) - final_pos.min(axis=0)
        ax.set_xlim(
            final_pos[:, 0].min() - margin * pos_range[0],
            final_pos[:, 0].max() + margin * pos_range[0],
        )
        ax.set_ylim(
            final_pos[:, 1].min() - margin * pos_range[1],
            final_pos[:, 1].max() + margin * pos_range[1],
        )
    else:
        ax.scatter(
            final_pos[:, 0],
            final_pos[:, 1],
            final_pos[:, 2],
            c="black",
            s=node_size,
            alpha=alpha,
        )

    fig.tight_layout()
    return fig


def plot_coarsening_evolution(Gall, Call, save_path=None, **kwargs):
    """
    Enhanced wrapper for plotting coarsening evolution with additional features

    Parameters
    ----------
    Gall : list of torch_geometric.data.Data
        All graphs involved in the multilevel coarsening
    Call : list of torch.sparse tensors
        Coarsening matrices for each level
    save_path : str, optional
        Path to save the figure
    **kwargs : dict
        Additional arguments passed to plot_coarsening

    Returns
    -------
    fig : matplotlib figure
    stats : dict
        Statistics about the coarsening process
    """
    import matplotlib.pyplot as plt

    # Calculate coarsening statistics
    stats = {
        "levels": len(Gall) - 1,
        "original_nodes": Gall[0].num_nodes,
        "final_nodes": Gall[-1].num_nodes,
        "reduction_ratio": Gall[0].num_nodes / Gall[-1].num_nodes,
        "nodes_per_level": [G.num_nodes for G in Gall],
        "edges_per_level": [G.num_edges for G in Gall],
    }

    # Create the plot
    fig = plot_coarsening(Gall, Call, **kwargs)

    if fig is not None:
        # Add overall statistics as figure suptitle
        fig.suptitle(
            f"Graph Coarsening Evolution: {stats['original_nodes']} → {stats['final_nodes']} nodes "
            f"(reduction: {stats['reduction_ratio']:.2f}×)",
            fontsize=14,
            y=0.95,
        )

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

    return fig, stats


################################################################################
# Variation-based contraction algorithms
################################################################################


def contract_variation_edges(
    G: Data, A=None, K=10, r=0.5, algorithm="greedy", similarity_threshold=0.5
):
    """
    Sequential contraction with local variation and edge-based families.
    This is a specialized implementation for the edge-based family, that works
    slightly faster than the contract_variation() function, which works for
    any family.

    See contract_variation() for documentation.
    """
    N, deg, M = (
        G.num_nodes,
        degree(G.edge_index, G.num_nodes, G.edge_weight),
        G.num_edges,
    )
    ones = torch.ones(2)
    Pibot = torch.eye(2) - torch.outer(ones, ones) / 2

    # cost function for the edge
    def subgraph_cost(G, A, edge):
        # edge, w = edge[:2].astype(torch.int64), edge[2]
        edge = G.edge_index
        w = G.edge_weight
        deg_new = 2 * deg[edge] - w
        L = torch.tensor([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return torch.linalg.norm(B.T @ L @ B)

    # edges = np.array(G.get_edge_list()[0:2])
    edges = G.edge_index
    weights = torch.tensor([subgraph_cost(G, A, edges[:, e]) for e in range(M)])
    # weights = torch.zeros(M)
    # for e in range(M):
    #    weights[e] = subgraph_cost_old(G, A, edges[:,e])

    if algorithm == "optimal":
        # identify the minimum weight matching
        coarsening_list = matching_optimal(G, weights=weights, r=r)

    elif algorithm == "greedy":
        # find a heavy weight matching
        coarsening_list = matching_greedy(
            G, weights=-weights, r=r, similarity_threshold=similarity_threshold
        )

    return coarsening_list


# TODO(WHT): include features for each node
def contract_variation_linear(
    G: Data, A, r=0.5, mode="neighborhood", similarity_threshold=0.5
):
    """
    Sequential contraction with local variation and general families.
    This is an implemmentation that improves running speed,
    at the expense of being more greedy (and thus having slightly larger error).

    See contract_variation() for documentation.
    """

    N, deg, W_lil = G.num_nodes, degree(G.edge_index, G.num_nodes, G.edge_weight), G.W

    # The following is correct only for a single level of coarsening.

    # cost function for the subgraph induced by nodes array
    def subgraph_cost(nodes):
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, dtype=torch.long)
        nc = len(nodes)
        if nc <= 1:
            return 0.0

        # Create a sparse permutation matrix P
        # P will be of size nc x N where nc = len(nodes)
        # P[i,j] = 1 if j is the i-th node in 'nodes', otherwise 0
        P_indices = torch.stack(
            [
                torch.arange(nc, device=W_lil.device),  # Row indices
                nodes,  # Column indices
            ]
        )
        P_values = torch.ones(nc, device=W_lil.device)
        P = torch.sparse_coo_tensor(P_indices, P_values, (nc, N), device=W_lil.device)

        # Extract the subgraph weight matrix using P @ W_lil @ P.T
        # First compute P @ W_lil
        PW = torch.sparse.mm(P, W_lil)
        # Then compute (P @ W_lil) @ P.T
        W_sub = torch.sparse.mm(PW, P.t())

        # Compute subgraph degree vector
        d_sub = torch.sparse.sum(W_sub, dim=1).to_dense()

        # Create Laplacian using original degrees and subgraph connections
        diag_values = 2 * deg[nodes] - d_sub
        L_diag = torch.sparse_coo_tensor(
            torch.stack([torch.arange(nc, device=W_lil.device)] * 2),
            diag_values,
            (nc, nc),
            device=W_lil.device,
        )
        L = L_diag - W_sub

        # Extract A submatrix using the same permutation approach
        A_sub = A[nodes, :]

        # Compute projection matrix
        ones = torch.ones(nc, device=W_lil.device) / nc
        P_A_sub = A_sub - torch.outer(ones, torch.sum(A_sub, dim=0))

        # Compute the final cost
        if L.is_sparse:
            L_P_A = torch.sparse.mm(L, P_A_sub)
        else:
            L_P_A = L @ P_A_sub

        return torch.norm(P_A_sub.t() @ L_P_A) / (nc - 1)

    class CandidateSet:
        def __init__(self, candidate_list):
            self.set = candidate_list
            self.cost = subgraph_cost(candidate_list)

        def __lt__(self, other):
            return self.cost < other.cost

    family = []
    # W_bool = G.A + sparse_eye(G.num_nodes)
    if "neighborhood" in mode:
        for i in range(N):
            # i_set = G.A[i,:].indices # graph_utils.get_neighbors(G, i)
            # i_set = np.append(i_set, i)
            # i_set = W_bool[i, :].indices
            i_set = k_hop_subgraph(i, 1, G.edge_index, num_nodes=G.num_nodes)[0]
            family.append(CandidateSet(i_set))

    if "cliques" in mode:
        import networkx.convert_matrix as nx

        Gnx = nx.from_scipy_sparse_matrix(G.W)
        for clique in nx.find_cliques(Gnx):
            family.append(CandidateSet(torch.tensor(clique)))

    else:
        if "edges" in mode:
            for e in range(0, G.edge_index.shape[1]):
                family.append(CandidateSet(G.edge_index[:, e]))
        if "triangles" in mode:
            triangles = set([])
            edges = G.edge_index
            for e in range(0, edges.shape[1]):
                [u, v] = edges[:, e]
                for w in range(G.num_nodes):
                    if G.W[u, w] > 0 and G.W[v, w] > 0:
                        triangles.add(frozenset([u, v, w]))
            triangles = list(map(lambda x: torch.tensor(list(x)), triangles))
            for triangle in triangles:
                family.append(CandidateSet(triangle))

    family = SortedList(family)
    marked = torch.zeros(G.num_nodes, dtype=torch.bool)

    # ----------------------------------------------------------------------------
    # Construct a (minimum weight) independent set.
    # ----------------------------------------------------------------------------
    coarsening_list = []
    # n, n_target = N, (1-r)*N
    n_reduce = np.floor(r * N)  # how many nodes do we need to reduce/eliminate?
    X = G.embeddings if hasattr(G, "embeddings") else G.x

    while len(family) > 0:

        i_cset = family.pop(index=0)
        i_set = i_cset.set

        if len(i_set) <= 1:
            continue

        # check if marked
        i_marked = marked[i_set]

        if not any(i_marked):

            n_gain = len(i_set) - 1
            if n_gain > n_reduce:
                continue  # this helps avoid over-reducing

            if X is not None and len(i_set) > 1:
                sim = similarity(X[i_set])
            else:
                sim = 1.0

            # print(f"{sim=}, {len(i_set)=}, {i_set=}, {X[i_set]=}")

            # probability based on similarity merging
            if sim > similarity_threshold:

                # all vertices are unmarked: add i_set to the coarsening list
                marked[i_set] = True
                coarsening_list.append(
                    {"list": i_set, "similarity": sim, "cost": i_cset.cost}
                )
                # n -= len(i_set) - 1
                n_reduce -= n_gain

            # if n <= n_target: break
            if n_reduce <= 0:
                break

        # may be worth to keep this set
        # else:
        #     i_set = i_set[~i_marked]
        #     if len(i_set) > 1:
        #         # todo1: check whether to add to coarsening_list before adding to family
        #         # todo2: currently this will also select contraction sets that are disconnected
        #         # should we eliminate those?
        #         i_cset.set = i_set
        #         i_cset.cost = subgraph_cost(i_set)
        #         family.add(i_cset)

    return coarsening_list


################################################################################
# Edge-based contraction algorithms
################################################################################


def get_proximity_measure(G: Data, name, K=10):

    N = G.num_nodes
    W = G.W  # np.array(G.W.toarray(), dtype=np.float32)
    deg = degree(G.edge_index, G.num_nodes, G.edge_weight)  # np.sum(W, axis=0)
    edges = G.edge_index
    weights = G.edge_weight
    M = edges.shape[1]

    num_vectors = K  # int(1*K*np.log(K))
    if "lanczos" in name:
        L = G.L  # Assuming this returns a sparse tensor
        X_init = torch.randn(N, K, device=L.device)
        l_lan, X_lan = torch.lobpcg(L, X_init, largest=False, niter=1000, tol=1e-2)
    elif "cheby" in name:
        e = "laplacian of G"
        X_cheby = generate_test_vectors(
            G, num_vectors=num_vectors, method="Chebychev", lambda_cut=G.e[K + 1]
        )
    elif "JC" in name:
        X_jc = generate_test_vectors(
            G, num_vectors=num_vectors, method="JC", iterations=20
        )
    elif "GS" in name:
        X_gs = generate_test_vectors(
            G, num_vectors=num_vectors, method="GS", iterations=1
        )
    if "expected" in name:
        X = X_lan
        assert not torch.isnan(X).any()
        assert X.shape[0] == N
        K = X.shape[1]

    proximity = torch.zeros(M, dtype=torch.float32)

    # heuristic for mutligrid
    if name == "heavy_edge":
        wmax = torch.tensor(torch.max(G.W, 0).todense())[0] + 1e-5
        for e in range(0, M):
            proximity[e] = weights[e] / max(
                wmax[edges[:, e]]
            )  # select edges with large proximity
        return proximity

    # heuristic for mutligrid
    elif name == "algebraic_JC":
        proximity += torch.inf
        for e in range(0, M):
            i, j = edges[:, e]
            for kIdx in range(num_vectors):
                xk = X_jc[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(torch.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

        return proximity

    # heuristic for mutligrid
    elif name == "affinity_GS":
        c = torch.zeros((N, N))
        for e in range(0, M):
            i, j = edges[:, e]
            c[i, j] = (X_gs[i, :] @ X_gs[j, :].T) ** 2 / (
                (X_gs[i, :] @ X_gs[i, :].T) ** 2 * (X_gs[j, :] @ X_gs[j, :].T) ** 2
            )  # select

        c += c.T
        c -= torch.diag(torch.diag(c))
        for e in range(0, M):
            i, j = edges[:, e]
            proximity[e] = c[i, j] / (max(c[i, :]) * max(c[j, :]))

        return proximity

    for e in range(0, M):
        i, j = edges[:, e]

        if name == "heavy_edge_degree":
            proximity[e] = (
                deg[i] + deg[j] + 2 * G.W[i, j]
            )  # select edges with large proximity

        # loose as little information as possible (custom)
        elif "min_expected_loss" in name:
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [proximity[e], (xk[i] - xk[j]) ** 2]
                )  # select edges with small proximity
        # loose as little gradient information as possible (custom)
        elif name == "min_expected_gradient_loss":
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2 * (deg[i] + deg[j] + 2 * G.W[i, j]),
                    ]
                )  # select edges with small proximity

        # relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss":
            for kIdx in range(1, K):
                xk = G.U[:, kIdx]
                lk = G.e[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4)
                        / lk,
                    ]
                )  # select edges with small proximity

        # fast relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_lanczos":
            for kIdx in range(1, K):
                xk = X_lan[:, kIdx]
                lk = l_lan[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0.5 * (lk + lk))
                        / lk,
                    ]
                )  # select edges with small proximity

        # approximate relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_cheby":
            for kIdx in range(num_vectors):
                xk = X_cheby[:, kIdx]
                lk = xk.T @ G.L @ xk
                proximity[e] = sum(
                    [
                        proximity[e],
                        (
                            (xk[i] - xk[j]) ** 2
                            * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0 * lk)
                            / lk
                        ),
                    ]
                )  # select edges with small proximity

        # heuristic for mutligrid (algebraic multigrid)
        elif name == "algebraic_GS":
            proximity[e] = torch.inf
            for kIdx in range(num_vectors):
                xk = X_gs[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(torch.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

    if ("rss" in name) or ("expected" in name):
        proximity = -proximity

    return proximity


def generate_test_vectors(
    G, num_vectors=10, method="Gauss-Seidel", iterations=5, lambda_cut=0.1
):

    L = G.L
    N = G.num_nodes
    X = torch.randn(N, num_vectors) / torch.sqrt(N)

    if method == "GS" or method == "Gauss-Seidel":
        # Extract upper triangular part (excluding diagonal)
        indices = L.indices()
        values = L.values()

        # Create mask for upper triangular elements (i < j)
        upper_mask = indices[0] < indices[1]
        # Create mask for lower triangular elements (including diagonal) (i >= j)
        lower_diag_mask = indices[0] >= indices[1]

        # Create L_upper (strictly upper triangular part)
        L_upper_indices = indices[:, upper_mask]
        L_upper_values = values[upper_mask]
        L_upper = torch.sparse_coo_tensor(L_upper_indices, L_upper_values, L.shape)

        # Create L_lower_diag (lower triangular part including diagonal)
        L_lower_diag_indices = indices[:, lower_diag_mask]
        L_lower_diag_values = values[lower_diag_mask]
        L_lower_diag = torch.sparse_coo_tensor(
            L_lower_diag_indices, L_lower_diag_values, L.shape
        )

        # Convert to CSR format for more efficient operations
        L_upper = L_upper.coalesce()
        L_lower_diag = L_lower_diag.coalesce()

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                # Compute right hand side: -L_upper @ x
                rhs = -torch.sparse.mm(L_upper, x.unsqueeze(1)).squeeze()

            # Solve L_lower_diag @ x_new = rhs using forward substitution
            x_new = torch.zeros_like(x)
            for i in range(N):
                # Get row i of L_lower_diag
                row_indices = (L_lower_diag.indices()[0] == i).nonzero().squeeze()
                if len(row_indices.shape) == 0:  # Handle scalar case
                    row_indices = row_indices.unsqueeze(0)

                col_indices = L_lower_diag.indices()[1][row_indices]
                row_values = L_lower_diag.values()[row_indices]

                # Compute x[i] = (rhs[i] - sum(L[i,j] * x[j] for j < i)) / L[i,i]
                # Note: We only need to consider j < i for forward substitution
                diag_idx = (col_indices == i).nonzero().squeeze()
                if (
                    len(diag_idx.shape) == 0 and diag_idx.nelement() > 0
                ):  # Handle scalar case
                    diag_idx = diag_idx.unsqueeze(0)

                if diag_idx.nelement() > 0:  # Check if diagonal element exists
                    diag_val = row_values[diag_idx]
                off_diag_mask = col_indices < i
                if off_diag_mask.any():
                    off_diag_sum = torch.sum(
                        row_values[off_diag_mask] * x_new[col_indices[off_diag_mask]]
                    )
                    x_new[i] = (rhs[i] - off_diag_sum) / diag_val
                else:
                    x_new[i] = rhs[i] / diag_val

            x = x_new

            X[:, j] = x
        return X

    if method == "JC" or method == "Jacobi":

        deg = degree(G.edge_index, G.num_nodes, G.edge_weight)
        indices = torch.arange(len(deg)).repeat(2, 1)
        D = torch.sparse_coo_tensor(indices, deg, (len(deg), len(deg)))
        # D = sp.sparse.diags(deg, 0)
        deginv = deg ** (-1)
        deginv[deginv == torch.inf] = 0
        Dinv = torch.sparse_coo_tensor(indices, deginv, (len(deginv), len(deginv)))
        M = Dinv.dot(D - L)

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                x = 0.5 * x + 0.5 * M.dot(x)
            X[:, j] = x
        return X

    elif method == "Chebychev":
        from pygsp import filters

        f = filters.Filter(G, lambda x: ((x <= lambda_cut) * 1).astype(torch.float32))
        return f.filter(X, method="chebyshev", order=50)


def matching_optimal(edge_index, num_nodes, weights, r=0.4):
    """
    Generates a matching optimally with the objective of minimizing the total
    weight of all edges in the matching.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    ratio : float
        The desired dimensionality reduction (ratio = 1 - n/N)

    Notes:
    * The complexity of this is O(N^3)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """
    N = num_nodes

    # the edge set
    edges = edge_index
    M = edges.shape[1]

    max_weight = 1 * torch.max(weights)

    # prepare the input for the minimum weight matching problem
    edge_list = []
    for edgeIdx in range(M):
        [i, j] = edges[:, edgeIdx]
        if i == j:
            continue
        edge_list.append((i, j, max_weight - weights[edgeIdx]))

    assert min(weights) >= 0

    # solve it
    tmp = torch.tensor(maxWeightMatching.maxWeightMatching(edge_list))

    # format output
    m = tmp.shape[0]
    matching = torch.zeros((m, 2), dtype=int)
    matching[:, 0] = range(m)
    matching[:, 1] = tmp

    # remove null edges and duplicates
    idx = torch.where(tmp != -1)[0]
    matching = matching[idx, :]
    idx = torch.where(matching[:, 0] > matching[:, 1])[0]
    matching = matching[idx, :]

    assert matching.shape[0] >= 1

    # if the returned matching is larger than what is requested, select the min weight subset of it
    matched_weights = torch.zeros(matching.shape[0])
    for mIdx in range(matching.shape[0]):
        i = matching[mIdx, 0]
        j = matching[mIdx, 1]
        eIdx = [
            e
            for e, t in enumerate(edges[:, :].T)
            if ((t == [i, j]).all() or (t == [j, i]).all())
        ]
        matched_weights[mIdx] = weights[eIdx]

    keep = min(int(torch.ceil(r * N)), matching.shape[0])
    if keep < matching.shape[0]:
        idx = torch.argsort(matched_weights)[:keep]
        matching = matching[idx, :]

    return matching


def matching_greedy(G: Data, weights, r=0.4, similarity_threshold=0.5):
    """
    Generates a matching greedily by selecting at each iteration the edge
    with the largest weight and then removing all adjacent edges from the
    candidate set.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    X : np.array(N, D), optional
        node features, used for computing similarity
    r : float
        The desired dimensionality reduction (r = 1 - n/N)
    similarity_threshold : float
        The threshold for considering two nodes as similar

    Notes:
    * The complexity of this is O(M)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """

    N = G.num_nodes
    X = G.embeddings if hasattr(G, "embeddings") else G.x

    # the edge set
    edges = G.edge_index
    M = edges.shape[1]

    idx = torch.argsort(-weights)
    # idx = np.argsort(weights)[::-1]
    edges = edges[:, idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []

    # which vertices have been selected
    marked = torch.zeros(N, dtype=torch.bool)

    n, n_target = N, (1 - r) * N
    while len(candidate_edges) > 0:

        # pop a candidate edge
        [i, j] = candidate_edges.pop(0)

        # check if marked
        if any(marked[[i, j]]):
            continue

        if X is not None:
            sim = similarity(X[[i, j]])
        else:
            sim = 1.0

        if sim > similarity_threshold:
            marked[[i, j]] = True
            n -= 1

            # add it to the matching
            matching.append(torch.tensor([i, j]))

        # termination condition
        if n <= n_target:
            break

    return torch.stack(matching)
