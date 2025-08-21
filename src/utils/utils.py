from datetime import datetime
import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    add_self_loops,
    scatter,
)
from pygsp import graphs
import scipy as sp
from logger import getLOGGER

now = datetime.now().strftime("%Y%m%d_%H%M%S")

result_path = "results/"
save_path = f"{result_path}{now}/"

os.makedirs(save_path, exist_ok=True)
LOGGER = getLOGGER(
    name=f"{now}_Cora",
    log_on_file=True,
    save_path=save_path,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1)
np.random.seed(1)


def create_pyg_data(features, edges_idx, labels) -> Data:
    x = torch.FloatTensor(features.astype(np.float32))
    y = torch.LongTensor(labels)
    edge_index = torch.LongTensor(edges_idx.T)  # expects (2, num_edges)
    edge_index = to_undirected(edge_index)

    return Data(x=x, edge_index=edge_index, y=y)


def create_pygsp_graph(data: Data) -> graphs.Graph:
    # adjacency matrix
    adj_matrix = sp.sparse.coo_matrix(
        (np.ones(len(data.edge_index)), (data.edge_index[0], data.edge_index[1])),
        shape=(data.num_nodes, data.num_nodes),
    )
    # adj_matrix = adj_matrix + adj_matrix.T
    # adj_matrix.data = np.ones(len(adj_matrix.data))

    # pygsp graph
    G = graphs.Graph(adj_matrix)
    G.compute_laplacian()
    G.set_coordinates()

    return G


def create_train_val_test_split(num_nodes, train_ratio=0.6, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return train_idx, val_idx, test_idx


def degree(edge_index, num_nodes, edge_weights=None):
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    if edge_weights is not None:
        for i, j, w in zip(edge_index[0], edge_index[1], edge_weights):
            deg[i] += w
            deg[j] += w
    else:
        for i, j in zip(edge_index[0], edge_index[1]):
            deg[i] += 1
            deg[j] += 1
    return deg


def sparse_eye(size):
    indices = torch.arange(size).repeat(2, 1)
    values = torch.ones(size)
    C = torch.sparse_coo_tensor(indices, values, (size, size))
    return C


def graph_params(G: Data):
    num_nodes = G.num_nodes
    edge_index, edge_weight = remove_self_loops(G.edge_index, G.edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), dtype=torch.float32, device=edge_index.device
        )
    W = torch.sparse_coo_tensor(
        G.edge_index,
        G.edge_weight,
        size=(num_nodes, num_nodes),
        device=edge_index.device,
    )

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([-edge_weight, deg], dim=0)

    L = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes), device=edge_index.device
    )

    return W, L, deg


def create_P(indices, n, device="cpu"):
    n_t = len(indices)
    P_indices = torch.stack(
        [
            indices,  # Column indices
            torch.arange(n_t, device=device),  # Row indices
        ]
    )
    P_values = torch.ones(n_t, device=device)
    P = torch.sparse_coo_tensor(P_indices, P_values, (n, n_t), device=device)
    return P
