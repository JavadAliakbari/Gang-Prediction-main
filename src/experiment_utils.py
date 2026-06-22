"""Utility functions for pattern loading, evaluation, and experiment setup."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import torch
import numpy as np
import pandas as pd
import torch_geometric.data
import torch_geometric.transforms
from sklearn.preprocessing import MinMaxScaler

from src.pattern_models import Pattern, create_pattern
from src.preprocessor import DataPreprocessor
from src.utils.utils import *

from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    add_self_loops,
    scatter,
)


def _capture_pattern_lineage(
    patterns: List[Pattern],
    node_to_supernode: Optional[torch.Tensor],
    pseudo_labels: Optional[torch.Tensor],
) -> None:
    """Capture per-level super-node and pseudo-label lineage for pattern nodes."""

    for pattern in patterns:
        pattern.capture_level(
            node_to_supernode=node_to_supernode,
            pseudo_labels=pseudo_labels,
        )


def load_patterns_from_file(
    file_path: str, node_to_index: dict, label: str = "unknown"
) -> List[Pattern]:
    """
    Load patterns from CSV or parquet file and group accounts by pattern ID.

    Args:
        file_path: Path to CSV or parquet file with columns [modelID, accountID, type]
        node_to_index: Dict mapping account IDs to node indices
        min_nodes: Minimum number of mapped nodes required for a valid pattern

    Returns:
        patterns: dict mapping pattern_id -> list of node indices OR List[Pattern]
        pattern_types: dict mapping pattern_id -> pattern type
    """
    file_path = Path(file_path)
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    tmp = df.loc[
        df["accountID"].isin(node_to_index), ["modelID", "accountID", "type"]
    ].assign(account_idx=lambda d: d["accountID"].map(node_to_index))

    pattern_dicts = tmp.groupby("modelID")["account_idx"].agg(list).to_dict()
    pattern_types = tmp.groupby("modelID")["type"].first().to_dict()

    patterns = []
    for pattern_id, node_indices in pattern_dicts.items():
        pattern = create_pattern(
            pattern_id=pattern_id,
            nodes=np.unique(node_indices),
            pattern_type=pattern_types.get(pattern_id, "unknown"),
            label=label,
        )
        patterns.append(pattern)

    return patterns


def get_node_to_supernode_mapping(C):
    """
    Get mapping from original nodes to super nodes after coarsening.

    Args:
        Call: List containing cumulative coarsening matrix

    Returns:
        mapping: Tensor where mapping[i] = super node index for original node i
    """
    C_dense = C.to_dense()
    return torch.argmax(C_dense, dim=0)


def load_preprocessing_config(config_path: str) -> Dict:
    """
    Load preprocessing configuration with auto-discovery and path construction.

    Convention over configuration:
    - Experiment root: Auto-detected from config path (../../ from config file)
    - All paths: Auto-constructed based on experiment root

    Args:
        config_path: Path to preprocessing config YAML file (typically experiments/{name}/config/preprocessing.yaml)

    Returns:
        Complete configuration dictionary with auto-constructed paths
    """
    config_path = Path(config_path)

    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Auto-detect experiment root from config path
    # Config at: experiments/{name}/config/preprocessing.yaml
    # Root is: experiments/{name}/
    experiment_root = config_path.parent.parent

    # Auto-construct paths based on experiment root
    config["raw_data_file"] = str(experiment_root / "temporal" / "tx_log.parquet")
    config["preprocessed_data_dir"] = str(experiment_root / "preprocessed")

    return config


def load_and_preprocess_data(
    data_dir: Path,
    patterns_dir: Path,
    train_ratio: float = 0.5,
    to_undirected: bool = True,
    remove_overlaps: bool = True,
    device: Optional[torch.device] = "cpu",
    seed: Optional[int] = None,
) -> Tuple[torch_geometric.data.Data, List[Pattern], List[Pattern]]:
    """
    Load and preprocess AMLGentex dataset.

    Args:
        data_dir: Path to data directory
        patterns_dir: Path to patterns directory
        device: Device to use for tensor operations
        train_ratio: Ratio of patterns to use for training
        to_undirected: Whether to convert the graph to undirected

    Returns:
        G: PyTorch Geometric Data object
        node_to_index: Dict mapping account IDs to node indices
    """

    G, node_to_index = load_amlgentex_data(data_dir, to_undirected, device=device)

    # Load patterns
    alert_patterns, normal_patterns = load_all_patterns(patterns_dir, node_to_index)

    # Preprocess overlaps so each node belongs to at most one pattern.
    if remove_overlaps:
        # Alert patterns take priority over normal patterns for cross-class conflicts.
        alert_patterns, normal_patterns, overlap_stats = (
            preprocess_patterns_with_alert_priority(
                alert_patterns,
                normal_patterns,
                labels=G.y,
            )
        )
        print("\nPattern overlap preprocessing summary:")
        print(
            f"  Alert patterns: {overlap_stats['alert_patterns_before']} -> {overlap_stats['alert_patterns_after']} "
            f"(removed empty: {overlap_stats['removed_alert_patterns_empty']})"
        )
        print(
            f"  Normal patterns: {overlap_stats['normal_patterns_before']} -> {overlap_stats['normal_patterns_after']} "
            f"(removed empty: {overlap_stats['removed_normal_patterns_empty']})"
        )
        print(
            f"  Assigned nodes: alert={overlap_stats['assigned_alert_nodes']}, "
            f"normal={overlap_stats['assigned_normal_nodes']}, total={overlap_stats['assigned_total_nodes']}"
        )

    # Split patterns into train and test sets
    print("\n" + "=" * 60)
    print("Splitting Patterns into Train/Test Sets")
    print(f"  Train ratio: {train_ratio}")
    print("=" * 60)

    alert_train, alert_test = split_patterns(
        alert_patterns,
        train_ratio=train_ratio,
        seed=seed,
    )
    normal_train, normal_test = split_patterns(
        normal_patterns,
        train_ratio=train_ratio,
        seed=seed,
    )

    print(f"  Alert patterns: {len(alert_train)} train, {len(alert_test)} test")
    print(f"  Normal patterns: {len(normal_train)} train, {len(normal_test)} test")

    # Get node indices from train patterns to use as training nodes
    alert_train_nodes = get_pattern_node_indices(alert_train)
    normal_train_nodes = get_pattern_node_indices(normal_train)

    # Combine train nodes from both alert and normal patterns
    train_nodes = torch.unique(torch.cat([alert_train_nodes, normal_train_nodes]))
    # test_nodes = torch.unique(torch.cat([alert_test_nodes, normal_test_nodes]))
    all_nodes = torch.arange(G.num_nodes)
    test_nodes = all_nodes[
        ~torch.isin(all_nodes, train_nodes)
    ]  # Use all nodes for testing to evaluate generalization

    # Update graph training masks to use pattern-based nodes
    print(f"\n  Training nodes from patterns: {len(train_nodes)}")
    print(f"  Test nodes from patterns: {len(test_nodes)}")

    # Override train/val/test indices with pattern-based split
    G.train_idx = train_nodes
    G.val_idx = test_nodes  # Use test patterns for validation
    G.test_idx = test_nodes  # Use test patterns for testing

    return G, alert_train, normal_train, alert_test, normal_test


def load_amlgentex_data(
    config_dir: Path, to_undirected: bool = True, device: Optional[torch.device] = "cpu"
) -> Tuple[torch_geometric.data.Data, dict]:
    """
    Load and preprocess AMLGentex dataset.

    Args:
        config_dir: Path to config directory
        to_undirected: Whether to convert the graph to undirected

    Returns:
        G: PyTorch Geometric Data object
        node_to_index: Dict mapping account IDs to node indices
    """

    preproc_config = load_preprocessing_config(str(config_dir / "preprocessing.yaml"))

    print("Preprocessing configuration:")
    print(f"  Raw data: {preproc_config['raw_data_file']}")
    print(f"  Output dir: {preproc_config['preprocessed_data_dir']}")
    print("\nPreprocessing transactions...")
    print("Generating temporal features with rolling windows...\n")

    preprocessor = DataPreprocessor(preproc_config)
    datasets = preprocessor(preproc_config["raw_data_file"])

    nodes_df = datasets["trainset_nodes"]
    edges_df = datasets["trainset_edges"]

    # Create node ID to index mapping
    node_to_index = {
        account_id: idx for idx, account_id in enumerate(nodes_df["account"])
    }

    # Get train/val/test indices
    # train_idx = torch.tensor(
    #     [node_to_index[acc] for acc in nodes_df[nodes_df["train_mask"]]["account"]],
    #     dtype=torch.long,
    # )
    # val_idx = torch.tensor(
    #     [node_to_index[acc] for acc in nodes_df[nodes_df["val_mask"]]["account"]],
    #     dtype=torch.long,
    # )
    # test_idx = torch.tensor(
    #     [node_to_index[acc] for acc in nodes_df[nodes_df["test_mask"]]["account"]],
    #     dtype=torch.long,
    # )

    # Prepare features and labels
    nodes_df = nodes_df.drop(columns=["bank"])
    X = (
        nodes_df.drop(
            columns=["account", "train_mask", "val_mask", "test_mask", "is_sar"]
        )
        .to_numpy()
        .astype(np.float32)
    )
    y = nodes_df["is_sar"].to_numpy().astype(np.int64)

    # Map edges
    edges_df["src_idx"] = edges_df["src"].map(node_to_index)
    edges_df["dst_idx"] = edges_df["dst"].map(node_to_index)
    edges_df = edges_df.dropna(subset=["src_idx", "dst_idx"])
    edges = edges_df[["src_idx", "dst_idx"]].to_numpy().astype(np.int64)
    edges_index = torch.tensor(edges.T, dtype=torch.long).to(device)

    # Normalize features
    scaler = MinMaxScaler().fit(X)
    X_normalized = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.int64, device=device)

    # Create PyG Data object
    G = torch_geometric.data.Data(x=X_normalized, edge_index=edges_index, y=y)
    if to_undirected:
        G = torch_geometric.transforms.ToUndirected()(G)
    G.edge_weight = torch.ones(G.edge_index.size(1), dtype=torch.float32, device=device)
    # G.train_idx = train_idx
    # G.val_idx = val_idx
    # G.test_idx = test_idx
    G.W, G.L, G.dw = graph_params(G)

    return G, node_to_index


def split_patterns(
    patterns: List[Pattern],
    train_ratio: float = 0.7,
    seed: Optional[int] = None,
):
    """
    Split patterns into train and test sets.

    Args:
        patterns: List of Pattern objects.
        train_ratio: Fraction of patterns to use for training.
        seed: Optional random seed for reproducibility.

    Returns:
        train_patterns: Training patterns sorted by pattern type.
        test_patterns: Test patterns sorted by pattern type.
    """

    shuffled_patterns = list(patterns)
    rng = random.Random(seed)
    rng.shuffle(shuffled_patterns)

    n_train = int(len(shuffled_patterns) * train_ratio)
    train_patterns = shuffled_patterns[:n_train]
    test_patterns = shuffled_patterns[n_train:]

    # Keep split randomization while returning deterministic, type-grouped lists.
    sort_key = lambda p: (p.pattern_id)
    train_patterns = sorted(train_patterns, key=sort_key)
    test_patterns = sorted(test_patterns, key=sort_key)

    return train_patterns, test_patterns


def get_pattern_node_indices(patterns: dict) -> torch.Tensor:
    """
    Get all unique node indices from patterns.

    Args:
        patterns: Dict mapping pattern_id -> list of node indices

    Returns:
        Tensor of unique node indices from all patterns
    """
    all_nodes = set()

    for pattern in patterns:
        all_nodes.update(pattern.node_indices)

    return torch.tensor(sorted(all_nodes), dtype=torch.long)


def load_all_patterns(experiment_root: Path, node_to_index: dict):
    """
    Load alert and normal patterns from experiment directory.

    Args:
        experiment_root: Path to experiment directory
        node_to_index: Dict mapping account IDs to node indices

    Returns:
        alert_patterns: Dict of alert patterns
        alert_types: Dict of alert pattern types
        normal_patterns: Dict of normal patterns
        normal_types: Dict of normal pattern types
    """
    alert_patterns = {}
    normal_patterns = {}

    # Try parquet first, then CSV for alerts
    alert_csv = experiment_root / "spatial" / "alert_models.csv"

    if alert_csv.exists():
        alert_patterns = load_patterns_from_file(
            str(alert_csv), node_to_index, label="alert"
        )
        print(f"\nLoaded {len(alert_patterns)} alert patterns from CSV")
    else:
        print("\nNo alert patterns file found")

    # Load normal patterns
    normal_csv = experiment_root / "spatial" / "normal_models.csv"
    if normal_csv.exists():
        normal_patterns = load_patterns_from_file(
            str(normal_csv), node_to_index, label="normal"
        )
        print(f"\nLoaded {len(normal_patterns)} normal patterns")
    else:
        print("\nNo normal patterns file found")

    return alert_patterns, normal_patterns


def preprocess_patterns_with_alert_priority(
    alert_patterns: List[Pattern],
    normal_patterns: List[Pattern],
    labels: Optional[torch.Tensor] = None,
) -> Tuple[List[Pattern], List[Pattern], Dict[str, int]]:
    """Resolve node overlap so each node belongs to exactly one pattern.

    Rules:
    - Alert patterns have priority over normal patterns.
    - Within the same class, first pattern in list order keeps the node.
    - Patterns left with zero nodes are removed.
    - If labels tensor is provided, assigned alert/normal nodes are forced to 1/0.
    """

    def _unique_ordered(nodes) -> List[int]:
        seen = set()
        uniq = []
        for n in nodes:
            idx = int(n)
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)
        return uniq

    assigned_nodes = set()
    assigned_to_alert = set()
    assigned_to_normal = set()

    def _assign_patterns(patterns: List[Pattern], target: str) -> List[Pattern]:
        cleaned = []
        for pattern in patterns:
            nodes = _unique_ordered(pattern.node_indices)
            kept_nodes = []
            for node in nodes:
                if node in assigned_nodes:
                    continue
                assigned_nodes.add(node)
                kept_nodes.append(node)
                if target == "alert":
                    assigned_to_alert.add(node)
                else:
                    assigned_to_normal.add(node)

            pattern.nodes = kept_nodes
            pattern.label = target
            if kept_nodes:
                cleaned.append(pattern)
        return cleaned

    # Alert first => cross-class conflicts resolve to alert.
    alert_clean = _assign_patterns(alert_patterns, "alert")
    normal_clean = _assign_patterns(normal_patterns, "normal")

    if labels is not None:
        if assigned_to_alert:
            alert_idx = torch.tensor(
                sorted(assigned_to_alert), dtype=torch.long, device=labels.device
            )
            labels[alert_idx] = 1
        if assigned_to_normal:
            normal_idx = torch.tensor(
                sorted(assigned_to_normal), dtype=torch.long, device=labels.device
            )
            labels[normal_idx] = 0

    stats = {
        "alert_patterns_before": len(alert_patterns),
        "normal_patterns_before": len(normal_patterns),
        "alert_patterns_after": len(alert_clean),
        "normal_patterns_after": len(normal_clean),
        "removed_alert_patterns_empty": len(alert_patterns) - len(alert_clean),
        "removed_normal_patterns_empty": len(normal_patterns) - len(normal_clean),
        "assigned_alert_nodes": len(assigned_to_alert),
        "assigned_normal_nodes": len(assigned_to_normal),
        "assigned_total_nodes": len(assigned_nodes),
    }

    return alert_clean, normal_clean, stats


def graph_params(G: torch_geometric.data.Data):
    """Compute adjacency, Laplacian, and degree for a PyG graph."""
    num_nodes = G.num_nodes
    # edge_index, edge_weight = remove_self_loops(G.edge_index, G.edge_weight)
    edge_index, edge_weight = G.edge_index, G.edge_weight

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), dtype=torch.float32, device=edge_index.device
        )
    # Use the de-looped edge_index (not G.edge_index) for the adjacency matrix
    W = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
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


def create_subspace(alert_patterns, normal_patterns, num_nodes, device):
    # Create binary indicator matrices for alert and normal patterns
    alert_pattern_types = sorted(set(p.pattern_type for p in alert_patterns))
    normal_pattern_types = sorted(set(p.pattern_type for p in normal_patterns))
    V_alert = torch.zeros((num_nodes, len(alert_pattern_types)), device=device)
    V_normal = torch.zeros((num_nodes, len(normal_pattern_types)), device=device)
    pattern_type_to_idx_alert = {t: i for i, t in enumerate(alert_pattern_types)}
    pattern_type_to_idx_normal = {t: i for i, t in enumerate(normal_pattern_types)}
    for p in alert_patterns:
        idx = pattern_type_to_idx_alert[p.pattern_type]
        V_alert[p.nodes, :] = 0.0
        V_normal[p.nodes, :] = 0.0
        V_alert[p.nodes, idx] = 1.0
    for p in normal_patterns:
        idx = pattern_type_to_idx_normal[p.pattern_type]
        V_normal[p.nodes, :] = 0.0
        V_alert[p.nodes, :] = 0.0
        V_normal[p.nodes, idx] = 1.0
    U = torch.cat([V_alert, V_normal], dim=1)
    return U
    # return V_alert


def l_orthonormalize(Z, L, eps=1e-6, remove_constant=True):
    """
    Z: [N, k]
    L: [N, N] symmetric PSD
    returns U with approximately U.T @ L @ U = I
    """
    N, k = Z.shape

    if remove_constant:
        Z = Z - Z.mean(dim=0, keepdim=True)

    G = Z.T @ torch.sparse.mm(L.T, L) @ Z
    G = G + eps * torch.eye(k, device=Z.device, dtype=Z.dtype)

    # G = R^T R
    R = torch.linalg.cholesky(G, upper=True)

    # U = Z @ R^{-1}
    # better than explicit inverse:
    U = torch.linalg.solve_triangular(R, Z.T, upper=True, left=False).T

    return U


def calc_B_from_embeddings(embeddings: torch.Tensor, K: int = None) -> torch.Tensor:
    """
    Compute an orthonormal basis B from learned node embeddings.

    The embeddings learned by the GNN (with SupernodeEmbeddingLoss) naturally
    encode the coarsening structure: nodes in the same supernode have similar
    embeddings. We extract an orthonormal basis from these embeddings.

    Args:
        embeddings: [N, D] node embeddings (should be normalized)
        K: Number of basis vectors to return. If None, use min(N, D)

    Returns:
        B: [N, K] orthonormal basis matrix
    """
    N, D = embeddings.shape

    if K is None:
        K = min(N, D)
    K = min(K, N, D)

    # Normalize embeddings
    # embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # SVD to get orthonormal basis
    # U @ S @ V^T = embeddings
    # U: [N, K] orthonormal columns (our B)
    # S: [K] singular values
    # V: [D, K] orthonormal columns
    # try:
    #     U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)

    #     # Take the top K left singular vectors as basis
    #     B = U[:, :K]

    #     # Weight by singular values (optional, for smoothness)
    #     # S_weights = S[:K] / (S[:K].max() + 1e-6)
    #     # B = B * S_weights.unsqueeze(0)

    # except Exception as e:
    #     # Fallback to QR decomposition if SVD fails
    # print(f"SVD failed ({e}), falling back to QR decomposition")
    Q, R = torch.linalg.qr(embeddings)
    B = Q[:, :K]

    return B
