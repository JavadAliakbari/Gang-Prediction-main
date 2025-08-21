import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from graph_utils import *
from coarsening_utils import *
from utils.utils import *
from utils.visualization import *
from train_GNN_coarsening import train_GNN_coarsening_aware_loss2

import warnings

warnings.filterwarnings("ignore")

# Create a Planetoid dataset
dataset = Planetoid(root="data/Planetoid", name="Cora")
G = dataset[0]  # Get the first graph object
G.edge_index = to_undirected(G.edge_index)
G.edge_weight = torch.ones(G.edge_index.size(1), device=G.edge_index.device)

method = "variation_neighborhoods"
# epochs_per_lev = [1]
# thresholds = [0.50]
epochs_per_lev = [1, 2, 5, 10]
thresholds = [0.50, 0.75, 0.85, 0.95]

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


for ep_per_lev in epochs_per_lev:
    for threshold in thresholds:
        train_GNN_coarsening_aware_loss2(
            G,
            levels=25,
            K=20,
            lr=0.01,
            epoch_per_level=ep_per_lev,
            method=method,
            similarity_threshold=threshold,
        )
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}{name}", allow_pickle=True).item()
        LOGGER.info(f"epochs: {ep_per_lev}, threshold: {threshold}")
        LOGGER.info(
            f"nodes: {data['Gall'][-1].num_nodes}, edges: {data['Gall'][-1].num_edges}, coarse accuracy: {data['ycrs'][-1]:.4f}, fine accuracy: {data['yfine'][-1]:.4f}"
        )

    # same ratio and method in the same graph
    LOGGER.info(f"Epochs per Level: {ep_per_lev}")
    plt.figure(figsize=(16, 9))
    for idx, threshold in enumerate(thresholds):
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
        width = 2 if idx == len(thresholds) - 1 else 1
        plt.plot(
            data["ycrs"],
            color=colors[idx],
            linewidth=width,
            marker="*",
            label=f"Coarse {threshold*100:.0f}%",
        )
        plt.plot(
            data["yfine"],
            linestyle=":",
            marker="o",
            color=colors[idx],
            linewidth=width,
            label=f"Fine {threshold*100:.0f}%",
        )
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Coarse vs Fine Accuracy over Iterations")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{save_path}/iterative_custom_loss_accuracy_{ep_per_lev}.png")

    LOGGER.info(f"Epochs per Level: {ep_per_lev}")
    plt.figure(figsize=(16, 9))
    for idx, threshold in enumerate(thresholds):
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
        width = 2 if idx == len(thresholds) - 1 else 1
        plt.plot(
            np.array(data["num_nodes_coarse"]),
            data["ycrs"],
            color=colors[idx],
            linewidth=width,
            marker="*",
            label=f"Coarse {threshold*100:.0f}%",
        )
        plt.plot(
            np.array(data["num_nodes_coarse"]),
            data["yfine"],
            linestyle=":",
            marker="o",
            color=colors[idx],
            linewidth=width,
            label=f"Fine {threshold*100:.0f}%",
        )
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Coarse vs Fine Accuracy over Iterations")
    plt.tight_layout()
    plt.legend()
    plt.savefig(
        f"{save_path}/iterative_custom_loss_accuracy_{ep_per_lev}_with_nodes.png"
    )


plt.show()
