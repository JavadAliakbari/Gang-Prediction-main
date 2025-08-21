import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

# torch
import torch
import torch.optim as optim
from torch_geometric.utils import to_networkx
import networkx as nx

# graph coarsening - Loukas 2020
from coarsening_utils import *
from graph_utils import *

# utils
from utils.utils import *
from utils.visualization import *

from GNN_model import GCN, train_gnn_1_epoch, evaluate_model
from coarsening_aware_loss import *
from create_coarsening_gif import create_coarsening_gif


def train_GNN_coarsening_aware_loss2(
    data: Data,
    levels: int,
    epoch_per_level: int,
    lr=0.01,
    wd=5e-4,
    method="variation_neighborhoods",
    algorithm: str = "greedy",
    similarity_threshold=0.65,
    K=50,
    nhid=128,
    dropout=0.1,
    device="cpu",
    # save_path="results",
    create_gif=True,
):
    # model
    N = data.num_nodes  # 2708
    nclass = len(np.unique(data.y.numpy()))  # 7
    model = GCN(nfeat=data.num_features, nhid=nhid, nclass=nclass, dropout=dropout).to(
        device
    )

    original_data = data
    Gc = data

    train_idx, val_idx, test_idx = create_train_val_test_split(original_data.num_nodes)
    P_train = create_P(train_idx, N, device=device)
    P_val = create_P(val_idx, N, device=device)
    P_test = create_P(test_idx, N, device=device)

    # criterion
    criterion = CoarseningAwareLoss()

    # train data
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    x, ycrs, yfine, ylosst, ylossv, valacc, num_nodes_coarse = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Create one-hot encoding of labels
    device = Gc.y.device
    num_classes = len(torch.unique(Gc.y))

    labels_onehot = torch.zeros(len(Gc.y), num_classes, device=device)
    labels_onehot.scatter_(1, Gc.y.view(-1, 1), 1)
    Gc.soft_y = labels_onehot

    ################################
    C = sparse_eye(N)
    Gc.W, Gc.L, Gc.dw = graph_params(Gc)
    GG = to_networkx(Gc, to_undirected=True)
    pos = nx.spring_layout(GG)
    pos = [vals for vals in pos.values()]
    Gc.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    B = calc_B(Gc, K)
    iC = None

    Call, Gall, iCs = [], [], []
    Gall.append(Gc)
    # Call.append(C)

    bar = tqdm(total=levels)
    for level in range(1, levels + 1):
        ratio = np.log(level ** (4 / 3)) / 100 + 0.01
        # get embeddings from the GNN
        embeddings = model.get_embeddings(Gc.x, Gc.W)
        Gc.embeddings = F.normalize(embeddings, p=2, dim=1)

        iC, Gc, B = coarse_one_level(
            Gc,
            iC,
            B,
            K=K,
            method=method,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            level=level,
            r_cur=ratio,
        )
        C = torch.sparse.mm(iC, C)
        Gall.append(Gc)
        Call.append(C)
        iCs.append(iC)

        for epoch in range(epoch_per_level):
            # train the GNN on the coarsened graph
            C_train = C @ P_train
            mask = torch.sum(C_train, dim=1).to_dense() > 1e-5
            train_idx_c = torch.nonzero(mask).view(-1)
            test_idx_c = torch.nonzero(~mask).view(-1)
            y_soft_train = C_train @ original_data.soft_y[train_idx]
            Gc.y_train = torch.argmax(y_soft_train, dim=1)
            Gc.y = torch.argmax(Gc.soft_y, dim=1) if Gc.soft_y is not None else Gc.y

            train_loss, train_acc = train_gnn_1_epoch(
                model,
                optimizer,
                criterion,
                Gc,
                train_idx_c,
                coarse_loss=epoch == 0,
            )

        # evaluate the model on the coarsened graph
        acc_test_c, pred_test_c, pred = evaluate_model(
            model, Gc, test_idx_c, log_info=False
        )
        acc_test, pred_test, _ = evaluate_model(
            model, original_data, test_idx, log_info=False
        )

        pred_fine = torch.argmax(C.t() @ pred, dim=1)

        accuracy_fine = torch.sum(
            pred_fine[test_idx] == original_data.y[test_idx]
        ).item() / len(test_idx)

        bar.set_postfix_str(
            f"{epoch_per_level}/{similarity_threshold:0.2f}, r: {ratio:.2f}, nodes: {Gc.num_nodes} "
            # + f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
            + f"coarse: {acc_test_c:.4f}, orig: {acc_test:.4f} fine: {accuracy_fine:.4f}"
        )
        bar.update(1)

        # ploting data
        x.append(epoch + 1)
        ycrs.append(acc_test_c)
        yfine.append(accuracy_fine)
        ylosst.append(train_loss)
        # ylossv.append(validation_loss)
        # valacc.append(validation_accuracy)
        num_nodes_coarse.append(Gc.num_nodes)
    # fig = plot_coarsening(Gall, iCs)
    # fig.suptitle(
    #     f"Coarsening: {method}, ratio: {similarity_threshold}, epochs per level: {epoch_per_level}"
    # )
    # fig.savefig(f"{save_path}/coarsening_plot.png")

    # Create GIF showing coarsening evolution
    # if create_gif and len(Gall) > 1:
    #     print("Creating coarsening evolution GIF...")
    #     try:
    #         # Use efficient version for large graphs (>1000 nodes)
    #         # if Gall[0].num_nodes > 1000:
    #         #     gif_path = create_coarsening_gif_efficient(
    #         #         Gall,
    #         #         iCs,
    #         #         save_path=f"{save_path}/coarsening_evolution_efficient.gif",
    #         #         duration=1500,  # 1.5 seconds per frame
    #         #         max_nodes_display=500,
    #         #     )
    #         # else:
    #         gif_path = create_coarsening_gif(
    #             Gall,
    #             iCs,
    #             save_path=f"{save_path}/coarsening_evolution.gif",
    #             duration=1500,  # 1.5 seconds per frame
    #             figsize=(16, 9),
    #             node_size_base=10,
    #             # highlight_combined=True,
    #         )
    #         print(f"GIF saved to: {gif_path}")
    #     except Exception as e:
    #         print(f"Error creating GIF: {e}")
    #         print("Continuing without GIF...")

    name = f"data_gnn_CoarseningAwareLoss_V2_th_{similarity_threshold*100:.0f}_epochs_{epoch_per_level}.npy"
    np.save(
        f"{save_path}{name}",
        {
            "x": x,
            "ycrs": ycrs,
            "yfine": yfine,
            "ylosst": ylosst,
            "Gall": Gall,
            "Call": Call,
            # "ylossv": ylossv,
            # "valacc": valacc,
            "num_nodes_coarse": num_nodes_coarse,
            "description": f"Data obtained using: {method=}, {similarity_threshold=}, CoarseningAwareLoss() levels approach",
        },
    )

    # plt.show()
