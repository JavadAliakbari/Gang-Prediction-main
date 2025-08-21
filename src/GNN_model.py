import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import torch.optim as optim

# torch geometric
from torch_geometric.data import Data

from sklearn.metrics import accuracy_score, classification_report


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index):
        # x = x.to(next(self.parameters()).device)  # send x to the model's device
        # edge_index = edge_index.to(x.device)
        x = F.relu(self.conv1(x, edge_index))
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


def train_gnn_1_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    data: Data,
    # coarsening_matrix,
    train_idx: list,
    # val_idx: list,
    coarse_loss: bool,
):
    """
    Output:
        - train_loss
        - val_loss
        - val_accuracy
    """

    # data = data.to(next(model.parameters()).device)

    model.train()
    optimizer.zero_grad()

    output = model(data.x, data.W)
    # embeddings = model.get_embeddings(data.x, data.edge_index)

    # train
    loss = criterion(
        output, data.embeddings, data.y_train, train_idx, coarse_loss=coarse_loss
    )
    loss.backward()
    optimizer.step()
    train_acc = accuracy_score(
        data.y_train[train_idx].detach().cpu().numpy(),
        output[train_idx].max(1)[1].detach().cpu().numpy(),
    )

    # validate
    # model.eval()
    # with torch.no_grad():
    #     output = model(data.x, data.edge_index)
    #     # embeddings = model.get_embeddings(data.x, data.edge_index)

    #     # loss_val = criterion(output, data.embeddings, data.y, val_idx)
    #     # pred_val = output[val_idx].max(1)[1]
    #     # acc_val = accuracy_score(data.y[val_idx].cpu().numpy(), pred_val.cpu().numpy())

    return loss.item(), train_acc


def evaluate_model(model: nn.Module, data: Data, test_idx, log_info=True):
    """Evaluate the model on test set."""
    model.eval()
    data = data.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(data.x, data.W)
        pred_test = output[test_idx].max(1)[1]
        acc_test = accuracy_score(
            data.y[test_idx].cpu().numpy(), pred_test.cpu().numpy()
        )

        if log_info:
            print(f"\nTest Accuracy: {acc_test:.4f}")
            print("\nClassification Report:")
            print(
                classification_report(
                    data.y[test_idx].cpu().numpy(), pred_test.cpu().numpy()
                )
            )

    return acc_test, pred_test, output
