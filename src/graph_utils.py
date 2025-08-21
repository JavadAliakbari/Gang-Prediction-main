import numpy as np
import torch


def get_neighbors(G, i):
    return G.A[i, :].indices
    # return np.arange(G.N)[np.array((G.W[i,:] > 0).todense())[0]]


def get_S(G):
    """
    Construct the N x |E| gradient matrix S
    """
    # the edge set
    edges = G.get_edge_list()
    weights = np.array(edges[2])
    edges = np.array(edges[0:2])
    M = edges.shape[1]

    # Construct the N x |E| gradient matrix S
    S = np.zeros((G.N, M))
    for e in np.arange(M):
        S[edges[0, e], e] = np.sqrt(weights[e])
        S[edges[1, e], e] = -np.sqrt(weights[e])

    return S


# Compare the spectum of L and Lc
def eig(A, order="ascend"):

    # eigenvalue decomposition
    [l, X] = np.linalg.eigh(A)

    # reordering indices
    idx = l.argsort()
    if order == "descend":
        idx = idx[::-1]

    # reordering
    l = np.real(l[idx])
    X = X[:, idx]
    return (X, np.real(l))


def zero_diag(A):
    if hasattr(A, "is_sparse") and A.is_sparse:
        indices = A.indices()
        values = A.values()
                
        # Find diagonal entries (where row == col)
        diag_mask = indices[0, :] == indices[1, :]
                
        # Create new indices and values excluding diagonal elements
        new_indices = indices[:, ~diag_mask]
        new_values = values[~diag_mask]
                
        # Create a new sparse tensor without diagonal elements
        return torch.sparse.FloatTensor(new_indices, new_values, A.shape)
    else:
        D = A.diagonal()
        return A - np.diag(D)


def is_symmetric(As):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    As : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    from scipy import sparse

    if As.shape[0] != As.shape[1]:
        return False

    if not isinstance(As, sparse.coo_matrix):
        As = sparse.coo_matrix(As)

    r, c, v = As.row, As.col, As.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check
