#!POPCORN leaderboard eigh
#!POPCORN gpu B200

import torch
import os

try:
    from task import input_t, output_t
except ImportError:
    input_t = torch.Tensor
    output_t = tuple[torch.Tensor, torch.Tensor]


def base_kernel(data: input_t) -> output_t:
    values, vectors = torch.linalg.eigh(data)
    return vectors, values

def householder(A: torch.Tensor):
    """Householder tridiagonalization.

    Returns (T, Q) where Q^T A Q = T.
    A: (b, n, n) symmetric matrix (only upper triangle needs to be valid).
    """
    b, n, n = A.shape
    Q = torch.eye(n, dtype=A.dtype, device=A.device)[None].repeat((b, 1, 1))  # b, n, n

    for k in range(n - 2):
        x = A[:, k + 1 :, k]  # b, n-k
        norm_x = x.norm(dim=1)  # b,
        skip = norm_x < 1e-15  # b,

        v = torch.zeros(b, n, dtype=A.dtype, device=A.device)
        v[:, k + 1 :] = x
        v[:, k + 1] += torch.sign(x[:, 0]) * norm_x
        v /= v.norm(dim=-1, keepdim=True)

        v_col = v[:, :, None]  # b, n, 1
        v_row = v_col.mT  # b, 1, n

        Q_update = 2 * (Q @ v_col) @ v_row
        Q = torch.where(skip[:, None, None], Q, Q - Q_update)
        # Q' = QH = Q(I - 2vv') = Q - 2Qvv'

        Av = A @ v_col
        vTAv = v_row @ Av
        AvvT = Av @ v_row
        A_update = 2 * (AvvT.mT + AvvT - 2 * vTAv * (v_col @ v_row))
        A = torch.where(skip[:, None, None], A, A - A_update)

        A[:, k + 2 :, k] = 0
        A[:, k, k + 2 :] = 0

    return A, Q


def custom_kernel(data: input_t) -> output_t:
    norms = torch.linalg.matrix_norm(data, keepdim=True).clamp_(1e-8, 1)
    data = data / norms
    tridiagonal, transformation = householder(data.double())
    # n_range = torch.arange(data.shape[-1], device="cuda")
    # i, j = torch.meshgrid(n_range, n_range, indexing='ij')
    # mask = (i - j).abs() <= 1
    # A = tridiagonal * mask
    # print(A[:, ~mask].max())
    
    # torch.set_printoptions(sci_mode=False, precision=6, linewidth=120)
    # print(tridiagonal[0, :7, :7])
    # print(A[0, :7, :7])
    # print(mask[:7, :7])
    values, vectors = torch.linalg.eigh(tridiagonal)
    return (transformation @ vectors).float(), values.float() * norms[:, 0]