import torch


def sor(A: torch.Tensor, b: torch.Tensor, num_iter: int = 10, x0=None, omega=1.5) -> torch.Tensor:

    D = A.diag().diag()
    L = A.tril(diagonal=-1)
    U = A.triu(diagonal=1)
    I = torch.eye(A.shape[0])

    M1 = (I + omega * D.inverse() @ L).inverse()
    M2 = (1 - omega) * I - omega * D.inverse() @ U
    M = M1 @ M2

    Nb = omega * (D + omega * L).inverse() @ b

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0

    for i in range(num_iter):
        x = M @ x + Nb

    return x


class SOR(object):
    def __init__(self, A: torch.Tensor, omega=1.9):
        D = A.diag().diag().cuda()
        L = A.tril(diagonal=-1).cuda()
        U = A.triu(diagonal=1).cuda()
        I = torch.eye(A.shape[0]).cuda()

        M1 = (I + omega * D.inverse() @ L).inverse()
        M2 = (1 - omega) * I - omega * D.inverse() @ U
        self.N = omega * (D + omega * L).inverse()
        self.M = M1 @ M2

    def __call__(self, A: torch.Tensor, b: torch.Tensor, num_iter: int = 10, x0=None):
        Nb = self.N @ b

        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0

        for i in range(num_iter):
            x = self.M @ x + Nb

        return x
