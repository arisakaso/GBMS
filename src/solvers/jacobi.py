import torch


def jacobi(A: torch.Tensor, b: torch.Tensor, num_iter: int = 10, x0=None) -> torch.Tensor:
    """jacobi method.

    Args:
        A (torch.Tensor): [description]
        b (torch.Tensor): [description]
        N (int, optional): [description]. Defaults to 10.
        x0 ([type], optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]

    Examples:
        >>> A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        >>> b = torch.tensor([3.0, 3.0])
        >>> jacobi(A, b, num_iter=1000)
        tensor([1., 1.])
    """

    D = A.diag().diag()
    LU = A.tril(diagonal=-1) + A.triu(diagonal=1)
    D_inv = D.diag().reciprocal().diag()
    D_inv_b = D_inv @ b
    D_inv_LU = D_inv @ LU

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0

    for i in range(num_iter):
        x = D_inv_b - D_inv_LU @ x

    return x


class Jacobi(object):
    """jacobi method.

    Args:
        A (torch.Tensor): [description]
        b (torch.Tensor): [description]
        N (int, optional): [description]. Defaults to 10.
        x0 ([type], optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]

    Examples:
        >>> A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        >>> b = torch.tensor([3.0, 3.0])
        >>> jacobi(A, b, num_iter=1000)
        tensor([1., 1.])
    """

    def __init__(self, A: torch.Tensor, tensor_type=torch.cuda.DoubleTensor):
        self.tensor_type = tensor_type
        A = A.type(tensor_type)
        D = A.diag().diag()
        LU = A.tril(diagonal=-1) + A.triu(diagonal=1)
        self.D_inv = D.inverse()
        self.D_inv_LU = self.D_inv @ LU

    def __call__(self, A: torch.Tensor, b: torch.Tensor, num_iter: int = 10, x0=None):
        D_inv_b = self.D_inv @ b.type(self.tensor_type)

        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0
        x = x.type(self.tensor_type)

        for i in range(num_iter):
            x = D_inv_b - self.D_inv_LU @ x

        return x
