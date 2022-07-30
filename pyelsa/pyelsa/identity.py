from scipy.sparse.linalg import LinearOperator

__all__ = ["Identity"]

class Identity(LinearOperator):
    def __init__(self, shape, dtype=None):
        super().__init__(dtype, shape)

    def _matvec(self, x):
        return x

    def _matmat(self, X):
        return X

    def _rmatvec(self, x):
        return x

    def _rmatmat(self, X):
        return X

    def _adjoint(self):
        return self
