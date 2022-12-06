from .pyelsa_core import *
from .pyelsa_functionals import *
from .pyelsa_generators import *
from .pyelsa_io import *
from .pyelsa_operators import *
from .pyelsa_problems import *
from .pyelsa_proximity_operators import *
from .pyelsa_solvers import *
from .pyelsa_projectors import *

try:
    from .pyelsa_projectors_cuda import *

    importedCudaProjectors = True
except ModuleNotFoundError:
    print("pyelsa not build with CUDA projector support")
    importedCudaProjectors = False


def cudaProjectorsEnabled():
    return importedCudaProjectors


# TODO: Move to submodule, but I never seem to understand how Python works...
def to_scipy(elsaOp):
    """Convert an elsa 'LinearOperator' to a SciPy conforming interface"""
    from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator

    class WrapElsaLinearOperator(SciPyLinearOperator):
        def __init__(self, elsaOp: LinearOperator):
            import numpy as np

            self.shape = (
                elsaOp.getRangeDescriptor().getNumberOfCoefficients(),
                elsaOp.getDomainDescriptor().getNumberOfCoefficients(),
            )

            # TODO: I'm not sure hot to check this
            self.dtype = np.float32
            self.Op = elsaOp
            pass

        def _matvec(self, x):
            if isinstance(x, DataContainer):
                # If x is already a DataContainer we just leave it
                return self.Op.apply(x)
            else:
                # Assume x is an NumPy array
                import numpy as np

                dc = DataContainer(x, self.Op.getDomainDescriptor())
                return np.asarray(self.Op.apply(dc))

        def _rmatvec(self, y):
            if isinstance(y, DataContainer):
                # If y is already a DataContainer we just leave it
                return self.Op.applyAdjoint(y)
            else:
                # Assume x is an NumPy array
                import numpy as np

                dc = DataContainer(y, self.Op.getRangeDescriptor())
                return np.asarray(self.Op.applyAdjoint(dc))

    op = WrapElsaLinearOperator(elsaOp)
    return op
