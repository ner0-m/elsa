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
