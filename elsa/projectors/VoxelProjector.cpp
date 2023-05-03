#include "VoxelProjector.h"
#include "Timer.h"
#include "Assertions.h"

namespace elsa
{
    template class BlobVoxelProjector<float>;
    template class BlobVoxelProjector<double>;
    template class BSplineVoxelProjector<float>;
    template class BSplineVoxelProjector<double>;
}; // namespace elsa