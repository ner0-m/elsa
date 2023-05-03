#include "PhaseContrastProjector.h"

namespace elsa
{
    template class PhaseContrastBSplineVoxelProjector<float>;
    template class PhaseContrastBSplineVoxelProjector<double>;
    template class PhaseContrastBlobVoxelProjector<float>;
    template class PhaseContrastBlobVoxelProjector<double>;
}; // namespace elsa
