
#pragma once
#include "elsaDefines.h"
#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "Logger.h"

namespace elsa::phantoms
{

    /*
     *  Constant for this 3 dimensional usecase.
     *  Not in header to keep the namespace clean.
     */

    // INDEX of Width vector
    static const int INDEX_A{0};
    static const int INDEX_B{1};
    static const int INDEX_C{2};

    // INDEX of Coordinates
    static const int INDEX_X{0};
    static const int INDEX_Y{1};
    static const int INDEX_Z{2};

    // INDEX for eulers coordinates
    static const int INDEX_PHI{0};
    static const int INDEX_THETA{1};
    static const int INDEX_PSI{2};

    // Fix 3d vector
    using Vec3i = Eigen::Matrix<index_t, 3, 1>;

    // Fix 3d vector
    template <typename data_t = double,
              typename = std::enable_if_t<std::is_floating_point<data_t>::value>>
    using Vec3X = Eigen::Matrix<data_t, 3, 1>;

} // namespace elsa::phantoms