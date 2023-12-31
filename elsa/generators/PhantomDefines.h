
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

    template <typename data_t>
    void fillRotationMatrix(Vec3X<data_t> eulers, Eigen::Matrix<data_t, 3, 3>& rot);

    /**
     * @brief orientation of the cylinder along one of the axis
     */
    enum class Orientation { X_AXIS = 0, Y_AXIS = 1, Z_AXIS = 2 };

    std::string getString(Orientation o);

    enum class Blending { OVERWRITE = 0, ADDITION = 1 };

    template <Blending b, typename data_t>
    inline constexpr void blend(DataContainer<data_t>& dc, index_t index, data_t amplit)
    {
        // TODO maybe use `data_t& d` instead of `DataContainer<data_t>$ dc`
        if constexpr (b == Blending::ADDITION) {
            dc[index] += amplit;
        } else if constexpr (b == Blending::OVERWRITE) {
            dc[index] = amplit;
        } else {
            throw Error("Wrong Blending");
        }
    }

} // namespace elsa::phantoms
