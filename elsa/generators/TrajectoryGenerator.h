#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"

namespace elsa
{
    /**
     * @brief Parent class for other trajectory generator classes. Currently contains extracted
     * duplicate code.
     *
     * @author Andi Braimllari - initial code
     */
    class TrajectoryGenerator
    {
    protected:
        static std::pair<IndexVector_t, RealVector_t>
            calculateSizeAndSpacingPerGeometry(const DataDescriptor& volDescr,
                                               index_t numberOfPoses);
    };
} // namespace elsa
