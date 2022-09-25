#pragma once

#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"

#include <optional>

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
        static std::pair<IndexVector_t, RealVector_t> calculateSizeAndSpacingPerGeometry(
            const DataDescriptor& volDescr, index_t numberOfPoses,
            std::optional<IndexVector_t> detectorSize = std::nullopt,
            std::optional<RealVector_t> detectorSpacing = std::nullopt);
    };
} // namespace elsa
