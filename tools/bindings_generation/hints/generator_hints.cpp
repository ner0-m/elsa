#include "hints_base.h"

#include "CircleTrajectoryGenerator.h"

#include <pybind11/pybind11.h>
#include <functional>

namespace elsa
{
    namespace py = pybind11;

    class CircleTrajectoryGeneratorHints : public ClassHints<CircleTrajectoryGenerator>
    {
    public:
        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            c.def_static(
                "createTrajectory",
                [](index_t poses const DataDescriptor& volumeDescriptor, index_t arcDegrees,
                   index_t numberOfPoses, real_t sourceToCenter, real_t centerToDetector) {
                    return std::make_unique<DetectorDescriptor>(
                        poses, volumeDescriptor, arcDegrees, sourceToCenter, centerToDetector);
                })
        }
    };
} // namespace elsa
