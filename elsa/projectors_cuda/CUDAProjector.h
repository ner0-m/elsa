#include "LinearOperator.h"
#include "BoundingBox.h"

#include <memory>
#include <tuple>

namespace elsa
{
    template <typename data_t>
    class CUDAProjector : public LinearOperator<data_t>
    {
    public:
        /**
         * \brief Determine which part of the volume is responsible for the data measured in the
         * specified part of the image
         *
         * \param[in] startCoordinate the start coordinate of the image part
         * \param[in] endCoordinate the end coordinate of the image part
         *
         * \returns a pair containing the constrained CUDAProjector and the BoundingBox of the
         * reponsible part of the volume
         */
        virtual std::pair<std::unique_ptr<CUDAProjector<data_t>>, BoundingBox>
            constrainProjectionSpace(const IndexVector_t startCoordinate,
                                     const IndexVector_t endCoordinate) = 0;

    protected:
        CUDAProjector(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor)
            : LinearOperator<data_t>(domainDescriptor, rangeDescriptor)
        {
        }
    };
} // namespace elsa