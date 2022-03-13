#pragma once

#include "elsaDefines.h"
#include "Timer.h"
#include "Luts.hpp"
#include "SliceTraversal.h"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "DataContainer.h"
#include "BoundingBox.h"
#include "Logger.h"
#include "Blobs.h"
#include "CartesianIndices.h"

#include "spdlog/fmt/bundled/core.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"

namespace elsa
{
    template <typename data_t, typename Derived>
    class LutProjector : public LinearOperator<data_t>
    {
    public:
        LutProjector(const VolumeDescriptor& domainDescriptor,
                     const DetectorDescriptor& rangeDescriptor)
            : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
              _boundingBox{domainDescriptor.getNumberOfCoefficientsPerDimension()},
              _detectorDescriptor(static_cast<DetectorDescriptor&>(*_rangeDescriptor)),
              _volumeDescriptor(static_cast<VolumeDescriptor&>(*_domainDescriptor))
        {
            // sanity checks
            auto dim = _domainDescriptor->getNumberOfDimensions();
            if (dim < 2 || dim > 3) {
                throw InvalidArgumentError("LutProjector: only supporting 2d/3d operations");
            }

            if (dim != _rangeDescriptor->getNumberOfDimensions()) {
                throw InvalidArgumentError(
                    "LutProjector: domain and range dimension need to match");
            }

            if (_detectorDescriptor.getNumberOfGeometryPoses() == 0) {
                throw InvalidArgumentError("LutProjector: rangeDescriptor without any geometry");
            }
        }

        /// default destructor
        ~LutProjector() override = default;

        Derived& self() { return static_cast<Derived&>(*this); }

        const Derived& self() const { return static_cast<const Derived&>(*this); }

    protected:
        /// apply the binary method (i.e. forward projection)
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override
        {
            Timer t("LutProjector", "apply");

            // Be sure to zero out the result
            Ax = 0;

            const auto sizeRange = Ax.getSize();
            const auto volume_shape = x.getDataDescriptor().getNumberOfCoefficientsPerDimension();

            const IndexVector_t lower = _boundingBox._min.template cast<index_t>();
            const IndexVector_t upper = _boundingBox._max.template cast<index_t>();
            const auto support = self().support();

#pragma omp parallel for
            // Loop over all the poses, and for each pose loop over all detector pixels
            for (index_t rangeIndex = 0; rangeIndex < sizeRange; ++rangeIndex) {
                // --> get the current ray to the detector center
                auto ray = _detectorDescriptor.computeRayFromDetectorCoord(rangeIndex);

                index_t leadingdir = 0;
                ray.direction().array().cwiseAbs().maxCoeff(&leadingdir);

                auto rangeVal = Ax[rangeIndex];

                // Expand bounding box as rays have larger support now
                auto aabb = _boundingBox;
                aabb._min.array() -= support;
                aabb._min[leadingdir] += support;

                aabb._max.array() += support;
                aabb._max[leadingdir] -= support;

                // --> setup traversal algorithm
                SliceTraversal traversal(aabb, ray);

                for (const auto [curPos, curVoxel, t] : traversal) {
                    for (auto neighbour :
                         neighbours_in_slice(curVoxel, support, leadingdir, lower, upper)) {
                        // Correct position, such that the distance is still correct
                        auto correctedPos = neighbour.template cast<real_t>().array() + 0.5;

                        const auto distance = ray.distance(correctedPos);
                        const auto weight = self().weight(distance);

                        rangeVal += weight * x.at(neighbour);
                    }
                }

                Ax[rangeIndex] = rangeVal;
            }
        }

        /// apply the adjoint of the binary method (i.e. backward projection)
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override
        {
            Timer t("LutProjector", "apply");

            const auto sizeRange = y.getSize();
            Aty = 0;

            const auto shape = _domainDescriptor->getNumberOfCoefficientsPerDimension();

            const IndexVector_t lower = _boundingBox._min.template cast<index_t>();
            const IndexVector_t upper = _boundingBox._max.template cast<index_t>();
            const auto support = self().support();

#pragma omp parallel for
            // Loop over all the poses, and for each pose loop over all detector pixels
            for (index_t rangeIndex = 0; rangeIndex < sizeRange; ++rangeIndex) {
                // --> get the current ray to the detector center (from reference to
                // DetectorDescriptor)
                auto ray = _detectorDescriptor.computeRayFromDetectorCoord(rangeIndex);

                index_t leadingdir = 0;
                ray.direction().array().cwiseAbs().maxCoeff(&leadingdir);

                // Expand bounding box as rays have larger support now
                auto aabb = _boundingBox;
                aabb._min.array() -= support;
                aabb._min[leadingdir] += support;

                aabb._max.array() += support;
                aabb._max[leadingdir] -= support;

                // --> setup traversal algorithm
                SliceTraversal traversal(aabb, ray);

                const auto val = y[rangeIndex];

                for (const auto [curPos, curVoxel, t] : traversal) {
                    for (auto neighbour :
                         neighbours_in_slice(curVoxel, support, leadingdir, lower, upper)) {
                        // Correct position, such that the distance is still correct
                        auto correctedPos = neighbour.template cast<real_t>().array() + 0.5;

                        const auto distance = ray.distance(correctedPos);
                        const auto weight = self().weight(distance);
#pragma omp atomic
                        Aty(neighbour) += weight * val;
                    }
                }
            }
        }

        /// implement the polymorphic clone operation
        LutProjector<data_t, Derived>* cloneImpl() const override
        {
            return new LutProjector(_volumeDescriptor, _detectorDescriptor);
        }

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override
        {
            if (!LinearOperator<data_t>::isEqual(other))
                return false;

            auto otherOp = downcast_safe<LutProjector>(&other);
            return static_cast<bool>(otherOp);
        }

    private:
        /// the bounding box of the volume
        BoundingBox _boundingBox;

        /// Lift from base class
        using LinearOperator<data_t>::_domainDescriptor;

        /// Lift from base class
        using LinearOperator<data_t>::_rangeDescriptor;

    protected:
        /// Reference to DetectorDescriptor stored in LinearOperator
        DetectorDescriptor& _detectorDescriptor;

        /// Reference to VolumeDescriptor stored in LinearOperator
        VolumeDescriptor& _volumeDescriptor;
    };

    template <typename data_t = real_t>
    class BlobProjector : public LutProjector<data_t, BlobProjector<data_t>>
    {
    public:
        BlobProjector(data_t radius, data_t alpha, data_t order,
                      const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor);

        BlobProjector(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor);

        data_t weight(data_t distance) const { return lut_(distance); }

        index_t support() const { return static_cast<index_t>(std::ceil(lut_.radius())); }

        /// implement the polymorphic clone operation
        BlobProjector<data_t>* cloneImpl() const override
        {
            return new BlobProjector(_volumeDescriptor, _detectorDescriptor);
        }

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override
        {
            if (!Base::isEqual(other))
                return false;

            auto otherOp = downcast_safe<BlobProjector>(&other);
            return static_cast<bool>(otherOp);
        }

    private:
        ProjectedBlobLut<data_t, 100> lut_;

        using Base = LutProjector<data_t, BlobProjector<data_t>>;
        using Base::_detectorDescriptor;
        using Base::_volumeDescriptor;
    };
} // namespace elsa
