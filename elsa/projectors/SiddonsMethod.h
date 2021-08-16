#pragma once

#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"

#include <vector>
#include <utility>

#include <Eigen/Geometry>

namespace elsa
{
    /**
     * @brief Operator representing the discretized X-ray transform in 2d/3d using Siddon's method.
     *
     * @author David Frank - initial code
     * @author Nikola Dinev - modularization, fixes
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * The volume is traversed along the rays as specified by the Geometry. Each ray is traversed in
     * a continguous fashion (i.e. along long voxel borders, not diagonally) and each traversed
     * voxel is counted as a hit with weight according to the length of the path of the ray through
     * the voxel.
     *
     * The geometry is represented as a list of projection matrices (see class Geometry), one for
     * each acquisition pose.
     *
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     * This projector is matched.
     */
    template <typename data_t = real_t>
    class SiddonsMethod : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for Siddon's method traversal.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        SiddonsMethod(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor);

        /// default destructor
        ~SiddonsMethod() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        SiddonsMethod(const SiddonsMethod<data_t>&) = default;

        /// apply Siddon's method (i.e. forward projection)
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of Siddon's method (i.e. backward projection)
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        SiddonsMethod<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// the bounding box of the volume
        BoundingBox _boundingBox;

        /// Reference to DetectorDescriptor stored in LinearOperator
        DetectorDescriptor& _detectorDescriptor;

        /// Reference to VolumeDescriptor stored in LinearOperator
        VolumeDescriptor& _volumeDescriptor;

        /// the traversal routine (for both apply/applyAdjoint)
        template <bool adjoint>
        void traverseVolume(const DataContainer<data_t>& vector,
                            DataContainer<data_t>& result) const;

        /// convenience typedef for ray
        using Ray = Eigen::ParametrizedLine<real_t, Eigen::Dynamic>;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };

} // namespace elsa
