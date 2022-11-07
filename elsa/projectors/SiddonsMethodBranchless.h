#pragma once

#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"

#include "XrayProjector.h"

#include <vector>
#include <utility>

#include <Eigen/Geometry>

namespace elsa
{
    template <typename data_t = real_t>
    class SiddonsMethodBranchless;

    template <typename data_t>
    struct XrayProjectorInnerTypes<SiddonsMethodBranchless<data_t>> {
        using value_type = data_t;
        using forward_tag = ray_driven_tag;
        using backward_tag = ray_driven_tag;
    };
    /**
     * @brief Operator representing the discretized X-ray transform in 2d/3d using Siddon's method.
     *
     * The volume is traversed along the rays as specified by the Geometry. Each ray is traversed in
     * a contiguous fashion (i.e. along long voxel borders, not diagonally) and each traversed
     * voxel is counted as a hit with weight according to the length of the path of the ray through
     * the voxel.
     *
     * The geometry is represented as a list of projection matrices (see class Geometry), one for
     * each acquisition pose.
     *
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     * This projector is matched.
     *
     * @author David Frank - initial code, refactor to XrayProjector
     * @author Nikola Dinev - modularization, fixes
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     */
    template <typename data_t>
    class SiddonsMethodBranchless : public XrayProjector<SiddonsMethodBranchless<data_t>>
    {
    public:
        using self_type = SiddonsMethodBranchless<data_t>;
        using base_type = XrayProjector<self_type>;
        using value_type = typename base_type::value_type;
        using forward_tag = typename base_type::forward_tag;
        using backward_tag = typename base_type::backward_tag;

        /**
         * @brief Constructor for Siddon's method traversal.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        SiddonsMethodBranchless(const VolumeDescriptor& domainDescriptor,
                                const DetectorDescriptor& rangeDescriptor);

        /// default destructor
        ~SiddonsMethodBranchless() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        SiddonsMethodBranchless(const SiddonsMethodBranchless<data_t>&) = default;

    private:
        /// implement the polymorphic clone operation
        SiddonsMethodBranchless<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

        data_t traverseRayForward(BoundingBox aabb, const RealRay_t& ray,
                                  const DataContainer<data_t>& x) const;

        void traverseRayBackward(BoundingBox aabb, const RealRay_t& ray,
                                 const value_type& detectorValue, DataContainer<data_t>& Aty) const;

        template <int dim>
        data_t doTraverseRayForward(BoundingBox aabb, const RealRay_t& ray,
                                    const DataContainer<data_t>& x,
                                    const DataDescriptor& domain) const;
        template <int dim>
        void doTraverseRayBackward(BoundingBox aabb, const RealRay_t& ray,
                                   const value_type& detectorValue,
                                   DataContainer<data_t>& Aty) const;
        friend class XrayProjector<self_type>;
    };

} // namespace elsa
