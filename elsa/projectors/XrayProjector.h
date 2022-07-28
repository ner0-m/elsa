#pragma once

#include "LinearOperator.h"
#include "DetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "BoundingBox.h"

namespace elsa
{
    struct ray_driven_tag {
    };
    struct voxel_driven_tag {
    };
    struct any_projection_tag {
    };

    template <typename T>
    struct XrayProjectorInnerTypes;

    /**
     * @brief Interface class for X-ray based projectors.
     *
     * For X-ray CT based methods there are mainly two different implementation methods: ray and
     * voxel driven methods. The first iterates all rays for each pose of the acquisition
     * trajectory through the volume. Along the way the ray either accumulates each visited voxels
     * values (forward) or writes to each visited voxels (backward). The second implementation
     * methods, iterates over all voxels of the volume and calculates their contribution to
     * a detector cell.
     *
     * Basically, the main difference is the outer most loop of the computation. In the ray driven
     * case, all rays are iterated and for each ray some calculations are performed. For the voxel
     * driven approach, all voxels are visited and for each some calculation is performed.
     *
     * This base class should aid in the implementation of any X-ray based projector. So, if you
     * want to implement a new projector, you'd first need to derive from this class, then the
     * following interface is required:
     * 1. specialize the `XrayProjectorInnerTypes` class with the following values:
     *   - value_type
     *   - forward_tag
     *   - backward_tag
     * 2. The class needs to implement
     *   - `_isEqual(const LinearOperator<data_t>&)` (should call `isEqual` of `LinearOperator`
     * base)
     *   - `_cloneImpl()`
     * 3. If `forward_tag` is equal to `ray_driven_tag`, then the class needs to implement:
     *   - `data_t traverseRayForward(const BoundingBox&, const RealRay_t&, const
     * DataContainer<data_t>&) const` it traverses a single ray through the given bounding box and
     * accumulates the voxel
     * 4. If `backward_tag` is equal to `ray_driven_tag`, then the class needs to implement:
     *   - `void traverseRayBackward(const BoundingBox&, const RealRay_t&, const value_type&,
     *
     * The interface for voxel based projectors is still WIP, therefore not documented here.
     *
     * The `any_projection_tag` should serve for methods, which do not fit any of the two others
     * (e.g. distance driven projectors), plus the legacy projectors, which can easily fit here for
     * now and be refactored later.
     *
     * At the time of writing, both the API is highly experimental and will evolve quickly.
     */
    template <typename D>
    class XrayProjector : public LinearOperator<typename XrayProjectorInnerTypes<D>::value_type>
    {
    public:
        using derived_type = D;
        using self_type = XrayProjector<D>;
        using inner_type = XrayProjectorInnerTypes<derived_type>;

        using value_type = typename inner_type::value_type;
        using data_t = value_type;
        using forward_tag = typename inner_type::forward_tag;
        using backward_tag = typename inner_type::backward_tag;

        using base_type = LinearOperator<data_t>;

        XrayProjector() = delete;
        ~XrayProjector() override = default;

        /// TODO: This is basically legacy, the projector itself does not need it...
        XrayProjector(const VolumeDescriptor& domain, const DetectorDescriptor& range)
            : base_type(domain, range)
        {
        }

    protected:
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override
        {
            forward(x, Ax, forward_tag{});
        }

        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override
        {
            backward(y, Aty, backward_tag{});
        }

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override
        {
            return self()._isEqual(other);
        }

        /// implement the polymorphic comparison operation
        self_type* cloneImpl() const override { return self()._cloneImpl(); }

        derived_type& self() { return static_cast<derived_type&>(*this); }
        const derived_type& self() const { return static_cast<const derived_type&>(*this); }

    private:
        void forward(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                     ray_driven_tag) const
        {
            /// the bounding box of the volume
            const BoundingBox aabb(x.getDataDescriptor().getNumberOfCoefficientsPerDimension());
            auto& detectorDesc = downcast<DetectorDescriptor>(Ax.getDataDescriptor());

            // --> loop either over every voxel that should  updated or every detector
            // cell that should be calculated
#pragma omp parallel for
            for (index_t rangeIndex = 0; rangeIndex < Ax.getSize(); ++rangeIndex) {
                // --> get the current ray to the detector center
                const auto ray = detectorDesc.computeRayFromDetectorCoord(rangeIndex);

                Ax[rangeIndex] = self().traverseRayForward(aabb, ray, x);
            }
        }

        void forward(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                     voxel_driven_tag) const
        {
            auto& detectorDesc = downcast<DetectorDescriptor>(Ax.getDataDescriptor());

            for (index_t domainIndex = 0; domainIndex < x.getSize(); ++domainIndex) {
                auto coord = x.getDataDescriptor().getCoordinateFromIndex(domainIndex);

                // TODO: Maybe we need a different interface, but I need an implementation for that
                self().forwardVoxel(coord, x[domainIndex], Ax);
            }
        }

        /// We can say nothing about it, so let it handle everything
        void forward(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                     any_projection_tag) const
        {
            const BoundingBox aabb(x.getDataDescriptor().getNumberOfCoefficientsPerDimension());
            self().forward(aabb, x, Ax);
        }

        void backward(const DataContainer<data_t>& y, DataContainer<data_t>& Aty,
                      ray_driven_tag) const
        {
            /// the bounding box of the volume
            const BoundingBox aabb(Aty.getDataDescriptor().getNumberOfCoefficientsPerDimension());
            auto& detectorDesc = downcast<DetectorDescriptor>(y.getDataDescriptor());

            // Just to be sure, zero out the result
            Aty = 0;

            // --> loop either over every voxel that should  updated or every detector
            // cell that should be calculated
#pragma omp parallel for
            for (index_t rangeIndex = 0; rangeIndex < y.getSize(); ++rangeIndex) {
                // --> get the current ray to the detector center
                const auto ray = detectorDesc.computeRayFromDetectorCoord(rangeIndex);

                self().traverseRayBackward(aabb, ray, y[rangeIndex], Aty);
            }
        }

        void backward(const DataContainer<data_t>& y, DataContainer<data_t>& Aty,
                      voxel_driven_tag) const
        {
            auto& detectorDesc = downcast<DetectorDescriptor>(y.getDataDescriptor());

            for (index_t domainIndex = 0; domainIndex < Aty.getSize(); ++domainIndex) {
                auto coord = Aty.getDataDescriptor().getCoordinateFromIndex(domainIndex);

                // TODO: Maybe we need a different interface, but I need an implementation for that
                Aty[domainIndex] = self().backwardVoxel(coord, y);
            }
        }

        void backward(const DataContainer<data_t>& y, DataContainer<data_t>& Aty,
                      any_projection_tag) const
        {
            const BoundingBox aabb(Aty.getDataDescriptor().getNumberOfCoefficientsPerDimension());
            self().backward(aabb, y, Aty);
        }
    };

} // namespace elsa
