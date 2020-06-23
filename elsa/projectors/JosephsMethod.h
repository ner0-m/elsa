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
     * \brief Operator representing the discretized X-ray transform in 2d/3d using Joseph's method.
     *
     * \author Christoph Hahn - initial implementation
     * \author Maximilian Hornung - modularization
     * \author Nikola Dinev - fixes
     *
     * \tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * The volume is traversed along the rays as specified by the Geometry. For interior voxels
     * the sampling point is located in the middle of the two planes orthogonal to the main
     * direction of the ray. For boundary voxels the sampling point is located at the center of the
     * ray intersection with the voxel.
     *
     * The geometry is represented as a list of projection matrices (see class Geometry), one for
     * each acquisition pose.
     *
     * Two modes of interpolation are available:
     * NN (NearestNeighbours) takes the value of the pixel/voxel containing the point
     * LINEAR performs linear interpolation for the nearest 2 pixels (in 2D)
     * or the nearest 4 voxels (in 3D).
     *
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     * This projector is matched.
     */
    template <typename data_t = real_t>
    class JosephsMethod : public LinearOperator<data_t>
    {
    public:
        /// Available interpolation modes
        enum class Interpolation { NN, LINEAR };

        /**
         * \brief Constructor for Joseph's traversal method.
         *
         * \param[in] domainDescriptor describing the domain of the operator (the volume)
         * \param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * \param[in] interpolation enum specifying the interpolation mode
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        JosephsMethod(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor,
                      Interpolation interpolation = Interpolation::LINEAR);

        /// default destructor
        ~JosephsMethod() = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        JosephsMethod(const JosephsMethod<data_t>&) = default;

        /// apply Joseph's method (i.e. forward projection)
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of Joseph's method (i.e. backward projection)
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        JosephsMethod<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// the bounding box of the volume
        BoundingBox _boundingBox;

        /// Reference to DetectorDescriptor stored in LinearOperator
        DetectorDescriptor& _detectorDescriptor;

        /// Reference to VolumeDescriptor stored in LinearOperator
        VolumeDescriptor& _volumeDescriptor;

        /// the interpolation mode
        Interpolation _interpolation;

        /// the traversal routine (for both apply/applyAdjoint)
        template <bool adjoint>
        void traverseVolume(const DataContainer<data_t>& vector,
                            DataContainer<data_t>& result) const;

        /// convenience typedef for ray
        using Ray = Eigen::ParametrizedLine<real_t, Eigen::Dynamic>;

        /**
         * \brief  Linear interpolation, works in any dimension
         *
         * \tparam adjoint true for backward projection, false for forward
         *
         * \param[in] vector the input DataContainer
         * \param[out] result DataContainer for results
         * \param[in] fractionals the fractional numbers used in the interpolation
         * \param[in] domainDim number of dimensions
         * \param[in] currentVoxel coordinates of voxel for interpolation
         * \param[in] intersection weighting for the interpolated values depending on the incidence
         * angle
         * \param[in] from index of the current vector position
         * \param[in] to index of the current result position
         * \param[in] mainDirection specifies the main direction of the ray
         */
        template <bool adjoint>
        void linear(const DataContainer<data_t>& vector, DataContainer<data_t>& result,
                    const RealVector_t& fractionals, index_t domainDim,
                    const IndexVector_t& currentVoxel, float intersection, index_t from, index_t to,
                    int mainDirection) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
} // namespace elsa
