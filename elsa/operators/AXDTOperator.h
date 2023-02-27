#pragma once

#include "BlockLinearOperator.h"
#include "XGIDetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "LinearOperator.h"

namespace elsa
{
    /// forward declaration for helper structs
    namespace axdt
    {
        template <typename data_t>
        struct SphericalFieldsTransform;

        template <typename data_t>
        struct SphericalFunctionInformation;
    } // namespace axdt

    /**
     * @brief The AXDT operator combines an arbitrary projector with
     * a column block linear operator, which represents the forward model
     * of the anisotropic dark-field signal
     * For details check https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.158101
     *
     * @author Matthias Wieczorek (wieczore@cs.tum.edu), original implementation
     * @author Nikola Dinev (nikola.dinev@tum.de), port to elsa
     * @author Shen Hu (shen.hu@tum.de), rewrite, integrate, port to elsa
     *
     * @tparam real_t real type
     *
     */
    template <typename data_t = real_t>
    class AXDTOperator : public LinearOperator<data_t>
    {
    public:
        /// Symmetry influence the coefficients of the underlying spherical harmonics
        /// even -> odd degrees will be deemed to have zero coefficients;
        /// odd -> vice versa;
        /// regular -> all non-zero (no simplification)
        enum Symmetry { even, odd, regular };

        using OperatorList = typename BlockLinearOperator<data_t>::OperatorList;

        /// 3D vector representing a sampling (scattering) direction
        using DirVec = Eigen::Matrix<data_t, 3, 1>;
        /// The collection of selected sampling directions
        using DirVecList = std::vector<DirVec>;

        /**
         * @brief Construct an AXDTOperator
         *
         * @param[in] domainDescriptor descriptor of the domain of the operator (the reconstructed
         * volume)
         * @param[in] rangeDescriptor descriptor of the range of the operator (the XGI Detector
         * descriptor)
         * @param[in] projector the projector representing the line integral
         * @param[in] sphericalFuncDirs vector containing all the sampling directions
         * @param[in] sphericalFuncWeights weights of the corresponding directions (must have
         * the same size)
         * @param[in] sphericalHarmonicsSymmetry symmetrical hint to simplify of the reconstructed
         * spherical harmonics coefficients
         * @param[in] sphericalHarmonicsMaxDegree maximal degree of the reconstructed spherical
         * harmonics coefficients degree
         *
         * @throw InvalidArgumentError if sizes of sphericalFuncDirs and sphericalFuncWeights do not
         * match
         */
        AXDTOperator(const VolumeDescriptor& domainDescriptor,
                     const XGIDetectorDescriptor& rangeDescriptor,
                     const LinearOperator<data_t>& projector, const DirVecList& sphericalFuncDirs,
                     const Vector_t<data_t>& sphericalFuncWeights,
                     const Symmetry& sphericalHarmonicsSymmetry,
                     index_t sphericalHarmonicsMaxDegree);

        ~AXDTOperator() override = default;

    protected:
        /// protected copy constructor; used for cloning
        AXDTOperator(const AXDTOperator& other);

        /// implement the polymorphic clone operation
        AXDTOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

        /// apply the AXDT operator
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the AXDT operator
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

    private:
        /// Ptr to a BlockLinearOperator, though saved as a ptr to LinearOperator
        std::unique_ptr<LinearOperator<data_t>> bl_op;

        /// Basing on the sampling direction, range and projection information to
        /// calculate the operator list for the BlocklinearOperator
        static OperatorList
            computeOperatorList(const XGIDetectorDescriptor& rangeDescriptor,
                                const axdt::SphericalFunctionInformation<data_t>& sf_info,
                                const LinearOperator<data_t>& projector);

        /// Compute the spherical harmonics weights for the OperatorList (i.e. the diagonal of
        /// all the [W_(k,m)] matrices in the aforementioned paper). The output matrix has
        /// shape = (J, BasisCnt), with J being (#detectorPixels x #measurements) or
        /// (#measurements). The former for cone beams and the latter for parallel beams.
        /// For BasisCnt, according to the paper, due to symmetricity typically we use
        /// degree = 4 and symmetry = even, so BasisCnt would be 15
        static std::unique_ptr<DataContainer<data_t>> computeSphericalHarmonicsWeights(
            const XGIDetectorDescriptor& rangeDescriptor,
            const axdt::SphericalFunctionInformation<data_t>& sf_info);

        /// Compute the spherical fields coefficients when the ray traces are modeled as
        /// cone beams. The output matrix has shape = (J, #samplingDirs). J has the same
        /// meaning as in computeSphericalHarmonicsWeights();
        static std::unique_ptr<DataContainer<data_t>>
            computeConeBeamWeights(const XGIDetectorDescriptor& rangeDescriptor,
                                   const axdt::SphericalFunctionInformation<data_t>& sf_info);

        /// Similar to computeConeBeamWeights(), though J has different definition and thus
        /// being faster for much less calculations. Again, the output matrix has
        /// shape = (J, #samplingDirs)
        static std::unique_ptr<DataContainer<data_t>>
            computeParallelWeights(const XGIDetectorDescriptor& rangeDescriptor,
                                   const axdt::SphericalFunctionInformation<data_t>& sf_info);
    };

    namespace axdt
    {
        /// Wrapper class for all spherical harmonics/spherical fields related information
        template <typename data_t = real_t>
        struct SphericalFunctionInformation {
            using Symmetry = typename AXDTOperator<data_t>::Symmetry;
            using DirVecList = typename AXDTOperator<data_t>::DirVecList;

            /// Vector of sampling directions
            DirVecList dirs;
            /// Weights of corresponding directions, must have the same dimension as dirs
            Vector_t<data_t> weights;

            /// Symmetry hint for the reconstructed spherical harmonics coefficients
            Symmetry symmetry;
            /// Maximal degree of the reconstructed spherical harmonics coefficients
            index_t maxDegree;
            /// Number of the reconstructed spherical harmonics coefficients for each voxel
            /// Computed based on symmetry and maxDegree
            index_t basisCnt;

            /// Compute basisCnt using symmetry and maxDegree
            static index_t calculate_basis_cnt(Symmetry symmetry, index_t maxDegree)
            {
                return (symmetry == Symmetry::regular)
                           ? (maxDegree + 1) * (maxDegree + 1)
                           : (symmetry == Symmetry::regular ? (maxDegree + 2) * (maxDegree / 2 + 1)
                                                            : (maxDegree + 1))
                                 * (maxDegree / 2 + 1);
            }

            /// Constructor for this wrapper class
            SphericalFunctionInformation(const DirVecList& sphericalFuncDirs,
                                         const Vector_t<data_t>& sphericalFuncWeights,
                                         const Symmetry& sphericalHarmonicsSymmetry,
                                         const index_t& sphericalHarmonicsMaxDegree)
                : dirs(sphericalFuncDirs),
                  weights(sphericalFuncWeights),
                  symmetry(sphericalHarmonicsSymmetry),
                  maxDegree(sphericalHarmonicsMaxDegree),
                  basisCnt(calculate_basis_cnt(symmetry, maxDegree))
            {
                if (dirs.size() != static_cast<size_t>(weights.size()))
                    throw std::invalid_argument(
                        "SphericalFunction: Sizes of direction list and weights do not match.");
            }
        };

        /// Helper class to transform weights parametrized by the sampling
        /// directions into weights parametrized by spherical harmonics.
        /// This could be done by a simple matrix multiplication between the matrix
        /// returned from computeConeBeamWeights() || computeParallelWeights()
        /// and the matrix returned from member function getForwardTransformationMatrix()
        /// of this struct
        template <typename data_t = real_t>
        struct SphericalFieldsTransform {
            using MatrixXd_t = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
            using Symmetry = typename AXDTOperator<data_t>::Symmetry;

            /// Constructor for this helper struct, taking only spherical function information
            explicit SphericalFieldsTransform(const SphericalFunctionInformation<data_t>& sf_info);

            /// Get the transform matrix for a single volume block
            /// The output shape = (#samplingDirs, BasisCnt)
            MatrixXd_t getForwardTransformationMatrix() const;

            SphericalFunctionInformation<data_t> sf_info;
            MatrixXd_t sphericalHarmonicsBasis;
        };
    } // namespace axdt
} // namespace elsa
