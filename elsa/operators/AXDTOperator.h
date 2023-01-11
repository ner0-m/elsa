#pragma once

#include "BlockLinearOperator.h"

#include "XGIDetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "LinearOperator.h"

namespace elsa
{
    /// forward declaration for helper struct
    namespace axdt
    {
        template <typename data_t>
        struct SphericalFieldsTransform;

        template <typename data_t>
        struct SphericalFunctionInformation;
    } // namespace axdt

    /**
     * @brief The AXDT operator is a column block linear operator.
     *
     * @author Matthias Wieczorek (wieczore@cs.tum.edu), original implementation
     * @author Nikola Dinev (nikola.dinev@tum.de), port to elsa
     * @author Shen Hu (shen.hu@tum.de), rewrite, integrate, port to elsa
     *
     * @tparam real_t real type
     *
     * For details check \cite Vogel:2015hn
     *
     */
    template <typename data_t = real_t>
    class AXDTOperator : public BlockLinearOperator<data_t>
    {
    public:
        /// Symmetry influence the coefficients of the underlying spherical harmonics
        /// even -> odd degrees will have zero coefficients; odd -> vice versa; regular -> all
        /// non-zero
        enum Symmetry { even, odd, regular };

        typedef BlockLinearOperator<data_t> B;
        using typename B::OperatorList;

        using DirVec = Eigen::Matrix<data_t, 3, 1>;
        using DirVecList = std::vector<DirVec>;
        using WeightVec = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        /**
         * @brief Construct an AXDTOperator
         *
         * @param[in] domainDescriptor descriptor of the domain of the operator (the reconstructed
         * volume)
         * @param[in] rangeDescriptor descriptor of the range of the operator (the XGI Detector
         * descriptor)
         * @param[in] projector the projector representing the line integral
         * @param[in] sphericalFuncDirs vector of the sampling directions
         * @param[in] sphericalFuncWeights weightings of the corresponding directions
         * @param[in] sphericalHarmonicsSymmetry symmetry of the reconstructed spherical harmonics
         * coefficients
         * @param[in] sphericalHarmonicsMaxDegree maximal degree of the reconstructed spherical
         * harmonics coefficients degree
         *
         * @throw InvalidArgumentError if sizes of sphericalFuncDirs and sphericalFuncWeights do not
         * match
         */
        AXDTOperator(const VolumeDescriptor& domainDescriptor,
                     const XGIDetectorDescriptor& rangeDescriptor,
                     const LinearOperator<data_t>& projector, const DirVecList& sphericalFuncDirs,
                     const WeightVec& sphericalFuncWeights,
                     const Symmetry& sphericalHarmonicsSymmetry,
                     const index_t& sphericalHarmonicsMaxDegree);

        ~AXDTOperator() override = default;

    protected:
        /// protected copy constructor; used for cloning
        AXDTOperator(const AXDTOperator& other);

        /// implement the polymorphic clone operation
        AXDTOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        static OperatorList
            computeOperatorList(const XGIDetectorDescriptor& rangeDescriptor,
                                const axdt::SphericalFunctionInformation<data_t>& sf_info,
                                const LinearOperator<data_t>& projector);

        static std::unique_ptr<DataContainer<data_t>> computeSphericalHarmonicsWeights(
            const XGIDetectorDescriptor& rangeDescriptor,
            const axdt::SphericalFunctionInformation<data_t>& sf_info);

        static std::unique_ptr<DataContainer<data_t>>
            computeConeBeamWeights(const XGIDetectorDescriptor& rangeDescriptor,
                                   const axdt::SphericalFunctionInformation<data_t>& sf_info);

        static std::unique_ptr<DataContainer<data_t>>
            computeParallelWeights(const XGIDetectorDescriptor& rangeDescriptor,
                                   const axdt::SphericalFunctionInformation<data_t>& sf_info);
    };

    namespace axdt
    {
        /// Wrapper class for all spherical harmonics/spherical function related information
        template <typename data_t = real_t>
        struct SphericalFunctionInformation {
            using Symmetry = typename AXDTOperator<data_t>::Symmetry;
            using DirVec = typename AXDTOperator<data_t>::DirVec;
            using DirVecList = typename AXDTOperator<data_t>::DirVecList;
            using WeightVec = typename AXDTOperator<data_t>::WeightVec;

            /// Vector of sampling directions
            DirVecList dirs;
            /// Weights of corresponding directions, must have the same dimension as dirs
            WeightVec weights;

            /// Symmetry of the reconstructed spherical harmonics coefficients
            Symmetry symmetry;
            /// Maximal degree of the reconstructed spherical harmonics coefficients
            index_t maxDegree;
            /// Number of the reconstructed spherical harmonics coefficients
            /// Computed based on symmetry and maxDegree
            index_t basisCnt;

            /// Constructor for this wrapper class
            SphericalFunctionInformation(const DirVecList& sphericalFuncDirs,
                                         const WeightVec& sphericalFuncWeights,
                                         const Symmetry& sphericalHarmonicsSymmetry,
                                         const index_t& sphericalHarmonicsMaxDegree)
                : dirs(sphericalFuncDirs),
                  weights(sphericalFuncWeights),
                  symmetry(sphericalHarmonicsSymmetry),
                  maxDegree(sphericalHarmonicsMaxDegree),
                  basisCnt((symmetry == Symmetry::regular)
                               ? (maxDegree + 1) * (maxDegree + 1)
                               : (symmetry == Symmetry::regular
                                      ? (maxDegree + 2) * (maxDegree / 2 + 1)
                                      : (maxDegree + 1))
                                     * (maxDegree / 2 + 1))
            {
                if (dirs.size() != static_cast<size_t>(weights.size()))
                    throw std::invalid_argument(
                        "SphericalFunction: Sizes of directions list and weights do not match.");
            }
        };

        /// Helper class to transform weights parametrized by spherical functions with sampling
        /// directions into weights parametrized by spherical harmonics
        template <typename data_t = real_t>
        struct SphericalFieldsTransform {
            using MatrixXd_t = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
            using Symmetry = typename AXDTOperator<data_t>::Symmetry;

            /// Constructor for the helper class
            explicit SphericalFieldsTransform(const SphericalFunctionInformation<data_t>& sf_info);

            // const MatrixXd_t& getInverseTransformationMatrix() const;
            /// get the transform matrix for a single volume block
            MatrixXd_t getForwardTransformationMatrix() const;

            SphericalFunctionInformation<data_t> sf_info;
            MatrixXd_t sphericalHarmonicsBasis;
        };
    } // namespace axdt
} // namespace elsa
