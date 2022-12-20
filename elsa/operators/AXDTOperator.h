#pragma once

#include "BlockLinearOperator.h"

#include "XGIDetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * \brief The XTT operator is a column block linear operators.
     *
     * \author Matthias Wieczorek (wieczore@cs.tum.edu), original implementation
     * \author Nikola Dinev (nikola.dinev@tum.de), port to elsa
     * \author Shen Hu (shen.hu@tum.de), rewrite, integrate, port to elsa
     *
     * \tparam real_t real type
     *
     * For details check \cite Vogel:2015hn
     *
     */
    template <typename data_t = real_t>
    class AXDTOperator : public BlockLinearOperator<data_t>
    {
    public:
        enum Symmetry { even, odd, regular };

    private:
        typedef BlockLinearOperator<data_t> B;
        using typename B::OperatorList;

        using DirVec = Eigen::Matrix<data_t, 3, 1>;
        using DirVecList = std::vector<DirVec>;
        using WeightVec = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        struct SphericalFunctionInformation {
            DirVecList dirs;
            WeightVec weights;

            Symmetry symmetry;
            index_t maxDegree;
            index_t basisCnt;

            SphericalFunctionInformation(const DirVecList& sphericalFuncDirs,
                                         const WeightVec& sphericalFuncWeights,
                                         const Symmetry& sphericalHarmonicsSymmetry,
                                         const index_t& sphericalHarmonicsMaxDegree)
                : dirs(sphericalFuncDirs),
                  weights(sphericalFuncWeights),
                  symmetry(sphericalHarmonicsSymmetry),
                  maxDegree(sphericalHarmonicsMaxDegree),
                  basisCnt((symmetry == regular)
                               ? (maxDegree + 1) * (maxDegree + 1)
                               : (symmetry == odd ? (maxDegree + 2) * (maxDegree / 2 + 1)
                                                  : (maxDegree + 1))
                                     * (maxDegree / 2 + 1))
            {
                if (dirs.size() != static_cast<size_t>(weights.size()))
                    throw std::invalid_argument(
                        "SphericalFunction: Sizes of directions list and weights do not match.");
            }
        };

        struct SphericalFieldsTransform {
            using MatrixXd_t = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;

            explicit SphericalFieldsTransform(const SphericalFunctionInformation& sf_info);

            // const MatrixXd_t& getInverseTransformationMatrix() const;
            MatrixXd_t getForwardTransformationMatrix() const;

            SphericalFunctionInformation sf_info;
            MatrixXd_t sphericalHarmonicsBasis;
        };

    public:
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
        static OperatorList computeOperatorList(const XGIDetectorDescriptor& rangeDescriptor,
                                                const SphericalFunctionInformation& sf_info,
                                                const LinearOperator<data_t>& projector);

        /// domainDescriptor gives the sphericalHarmonics descriptor
        /// sphericalFunctionDescriptor gives the sampling pattern used to compute the spherical
        /// harmonics transform rangeDescriptor gives the DciProjDescriptor
        static std::unique_ptr<DataContainer<data_t>>
            computeSphericalHarmonicsWeights(const XGIDetectorDescriptor& rangeDescriptor,
                                             const SphericalFunctionInformation& sf_info);

        static std::unique_ptr<DataContainer<data_t>>
            computeConeBeamWeights(const XGIDetectorDescriptor& rangeDescriptor,
                                   const SphericalFunctionInformation& sf_info);

        static std::unique_ptr<DataContainer<data_t>>
            computeParallelWeights(const XGIDetectorDescriptor& rangeDescriptor,
                                   const SphericalFunctionInformation& sf_info);
    };
} // namespace elsa
