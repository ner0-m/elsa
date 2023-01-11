/**
 * @file test_AXDTOperator.cpp
 *
 * @brief Tests for the ADXTOperator
 *
 * @author Shen Hu - main code
 */

#include "doctest/doctest.h"
#include "AXDTOperator.h"
#include "IdenticalBlocksDescriptor.h"
#include "Geometry.h"
#include "StrongTypes.h"
#include "DataContainer.h"
#include "Identity.h"

using namespace elsa;
using namespace doctest;
using namespace elsa::axdt;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("AXDTOperator: Testing SphericalFieldsTransform", data_t, float, double)
{
    IndexVector_t volSize(2);
    volSize << 2, 2;
    VolumeDescriptor volDesc{volSize};

    typename AXDTOperator<data_t>::DirVecList samplingPattern;
    samplingPattern.emplace_back(0.50748, -0.3062, 0.80543);
    samplingPattern.emplace_back(-0.3062, 0.80543, 0.50748);
    samplingPattern.emplace_back(-0.50748, 0.3062, 0.80543);
    samplingPattern.emplace_back(0.80543, 0.50748, -0.3062);
    samplingPattern.emplace_back(0.3062, 0.80543, -0.50748);
    samplingPattern.emplace_back(0.80543, -0.50748, 0.3062);
    samplingPattern.emplace_back(0.3062, -0.80543, 0.50748);
    samplingPattern.emplace_back(-0.80543, -0.50748, -0.3062);
    samplingPattern.emplace_back(-0.3062, -0.80543, -0.50748);
    samplingPattern.emplace_back(-0.80543, 0.50748, 0.3062);
    samplingPattern.emplace_back(0.50748, 0.3062, -0.80543);
    samplingPattern.emplace_back(-0.50748, -0.3062, -0.80543);
    samplingPattern.emplace_back(0.62636, -0.24353, -0.74052);
    samplingPattern.emplace_back(-0.24353, -0.74052, 0.62636);
    samplingPattern.emplace_back(-0.62636, 0.24353, -0.74052);
    samplingPattern.emplace_back(-0.74052, 0.62636, -0.24353);
    samplingPattern.emplace_back(0.24353, -0.74052, -0.62636);
    samplingPattern.emplace_back(-0.74052, -0.62636, 0.24353);
    samplingPattern.emplace_back(0.24353, 0.74052, 0.62636);
    samplingPattern.emplace_back(0.74052, -0.62636, -0.24353);
    samplingPattern.emplace_back(-0.24353, 0.74052, -0.62636);
    samplingPattern.emplace_back(0.74052, 0.62636, 0.24353);
    samplingPattern.emplace_back(0.62636, 0.24353, 0.74052);
    samplingPattern.emplace_back(-0.62636, -0.24353, 0.74052);
    samplingPattern.emplace_back(-0.28625, 0.95712, -0.044524);
    samplingPattern.emplace_back(0.95712, -0.044524, -0.28625);
    samplingPattern.emplace_back(0.28625, -0.95712, -0.044524);
    samplingPattern.emplace_back(-0.044524, -0.28625, 0.95712);
    samplingPattern.emplace_back(-0.95712, -0.044524, 0.28625);
    samplingPattern.emplace_back(-0.044524, 0.28625, -0.95712);
    samplingPattern.emplace_back(-0.95712, 0.044524, -0.28625);
    samplingPattern.emplace_back(0.044524, 0.28625, 0.95712);
    samplingPattern.emplace_back(0.95712, 0.044524, 0.28625);
    samplingPattern.emplace_back(0.044524, -0.28625, -0.95712);
    samplingPattern.emplace_back(-0.28625, -0.95712, 0.044524);
    samplingPattern.emplace_back(0.28625, 0.95712, 0.044524);

    typename AXDTOperator<data_t>::WeightVec weights =
        static_cast<real_t>(4.0 * pi<data_t>) / static_cast<real_t>(samplingPattern.size())
        * AXDTOperator<data_t>::WeightVec::Ones(static_cast<index_t>(samplingPattern.size()));

    SphericalFunctionInformation<data_t> sf_info{samplingPattern, weights,
                                                 AXDTOperator<data_t>::Symmetry::regular, 4};

    GIVEN("sf measurements and ground truth transorm results")
    {
        Vector_t<double> input(36);
        input << 0.754358291407775, -0.306820645184338, -0.123204435921810, 0.849556048213156,
            -1.274938108853798, -0.573111100611586, 0.627576237773633, 0.177121393717249,
            0.104396351299045, 0.142036209457050, 2.231368511612128, -0.015059877844922,
            0.805849500639438, -1.085409856384089, 0.325126697713267, 0.238064050790899,
            -1.072604254662862, -0.519390459070897, 0.020086630029301, -1.463706932543725,
            -0.424752454749628, 0.120423823957031, 0.930582112066446, 0.338658848788792,
            0.147279798616856, 0.756725110766516, 0.881930324545221, -0.087410216765013,
            1.031659345636263, 0.864097492602849, 0.401666595949420, -0.112340125336992,
            0.129707228273439, -0.081263651974596, 0.251449774448726, -0.909181928284521;

        // matlab forward transform result
        Vector_t<double> matlab_result(25);
        matlab_result << 0.401808033751943, 0.075966691690841, 0.239916153553659, 0.123318934835165,
            0.183907788282417, 0.239952525664901, 0.417267069084371, 0.049654430325743,
            0.902716109915281, 0.944787189721647, 0.490864092468080, 0.489252638400017,
            0.337719409821378, 0.900053846417662, 0.369246781120215, 0.111202755293787,
            0.780252068321136, 0.389738836961254, 0.241691285913831, 0.403912145588112,
            0.096454525168390, 0.131973292606336, 0.942050590775488, 0.956134540229803,
            0.575208595078468;

        auto spfFieldDesc = std::make_unique<IdenticalBlocksDescriptor>(36, volDesc);
        auto spfWeights = std::make_unique<DataContainer<data_t>>(*spfFieldDesc);
        const auto& weightVolDesc = spfFieldDesc->getDescriptorOfBlock(0);

        auto sphWeightsDesc =
            std::make_unique<IdenticalBlocksDescriptor>(sf_info.basisCnt, weightVolDesc);
        auto sphWeights = std::make_unique<DataContainer<data_t>>(*sphWeightsDesc);

        SphericalFieldsTransform<data_t> sft(sf_info);

        index_t voxelCount = weightVolDesc.getNumberOfCoefficients();
        auto samplingDirs = static_cast<index_t>(sf_info.dirs.size());
        index_t sphCoeffsCount = sf_info.basisCnt;

        WHEN("apply arbitrary weights and perform the transform")
        {
            for (index_t z = 0; z < 36; ++z) {
                for (index_t y = 0; y < 2; ++y) {
                    for (index_t x = 0; x < 2; ++x) {
                        IndexVector_t idx(3);
                        idx << x, y, z;
                        (*spfWeights)(idx) =
                            static_cast<data_t>(2 * y + x + 1) * static_cast<data_t>(input[z]);
                    }
                }
            }

            // transpose of mode-4 unfolding of x
            Eigen::Map<const typename SphericalFieldsTransform<data_t>::MatrixXd_t> x4(
                &((spfWeights->storage())[0]), voxelCount, samplingDirs);

            // transpose of mode-4 unfolding of Ax
            Eigen::Map<typename SphericalFieldsTransform<data_t>::MatrixXd_t> Ax4(
                &((sphWeights->storage())[0]), voxelCount, sphCoeffsCount);

            typename SphericalFieldsTransform<data_t>::MatrixXd_t WVt =
                sft.getForwardTransformationMatrix().transpose();

            Ax4 = x4 * WVt;
            THEN("results should be close to ground truth")
            {
                for (index_t z = 0; z < 25; ++z) {
                    for (index_t y = 0; y < 2; ++y) {
                        for (index_t x = 0; x < 2; ++x) {
                            IndexVector_t idx(3);
                            idx << x, y, z;
                            REQUIRE((*sphWeights)(idx)
                                    == Approx(static_cast<data_t>(2 * y + x + 1) * matlab_result[z])
                                           .epsilon(1e-3));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("AXDTOperator: Testing apply & applyAdjoint", data_t, float, double)
{
    const index_t domainSize{2};
    IndexVector_t domain{3};
    domain << domainSize, domainSize, domainSize;
    VolumeDescriptor volDesc{domain};

    const index_t rangeSize{domainSize};
    const index_t numOfImgs{domainSize};
    IndexVector_t range{3};
    range << rangeSize, rangeSize, numOfImgs;
    RealVector_t rangeSpacing{3};
    rangeSpacing << 1, 1, 1;

    geometry::SourceToCenterOfRotation s2c{100};
    geometry::CenterOfRotationToDetector c2d{50};

    // arbitrary geometry
    std::vector<Geometry> geos;
    for (index_t i = 0; i < numOfImgs; ++i) {
        geos.emplace_back(s2c, c2d, geometry::VolumeData3D{geometry::Size3D{domain}},
                          geometry::SinogramData3D{geometry::Size3D{range}},
                          geometry::Gamma{static_cast<real_t>(i)});
    }

    typename AXDTOperator<data_t>::DirVecList samplingPattern;
    samplingPattern.emplace_back(0.50748, -0.3062, 0.80543);
    samplingPattern.emplace_back(-0.3062, 0.80543, 0.50748);
    samplingPattern.emplace_back(-0.50748, 0.3062, 0.80543);
    samplingPattern.emplace_back(0.80543, 0.50748, -0.3062);
    samplingPattern.emplace_back(0.3062, 0.80543, -0.50748);
    samplingPattern.emplace_back(0.80543, -0.50748, 0.3062);
    samplingPattern.emplace_back(0.3062, -0.80543, 0.50748);
    samplingPattern.emplace_back(-0.80543, -0.50748, -0.3062);
    samplingPattern.emplace_back(-0.3062, -0.80543, -0.50748);
    samplingPattern.emplace_back(-0.80543, 0.50748, 0.3062);
    samplingPattern.emplace_back(0.50748, 0.3062, -0.80543);
    samplingPattern.emplace_back(-0.50748, -0.3062, -0.80543);
    samplingPattern.emplace_back(0.62636, -0.24353, -0.74052);
    samplingPattern.emplace_back(-0.24353, -0.74052, 0.62636);
    samplingPattern.emplace_back(-0.62636, 0.24353, -0.74052);
    samplingPattern.emplace_back(-0.74052, 0.62636, -0.24353);
    samplingPattern.emplace_back(0.24353, -0.74052, -0.62636);
    samplingPattern.emplace_back(-0.74052, -0.62636, 0.24353);
    samplingPattern.emplace_back(0.24353, 0.74052, 0.62636);
    samplingPattern.emplace_back(0.74052, -0.62636, -0.24353);
    samplingPattern.emplace_back(-0.24353, 0.74052, -0.62636);
    samplingPattern.emplace_back(0.74052, 0.62636, 0.24353);
    samplingPattern.emplace_back(0.62636, 0.24353, 0.74052);
    samplingPattern.emplace_back(-0.62636, -0.24353, 0.74052);
    samplingPattern.emplace_back(-0.28625, 0.95712, -0.044524);
    samplingPattern.emplace_back(0.95712, -0.044524, -0.28625);
    samplingPattern.emplace_back(0.28625, -0.95712, -0.044524);
    samplingPattern.emplace_back(-0.044524, -0.28625, 0.95712);
    samplingPattern.emplace_back(-0.95712, -0.044524, 0.28625);
    samplingPattern.emplace_back(-0.044524, 0.28625, -0.95712);
    samplingPattern.emplace_back(-0.95712, 0.044524, -0.28625);
    samplingPattern.emplace_back(0.044524, 0.28625, 0.95712);
    samplingPattern.emplace_back(0.95712, 0.044524, 0.28625);
    samplingPattern.emplace_back(0.044524, -0.28625, -0.95712);
    samplingPattern.emplace_back(-0.28625, -0.95712, 0.044524);
    samplingPattern.emplace_back(0.28625, 0.95712, 0.044524);

    typename AXDTOperator<data_t>::WeightVec weights =
        static_cast<real_t>(4.0 * pi<data_t>) / static_cast<real_t>(samplingPattern.size())
        * AXDTOperator<data_t>::WeightVec::Ones(static_cast<index_t>(samplingPattern.size()));

    GIVEN("an AXDTOperator")
    {
        XGIDetectorDescriptor xgiDesc{range, rangeSpacing, geos,
                                      XGIDetectorDescriptor::DirVec(1, 0, 0), true};

        Identity<data_t> proj(volDesc); // pretend to be a legit projector... (restricted
                                        // domain/range dimension sizes)

        AXDTOperator<data_t> axdtOp{
            volDesc, xgiDesc, proj, samplingPattern, weights, AXDTOperator<data_t>::Symmetry::even,
            4};

        WHEN("apply the transform")
        {
            DataContainer<data_t> input(
                IdenticalBlocksDescriptor(15, volDesc),
                Eigen::Matrix<data_t, 15 * domainSize * domainSize * domainSize, 1>::Ones());
            DataContainer<data_t> output(xgiDesc);
            axdtOp.apply(input, output);

            THEN("results should be close to ground truth")
            {
                Vector_t<double> groundTruth(rangeSize * rangeSize * numOfImgs);
                groundTruth << 1.10996, 1.10996, 1.10996, 1.10996, 0.597108, 0.597108, 0.597108,
                    0.597108;

                for (index_t i = 0; i < xgiDesc.getNumberOfCoefficients(); ++i) {
                    REQUIRE(output[i] == Approx(groundTruth[i]).epsilon(1e-3));
                }
            }
        }

        WHEN("apply the adjoint transform")
        {
            DataContainer<data_t> input(IdenticalBlocksDescriptor(15, volDesc));
            DataContainer<data_t> output(
                xgiDesc, Eigen::Matrix<data_t, rangeSize * rangeSize * numOfImgs, 1>::Ones());
            axdtOp.applyAdjoint(output, input);

            THEN("results should be close to ground truth")
            {
                Vector_t<double> groundTruth(domainSize * domainSize * domainSize * 15);
                groundTruth << 0.945309, 0.945309, 0.945309, 0.945309, 0.945309, 0.945309, 0.945309,
                    0.945309, 9.31323e-10, 9.31323e-10, 9.31323e-10, 9.31323e-10, 7.45058e-09,
                    7.45058e-09, 7.45058e-09, 7.45058e-09, -3.25963e-08, -3.25963e-08, -3.25963e-08,
                    -3.25963e-08, 0, 0, 0, 0, -0.603937, -0.603937, -0.603937, -0.603937, 0.518595,
                    0.518595, 0.518595, 0.518595, -7.45058e-09, -7.45058e-09, -7.45058e-09,
                    -7.45058e-09, -0.832273, -0.832273, -0.832273, -0.832273, 0.784536, 0.784536,
                    0.784536, 0.784536, 0.136442, 0.136442, 0.136442, 0.136442, -1.86265e-08,
                    -1.86265e-08, -1.86265e-08, -1.86265e-08, 1.11759e-08, 1.11759e-08, 1.11759e-08,
                    1.11759e-08, 6.33299e-08, 6.33299e-08, 6.33299e-08, 6.33299e-08, 7.45058e-09,
                    7.45058e-09, 7.45058e-09, 7.45058e-09, -2.6077e-08, -2.6077e-08, -2.6077e-08,
                    -2.6077e-08, -4.47035e-08, -4.47035e-08, -4.47035e-08, -4.47035e-08,
                    4.47035e-08, 4.47035e-08, 4.47035e-08, 4.47035e-08, 4.47035e-08, 4.47035e-08,
                    4.47035e-08, 4.47035e-08, 0.135046, 0.135046, 0.135046, 0.135046, -0.109205,
                    -0.109205, -0.109205, -0.109205, 1.49012e-08, 1.49012e-08, 1.49012e-08,
                    1.49012e-08, -0.141402, -0.141402, -0.141402, -0.141402, -0.150987, -0.150987,
                    -0.150987, -0.150987, 0.0674836, 0.0674836, 0.0674836, 0.0674836, 1.49012e-08,
                    1.49012e-08, 1.49012e-08, 1.49012e-08, 0.0534442, 0.0534442, 0.0534442,
                    0.0534442, -4.67896e-06, -4.67896e-06, -4.67896e-06, -4.67896e-06, -0.0412863,
                    -0.0412863, -0.0412863, -0.0412863;

                for (index_t i = 0; i < input.getDataDescriptor().getNumberOfCoefficients(); ++i) {
                    REQUIRE(input[i] == Approx(groundTruth[i]).epsilon(1e-3));
                }
            }
        }
    }
}

TEST_SUITE_END();