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
#include "Scaling.h"

using namespace elsa;
using namespace doctest;
using namespace elsa::axdt;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("AXDTOperator: Testing SphericalFieldsTransform", data_t, float, double)
{
    const size_t volSize{2};
    IndexVector_t vol(2);
    vol << volSize, volSize;
    VolumeDescriptor volDesc{vol};

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

    Vector_t<data_t> weights =
        static_cast<real_t>(4.0 * pi<data_t>) / static_cast<real_t>(samplingPattern.size())
        * Vector_t<data_t>::Ones(static_cast<index_t>(samplingPattern.size()));

    SphericalFunctionInformation<data_t> sf_info{samplingPattern, weights,
                                                 AXDTOperator<data_t>::Symmetry::regular, 4};

    auto samplingDirCnt{samplingPattern.size()};
    auto sphCoeffsCnt{sf_info.basisCnt};

    GIVEN("sf measurements and ground truth transorm results")
    {
        Vector_t<double> input(samplingDirCnt);
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
        Vector_t<double> matlab_result(sphCoeffsCnt);
        matlab_result << 0.401808033751943, 0.075966691690841, 0.239916153553659, 0.123318934835165,
            0.183907788282417, 0.239952525664901, 0.417267069084371, 0.049654430325743,
            0.902716109915281, 0.944787189721647, 0.490864092468080, 0.489252638400017,
            0.337719409821378, 0.900053846417662, 0.369246781120215, 0.111202755293787,
            0.780252068321136, 0.389738836961254, 0.241691285913831, 0.403912145588112,
            0.096454525168390, 0.131973292606336, 0.942050590775488, 0.956134540229803,
            0.575208595078468;

        auto spfFieldDesc = std::make_unique<IdenticalBlocksDescriptor>(samplingDirCnt, volDesc);
        auto spfWeights = std::make_unique<DataContainer<data_t>>(*spfFieldDesc);
        const auto& weightVolDesc = spfFieldDesc->getDescriptorOfBlock(0);

        auto sphWeightsDesc =
            std::make_unique<IdenticalBlocksDescriptor>(sf_info.basisCnt, weightVolDesc);
        auto sphWeights = std::make_unique<DataContainer<data_t>>(*sphWeightsDesc);

        SphericalFieldsTransform<data_t> sft(sf_info);

        index_t volCnt = weightVolDesc.getNumberOfCoefficients();

        WHEN("apply arbitrary weights and perform the transform")
        {
            for (index_t z = 0; z < static_cast<index_t>(samplingDirCnt); ++z) {
                for (index_t y = 0; y < static_cast<index_t>(volSize); ++y) {
                    for (index_t x = 0; x < static_cast<index_t>(volSize); ++x) {
                        IndexVector_t idx(3);
                        idx << x, y, z;
                        (*spfWeights)(idx) =
                            static_cast<data_t>(2 * y + x + 1) * static_cast<data_t>(input[z]);
                    }
                }
            }

            Eigen::Map<const typename SphericalFieldsTransform<data_t>::MatrixXd_t> x(
                &((spfWeights->storage())[0]), volCnt, static_cast<index_t>(samplingDirCnt));

            Eigen::Map<typename SphericalFieldsTransform<data_t>::MatrixXd_t> Ax(
                &((sphWeights->storage())[0]), volCnt, sphCoeffsCnt);

            typename SphericalFieldsTransform<data_t>::MatrixXd_t WVt =
                sft.getForwardTransformationMatrix().transpose();

            Ax = x * WVt;
            THEN("results should be close to ground truth")
            {
                for (index_t z = 0; z < static_cast<index_t>(sphCoeffsCnt); ++z) {
                    for (index_t y = 0; y < static_cast<index_t>(volSize); ++y) {
                        for (index_t x = 0; x < static_cast<index_t>(volSize); ++x) {
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

    Vector_t<data_t> weights =
        static_cast<real_t>(4.0 * pi<data_t>) / static_cast<real_t>(samplingPattern.size())
        * Vector_t<data_t>::Ones(static_cast<index_t>(samplingPattern.size()));

    GIVEN("an AXDTOperator with an identity operator as the projector")
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
    GIVEN("an AXDTOperator with an anisotropic scaling operator as the projector")
    {
        XGIDetectorDescriptor xgiDesc{range, rangeSpacing, geos,
                                      XGIDetectorDescriptor::DirVec(1, 0, 0), true};

        Eigen::Matrix<data_t, domainSize * domainSize * domainSize, 1> rawScaleValues;
        rawScaleValues << 2, 3, 5, 7, 11, 13, 17, 19;
        DataContainer<data_t> scaleValues(volDesc, rawScaleValues);
        Scaling proj(volDesc, scaleValues); // pretend to be a legit projector... (restricted
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
                groundTruth << 2.21993, 3.32989, 5.54981, 7.76974, 6.56819, 7.7624, 10.1508, 11.345;

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
                groundTruth << 1.89062, 2.83593, 4.72654, 6.61716, 10.3984, 12.289, 16.0702,
                    17.9609, 1.86265e-09, 2.79397e-09, 4.65661e-09, 6.51926e-09, 8.19564e-08,
                    9.68575e-08, 1.2666e-07, 1.41561e-07, -6.51926e-08, -9.77889e-08, -1.62981e-07,
                    -2.28174e-07, 0, 0, 0, 0, -1.20787, -1.81181, -3.01968, -4.22756, 5.70455,
                    6.74174, 8.81612, 9.85331, -1.49012e-08, -2.23517e-08, -3.72529e-08,
                    -5.21541e-08, -9.155, -10.8195, -14.1486, -15.8132, 1.56907, 2.35361, 3.92268,
                    5.49175, 1.50086, 1.77374, 2.31951, 2.59239, -3.72529e-08, -5.58794e-08,
                    -9.31323e-08, -1.30385e-07, 1.22935e-07, 1.45286e-07, 1.8999e-07, 2.12342e-07,
                    1.2666e-07, 1.8999e-07, 3.1665e-07, 4.4331e-07, 8.19564e-08, 9.68575e-08,
                    1.2666e-07, 1.41561e-07, -5.21541e-08, -7.82311e-08, -1.30385e-07, -1.82539e-07,
                    -4.91738e-07, -5.81145e-07, -7.59959e-07, -8.49366e-07, 8.9407e-08, 1.3411e-07,
                    2.23517e-07, 3.12924e-07, 4.91738e-07, 5.81145e-07, 7.59959e-07, 8.49366e-07,
                    0.270092, 0.405139, 0.675231, 0.945323, -1.20126, -1.41967, -1.85649, -2.0749,
                    2.98023e-08, 4.47035e-08, 7.45058e-08, 1.04308e-07, -1.55542, -1.83822,
                    -2.40383, -2.68663, -0.301974, -0.452961, -0.754935, -1.05691, 0.74232,
                    0.877287, 1.14722, 1.28219, 2.98023e-08, 4.47035e-08, 7.45058e-08, 1.04308e-07,
                    0.587887, 0.694775, 0.908552, 1.01544, -9.35793e-06, -1.40369e-05, -2.33948e-05,
                    -3.27528e-05, -0.454149, -0.536722, -0.701867, -0.784439;

                for (index_t i = 0; i < input.getDataDescriptor().getNumberOfCoefficients(); ++i) {
                    REQUIRE(input[i] == Approx(groundTruth[i]).epsilon(1e-3));
                }
            }
        }
    }
}

TEST_SUITE_END();