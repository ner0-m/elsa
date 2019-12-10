#include <catch2/catch.hpp>

#include "SiddonsMethodCUDA.h"
#include "SiddonsMethod.h"
#include "Geometry.h"

using namespace elsa;

/*
 * checks whether two DataContainer<data_t>s contain approximately the same data using the same
 * method as Eigen
 * https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae8443357b808cd393be1b51974213f9c
 *
 * precision depends on the global elsa::real_t parameter, as the majority of the error is produced
 * by the traversal algorithm (which is executed with real_t precision regardless of the
 * DataContainer type)
 *
 */
template <typename data_t>
bool isApprox(const DataContainer<data_t>& x, const DataContainer<data_t>& y,
              real_t prec = Eigen::NumTraits<real_t>::dummy_precision())
{
    DataContainer<data_t> z = x;
    z -= y;
    return sqrt(z.squaredL2Norm()) <= prec * sqrt(std::min(x.squaredL2Norm(), y.squaredL2Norm()));
}

/*
 * this function declaration can be used in conjunction with decltype to deduce the
 * template parameter of a templated class at compile time
 *
 * the template parameter must be a typename
 */
template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEMPLATE_TEST_CASE("Scenario: Calls to functions of super class", "", SiddonsMethodCUDA<float>,
                   SiddonsMethodCUDA<double>, SiddonsMethod<float>, SiddonsMethod<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    GIVEN("A projector")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 50;
        const index_t detectorSize = 50;
        const index_t numImgs = 50;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        volume = 1;
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;
        for (std::size_t i = 0; i < numImgs; i++) {
            real_t angle = i * 2 * pi_t / 50;
            geom.emplace_back(20 * volSize, volSize, angle, volumeDescriptor, sinoDescriptor);
        }
        TestType op(volumeDescriptor, sinoDescriptor, geom);

        WHEN("Projector is cloned")
        {
            auto opClone = op.clone();
            auto sinoClone = DataContainer<data_t>(sinoDescriptor);
            auto volumeClone = DataContainer<data_t>(volumeDescriptor);

            THEN("Results do not change (may still be slightly different due to summation being "
                 "performed in a different order)")
            {
                op.apply(volume, sino);
                opClone->apply(volume, sinoClone);
                REQUIRE(isApprox(sino, sinoClone));

                op.applyAdjoint(sino, volume);
                opClone->applyAdjoint(sino, volumeClone);
                REQUIRE(isApprox(volume, volumeClone));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Output DataContainer<data_t> is not zero initialized", "",
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    GIVEN("A 2D setting")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;
        geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor);
        TestType op(volumeDescriptor, sinoDescriptor, geom);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));
            }
        }

        WHEN("Volume container is not zero initialized and we backproject from an empty sinogram")
        {
            sino = 0;
            volume = 1;

            THEN("Result is zero")
            {
                op.applyAdjoint(sino, volume);
                REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
            }
        }
    }

    GIVEN("A 3D setting")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, 0);
        TestType op(volumeDescriptor, sinoDescriptor, geom);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));
            }
        }

        WHEN("Volume container is not zero initialized and we backproject from an empty sinogram")
        {
            sino = 0;
            volume = 1;

            THEN("Result is zero")
            {
                op.applyAdjoint(sino, volume);
                REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Rays not intersecting the bounding box are present", "",
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    GIVEN("A 2D setting")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        volume = 1;
        sino = 1;
        std::vector<Geometry> geom;

        WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin")
        {
            geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              volSize);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
                }
            }
        }

        WHEN("Tracing along a y-axis-aligned ray with a x-coordinate of origin beyond the bounding "
             "box")
        {
            geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              -volSize);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a negative y-coordinate of origin")
        {
            geom.emplace_back(20 * volSize, volSize, pi_t / 2, volumeDescriptor, sinoDescriptor,
                              0.0, 0.0, volSize);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a y-coordinate of origin beyond the bounding "
             "box")
        {
            geom.emplace_back(20 * volSize, volSize, pi_t / 2, volumeDescriptor, sinoDescriptor,
                              0.0, 0.0, -volSize);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
                }
            }
        }
    }

    GIVEN("A 3D setting")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        volume = 1;
        sino = 1;
        std::vector<Geometry> geom;

        const index_t numCases = 9;
        real_t alpha[numCases] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        real_t beta[numCases] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pi_t / 2, pi_t / 2, pi_t / 2};
        real_t gamma[numCases] = {0.0,      0.0,      0.0,      pi_t / 2, pi_t / 2,
                                  pi_t / 2, pi_t / 2, pi_t / 2, pi_t / 2};
        real_t offsetx[numCases] = {volSize, 0.0, volSize, 0.0, 0.0, 0.0, volSize, 0.0, volSize};
        real_t offsety[numCases] = {0.0, volSize, volSize, volSize, 0.0, volSize, 0.0, 0.0, 0.0};
        real_t offsetz[numCases] = {0.0, 0.0, 0.0, 0.0, volSize, volSize, 0.0, volSize, volSize};
        std::string neg[numCases] = {"x", "y", "x and y", "y", "z", "y and z", "x", "z", "x and z"};
        std::string ali[numCases] = {"z", "z", "z", "x", "x", "x", "y", "y", "y"};

        for (int i = 0; i < numCases; i++) {
            WHEN("Tracing along a " + ali[i] + "-axis-aligned ray with negative " + neg[i]
                 + "-coodinate of origin")
            {
                geom.emplace_back(20 * volSize, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i], alpha[i], 0.0, 0.0, offsetx[i], offsety[i], offsetz[i]);

                TestType op(volumeDescriptor, sinoDescriptor, geom);

                THEN("Result of forward projection is zero")
                {
                    op.apply(volume, sino);
                    REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                    AND_THEN("Result of backprojection is zero")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Axis-aligned rays are present", "", SiddonsMethodCUDA<float>,
                   SiddonsMethodCUDA<double>, SiddonsMethod<float>, SiddonsMethod<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    GIVEN("A 2D setting with a single ray")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        const index_t numCases = 4;
        const real_t angles[numCases] = {0.0, pi_t / 2, pi_t, 3 * pi_t / 2};
        Eigen::Matrix<data_t, volSize * volSize, 1> backProj[2];
        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray with an angle of " + std::to_string(angles[i])
                 + " radians passes through the center of a pixel")
            {
                geom.emplace_back(volSize * 20, volSize, angles[i], volumeDescriptor,
                                  sinoDescriptor);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
                THEN("The result of projecting through a pixel is exactly the pixel value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i % 2 == 0)
                            volume(volSize / 2, j) = 1;
                        else
                            volume(j, volSize / 2) = 1;

                        op.apply(volume, sino);
                        REQUIRE(sino[0] == 1);
                    }

                    AND_THEN("The backprojection sets the values of all hit pixels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume,
                                         DataContainer<data_t>(volumeDescriptor, backProj[i % 2])));
                    }
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along the left voxel boundary")
        {
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              -0.5);
            TestType op(volumeDescriptor, sinoDescriptor, geom);
            THEN("The result of projecting through a pixel is the value of the pixel with the "
                 "higher index")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(volSize / 2, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1.0));
                }

                AND_THEN("The backprojection yields the exact adjoint")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj[0])));
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along the right volume boundary")
        {
            // For Siddons's values in the range [0,boxMax) are considered, a ray running along
            // boxMax should be ignored
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              (volSize * 0.5));
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The result of projecting is zero")
            {
                volume = 0;
                op.apply(volume, sino);
                REQUIRE(sino[0] == 0.0);

                AND_THEN("The result of backprojection is also zero")
                {
                    sino[0] = 1;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
                }
            }
        }

        backProj[0] << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0;

        WHEN("A y-axis-aligned ray runs along the left volume boundary")
        {
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              -volSize / 2.0);
            TestType op(volumeDescriptor, sinoDescriptor, geom);
            THEN("The result of projecting through a pixel is exactly the pixel's value")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(0, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == 1);
                }

                AND_THEN("The backprojection yields the exact adjoint")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj[0])));
                }
            }
        }
    }

    GIVEN("A 3D setting with a single ray")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        const index_t numCases = 6;
        real_t beta[numCases] = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
        real_t gamma[numCases] = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};
        std::string al[numCases] = {"z", "-z", "x", "-x", "y", "-y"};

        Eigen::Matrix<data_t, volSize * volSize * volSize, 1> backProj[numCases];

        backProj[2] << 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 1, 0, 0, 1, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 1, 1, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0,

            0, 0, 0, 0, 1, 0, 0, 0, 0,

            0, 0, 0, 0, 1, 0, 0, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("A " + al[i] + "-axis-aligned ray passes through the center of a pixel")
            {
                geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i]);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
                THEN("The result of projecting through a voxel is exactly the voxel value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i / 2 == 0)
                            volume(volSize / 2, volSize / 2, j) = 1;
                        else if (i / 2 == 1)
                            volume(j, volSize / 2, volSize / 2) = 1;
                        else if (i / 2 == 2)
                            volume(volSize / 2, j, volSize / 2) = 1;

                        op.apply(volume, sino);
                        REQUIRE(sino[0] == 1);
                    }

                    AND_THEN("The backprojection sets the values of all hit voxels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume,
                                         DataContainer<data_t>(volumeDescriptor, backProj[i / 2])));
                    }
                }
            }
        }

        real_t offsetx[numCases];
        real_t offsety[numCases];

        offsetx[0] = volSize / 2.0;
        offsetx[3] = -(volSize / 2.0);
        offsetx[1] = 0.0;
        offsetx[4] = 0.0;
        offsetx[5] = -(volSize / 2.0);
        offsetx[2] = (volSize / 2.0);

        offsety[0] = 0.0;
        offsety[3] = 0.0;
        offsety[1] = volSize / 2.0;
        offsety[4] = -(volSize / 2.0);
        offsety[5] = -(volSize / 2.0);
        offsety[2] = (volSize / 2.0);

        backProj[0] << 0, 0, 0, 1, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 0, 0, 0, 0, 0;

        backProj[1] << 0, 1, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 0, 0, 0, 0, 0;

        backProj[2] << 1, 0, 0, 0, 0, 0, 0, 0, 0,

            1, 0, 0, 0, 0, 0, 0, 0, 0,

            1, 0, 0, 0, 0, 0, 0, 0, 0;

        al[0] = "left border";
        al[1] = "bottom border";
        al[2] = "bottom left border";
        al[3] = "right border";
        al[4] = "top border";
        al[5] = "top right edge";

        for (index_t i = 0; i < numCases / 2; i++) {
            WHEN("A z-axis-aligned ray runs along the " + al[i] + " of the volume")
            {
                // x-ray source must be very far from the volume center to make testing of the op
                // backprojection simpler
                geom.emplace_back(volSize * 2000, volSize, volumeDescriptor, sinoDescriptor, 0.0,
                                  0.0, 0.0, 0.0, 0.0, -offsetx[i], -offsety[i]);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
                THEN("The result of projecting through a voxel is exactly the voxel's value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        switch (i) {
                            case 0:
                                volume(0, volSize / 2, j) = 1;
                                break;
                            case 1:
                                volume(volSize / 2, 0, j) = 1;
                                break;
                            case 2:
                                volume(0, 0, j) = 1;
                                break;
                            default:
                                break;
                        }

                        op.apply(volume, sino);
                        REQUIRE(sino[0] == 1);
                    }

                    AND_THEN("The backprojection yields the exact adjoints")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);

                        REQUIRE(
                            isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj[i])));
                    }
                }
            }
        }

        for (index_t i = numCases / 2; i < numCases; i++) {
            WHEN("A z-axis-aligned ray runs along the " + al[i] + " of the volume")
            {
                // x-ray source must be very far from the volume center to make testing of the op
                // backprojection simpler
                geom.emplace_back(volSize * 2000, volSize, volumeDescriptor, sinoDescriptor, 0.0,
                                  0.0, 0.0, 0.0, 0.0, -offsetx[i], -offsety[i]);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
                THEN("The result of projecting is zero")
                {
                    volume = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == 0);

                    AND_THEN("The result of backprojection is also zero")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);

                        REQUIRE(volume == DataContainer<data_t>(volumeDescriptor));
                    }
                }
            }
        }
    }

    GIVEN("A 2D setting with multiple projection angles")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 4;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("Both x- and y-axis-aligned rays are present")
        {
            geom.emplace_back(20 * volSize, volSize, 0, volumeDescriptor, sinoDescriptor);
            geom.emplace_back(20 * volSize, volSize, 90 * pi_t / 180., volumeDescriptor,
                              sinoDescriptor);
            geom.emplace_back(20 * volSize, volSize, 180 * pi_t / 180., volumeDescriptor,
                              sinoDescriptor);
            geom.emplace_back(20 * volSize, volSize, 270 * pi_t / 180., volumeDescriptor,
                              sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is dones
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    volume(i, volSize / 2) = 1;
                    volume(volSize / 2, i) = 1;
                }

                op.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE(sino[i] == Approx(5.0));

                AND_THEN("Backprojection yields the exact adjoint")
                {
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize);

                    cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0,
                        10, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
                }
            }
        }
    }

    GIVEN("A 3D setting with multiple projection angles")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 6;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("x-, y and z-axis-aligned rays are present")
        {
            real_t beta[numImgs] = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
            real_t gamma[numImgs] = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};

            for (index_t i = 0; i < numImgs; i++)
                geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i]);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is dones
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    volume(i, volSize / 2, volSize / 2) = 1;
                    volume(volSize / 2, i, volSize / 2) = 1;
                    volume(volSize / 2, volSize / 2, i) = 1;
                }

                op.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE(sino[i] == Approx(3.0));

                AND_THEN("Backprojection yields the exact adjoint")
                {
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize * volSize);

                    cmp << 0, 0, 0, 0, 6, 0, 0, 0, 0,

                        0, 6, 0, 6, 18, 6, 0, 6, 0,

                        0, 0, 0, 0, 6, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Projection under an angle", "", SiddonsMethodCUDA<float>,
                   SiddonsMethodCUDA<double>, SiddonsMethod<float>, SiddonsMethod<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    GIVEN("A 2D setting with a single ray")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 4;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("Projecting under an angle of 30 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;
                volume(2, 0) = 0;
                volume(2, 1) = 0;

                volume(1, 2) = 0;
                volume(1, 3) = 0;
                volume(0, 3) = 0;
                volume(2, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;
                    volume(2, 0) = 2;
                    volume(2, 1) = 3;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt(3) + 2));

                    // on the other side of the center
                    volume = 0;
                    volume(1, 2) = 3;
                    volume(1, 3) = 2;
                    volume(0, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt(3) + 2));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 0, 0, 2 - 2 / sqrt(3), 4 / sqrt(3) - 2, 0, 0, 2 / sqrt(3), 0, 0,
                        2 / sqrt(3), 0, 0, 4 / sqrt(3) - 2, 2 - 2 / sqrt(3), 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     0.0001));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray enters through the right border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, sqrt(3));
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 1) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;
                volume(2, 3) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 1) = 4;
                    volume(3, 2) = 3;
                    volume(3, 3) = 2;
                    volume(2, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt(3)));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 0, 0, 0, 0, 0, 0, 0, 4 - 2 * sqrt(3), 0, 0, 0, 2 / sqrt(3), 0, 0,
                        2 - 2 / sqrt(3), 4 / sqrt(3) - 2;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray exits through the left border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, -sqrt(3));
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(1, 0) = 0;
                volume(0, 1) = 0;
                volume(0, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 1;
                    volume(0, 0) = 2;
                    volume(0, 1) = 3;
                    volume(0, 2) = 4;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt(3)));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 4 / sqrt(3) - 2, 2 - 2 / sqrt(3), 0, 0, 2 / sqrt(3), 0, 0, 0,
                        4 - 2 * sqrt(3), 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray only intersects a single pixel")
        {
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, -2 - sqrt(3) / 2);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 / sqrt(3)));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 1 / sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(0, 1) = 0;
                volume(1, 1) = 0;
                volume(2, 2) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;
                // volume(1,2) hit due to numerical error
                volume(1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;
                    volume(0, 1) = 2;
                    volume(1, 1) = 3;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt(3) + 2));

                    // on the other side of the center
                    volume = 0;
                    volume(2, 2) = 3;
                    volume(3, 2) = 2;
                    volume(3, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt(3) + 2));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                    expected << 4 / sqrt(3) - 2, 0, 0, 0, 2 - 2 / sqrt(3), 2 / sqrt(3), 0, 0, 0, 0,
                        2 / sqrt(3), 2 - 2 / sqrt(3), 0, 0, 0, 4 / sqrt(3) - 2;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray enters through the top border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, sqrt(3));
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 2) = 0;
                volume(0, 3) = 0;
                volume(1, 3) = 0;
                volume(2, 3) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 2) = 1;
                    volume(0, 3) = 2;
                    volume(1, 3) = 3;
                    volume(2, 3) = 4;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt(3)));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                    expected << 0, 0, 0, 0, 0, 0, 0, 0, 2 - 2 / sqrt(3), 0, 0, 0, 4 / sqrt(3) - 2,
                        2 / sqrt(3), 4 - 2 * sqrt(3), 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray exits through the bottom border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, -sqrt(3));
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(1, 0) = 0;
                volume(2, 0) = 0;
                volume(3, 0) = 0;
                volume(3, 1) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 4;
                    volume(2, 0) = 3;
                    volume(3, 0) = 2;
                    volume(3, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt(3)));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                    expected << 0, 4 - 2 * sqrt(3), 2 / sqrt(3), 4 / sqrt(3) - 2, 0, 0, 0,
                        2 - 2 / sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray only intersects a single pixel")
        {
            // This is a special case that is handled separately in both forward and backprojection
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, -2 - sqrt(3) / 2);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;

                op.apply(volume, sino);
                REQUIRE(sino == DataContainer<data_t>(sinoDescriptor));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 / sqrt(3)).epsilon(0.005));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 0, 0, 0, 1 / sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected)));
                }
            }
        }
    }

    GIVEN("A 3D setting with a single ray")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        Eigen::Matrix<data_t, volSize * volSize * volSize, 1> backProj;

        WHEN("A ray with an angle of 30 degrees goes through the center of the volume")
        {
            // In this case the ray enters and exits the volume along the main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(1, 1, 1) = 0;
                volume(2, 1, 0) = 0;
                volume(1, 1, 0) = 0;
                volume(0, 1, 2) = 0;
                volume(1, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 1, 1) = 1;
                    volume(2, 1, 0) = 3;
                    volume(1, 1, 2) = 2;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(3 * sqrt(3) - 1));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 1 - 1 / sqrt(3), sqrt(3) - 1, 0, 0, 0,

                        0, 0, 0, 0, 2 / sqrt(3), 0, 0, 0, 0,

                        0, 0, 0, sqrt(3) - 1, 1 - 1 / sqrt(3), 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees enters through the right border")
        {
            // In this case the ray enters through a border orthogonal to a non-main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, 1);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(2, 1, 1) = 0;
                volume(2, 1, 0) = 0;
                volume(2, 1, 2) = 0;
                volume(1, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 1, 0) = 4;
                    volume(1, 1, 2) = 3;
                    volume(2, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 - 2 / sqrt(3) + 3 * sqrt(3)));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 0, 1 - 1 / sqrt(3), 0, 0, 0,

                        0, 0, 0, 0, 0, 2 / sqrt(3), 0, 0, 0,

                        0, 0, 0, 0, sqrt(3) - 1, 1 - 1 / sqrt(3), 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees exits through the left border")
        {
            // In this case the ray exit through a border orthogonal to a non-main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, -1);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;
                volume(1, 1, 0) = 0;
                volume(0, 1, 1) = 0;
                volume(0, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 2) = 4;
                    volume(1, 1, 0) = 3;
                    volume(0, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(3 * sqrt(3) + 1 - 2 / sqrt(3)));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 1 - 1 / sqrt(3), sqrt(3) - 1, 0, 0, 0, 0,

                        0, 0, 0, 2 / sqrt(3), 0, 0, 0, 0, 0,

                        0, 0, 0, 1 - 1 / sqrt(3), 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees only intersects a single voxel")
        {
            // special case - no interior voxels, entry and exit voxels are the same
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, -2);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(sqrt(3) - 1));

                    sino[0] = 1;
                    backProj << 0, 0, 0, sqrt(3) - 1, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                }
            }
        }
    }
}
