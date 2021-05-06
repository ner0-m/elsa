/**
 * @file test_VolumeDescriptor.cpp
 *
 * @brief Tests for VolumeDescriptor class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite to use doctest and BDD
 * @author Tobias Lasser - rewrite and added code coverage
 * @author Nikola Dinev - tests for automatic descriptor generation
 */

#include "doctest/doctest.h"
#include "Error.h"
#include "VolumeDescriptor.h"
#include "PartitionDescriptor.h"
#include "DescriptorUtils.h"
#include <stdexcept>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("VolumeDescriptor: Constructing VolumeDescriptors")
{
    GIVEN("various 1D descriptor sizes")
    {
        // Double {{ and }} needed!
        IndexVector_t validNumCoeff{{20}};
        RealVector_t validSpacing{{2.5}};
        RealVector_t invalidSpacing{{3.5, 1.5}};
        IndexVector_t invalidNumCoeff{{-10}};

        WHEN("using a valid number of coefficients and no spacing")
        {
            // Origin of volume
            const RealVector_t origin = 0.5 * (validNumCoeff.cast<real_t>().array());

            const VolumeDescriptor dd(validNumCoeff);

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd.getSpacingPerDimension(), RealVector_t::Ones(1));
                REQUIRE_EQ(dd.getLocationOfOrigin(), origin);
            }

            const VolumeDescriptor dd1({20});

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd.getSpacingPerDimension(), RealVector_t::Ones(1));
                REQUIRE_EQ(dd.getLocationOfOrigin(), origin);
            }
        }

        WHEN("using an invalid number of coefficients and no spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(VolumeDescriptor({-10}), InvalidArgumentError);
            }
        }

        WHEN("using an invalid number of coefficients and valid spacing")
        {
            THEN("an exception is thrown")
            {
                // Try all possible combinations of constructors with initializer list
                REQUIRE_THROWS_AS(VolumeDescriptor(invalidNumCoeff, validSpacing),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({-10}, {2.5}), InvalidArgumentError);
            }
        }

        WHEN("using a valid number of coefficients and spacing")
        {
            // Origin of volume
            RealVector_t origin =
                0.5 * (validNumCoeff.cast<real_t>().array() * validSpacing.array());

            const VolumeDescriptor dd1(validNumCoeff, validSpacing);

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd1.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd1.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd1.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd1.getSpacingPerDimension(), validSpacing);
                REQUIRE_EQ(dd1.getLocationOfOrigin(), origin);
            }

            const VolumeDescriptor dd2({20}, {2.5});

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd2.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd2.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd2.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd2.getSpacingPerDimension(), validSpacing);
                REQUIRE_EQ(dd2.getLocationOfOrigin(), origin);
            }
        }

        WHEN("using a valid number of coefficients and mismatched spacing")
        {
            THEN("an exception is thrown")
            {
                // Try all possible combinations of constructors
                REQUIRE_THROWS_AS(VolumeDescriptor(validNumCoeff, invalidSpacing),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({20}, {3.5, 1.5}), InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({20}, {-3.5}), InvalidArgumentError);
            }
        }
    }

    GIVEN("various 2D descriptor sizes")
    {
        IndexVector_t validNumCoeff{{12, 15}};
        RealVector_t validSpacing{{1.5, 2.5}};
        RealVector_t invalidSpacing{{1.5}};
        IndexVector_t invalidNumCoeff{{12, -1, 18}};

        WHEN("using a valid number of coefficients and no spacing")
        {
            // Origin of volume
            RealVector_t origin = 0.5 * (validNumCoeff.cast<real_t>().array());

            const VolumeDescriptor dd1(validNumCoeff);

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd1.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd1.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd1.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd1.getSpacingPerDimension(), RealVector_t::Ones(2));
                REQUIRE_EQ(dd1.getLocationOfOrigin(), origin);
            }

            const VolumeDescriptor dd2({12, 15});

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd2.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd2.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd2.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd2.getSpacingPerDimension(), RealVector_t::Ones(2));
                REQUIRE_EQ(dd2.getLocationOfOrigin(), origin);
            }
        }

        WHEN("using an invalid number of coefficients and no spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(VolumeDescriptor({12, -1, 18}), InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({12, -1}), InvalidArgumentError);
            }
        }

        WHEN("using an invalid number of coefficients and valid spacing")
        {
            IndexVector_t invalidNumCoeff2 = validNumCoeff;
            invalidNumCoeff2[0] = -1;

            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(VolumeDescriptor(invalidNumCoeff2, validSpacing),
                                  InvalidArgumentError);

                REQUIRE_THROWS_AS(VolumeDescriptor({12, -1}, {1.5, 2.5}), InvalidArgumentError);
            }
        }

        WHEN("using a valid number of coefficients and spacing")
        {
            // Origin of volume
            RealVector_t origin =
                0.5 * (validNumCoeff.cast<real_t>().array() * validSpacing.array());

            const VolumeDescriptor dd1(validNumCoeff, validSpacing);

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd1.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd1.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd1.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd1.getSpacingPerDimension(), validSpacing);
                REQUIRE_EQ(dd1.getLocationOfOrigin(), origin);
            }

            const VolumeDescriptor dd2({12, 15}, {1.5, 2.5});

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd2.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd2.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd2.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd2.getSpacingPerDimension(), validSpacing);
                REQUIRE_EQ(dd2.getLocationOfOrigin(), origin);
            }
        }

        WHEN("using a valid number of coefficients and mismatched spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(VolumeDescriptor(validNumCoeff, invalidSpacing),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({12, 15}, {-1.5, 2.0}), InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({12, 15}, {1.5, 2.0, 3.5}),
                                  InvalidArgumentError);
            }
        }
    }

    GIVEN("various 3D descriptor sizes")
    {
        IndexVector_t validNumCoeff{{12, 15, 25}};
        RealVector_t validSpacing{{1.5, 2.5, 4.5}};
        RealVector_t invalidSpacing{{1.5, 2.5}};
        IndexVector_t invalidNumCoeff{{12, 15, -1}};

        WHEN("using a valid number of coefficients and no spacing")
        {
            RealVector_t origin = 0.5 * (validNumCoeff.cast<real_t>().array());

            const VolumeDescriptor dd1(validNumCoeff);

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd1.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd1.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd1.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd1.getSpacingPerDimension(), RealVector_t::Ones(3));
                REQUIRE_EQ(dd1.getLocationOfOrigin(), origin);
            }

            const VolumeDescriptor dd2({12, 15, 25});

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd2.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd2.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd2.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd2.getSpacingPerDimension(), RealVector_t::Ones(3));
                REQUIRE_EQ(dd2.getLocationOfOrigin(), origin);
            }
        }

        WHEN("using an invalid number of coefficients and no spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(VolumeDescriptor({12, 15, -1}), InvalidArgumentError);
            }
        }

        WHEN("using an invalid number of coefficients and valid spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(VolumeDescriptor(invalidNumCoeff, validSpacing),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({12, 15, -1}, {1.5, 2.5, 4.5}),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({12, 15}, {1.5, 2.5, 4.5}),
                                  InvalidArgumentError);
            }
        }

        WHEN("using a valid number of coefficients and spacing")
        {
            RealVector_t origin =
                0.5 * (validNumCoeff.cast<real_t>().array() * validSpacing.array());

            const VolumeDescriptor dd1(validNumCoeff, validSpacing);

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd1.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd1.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd1.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd1.getSpacingPerDimension(), validSpacing);
                REQUIRE_EQ(dd1.getLocationOfOrigin(), origin);
            }

            const VolumeDescriptor dd2({12, 15, 25}, {1.5, 2.5, 4.5});

            THEN("everything is set correctly")
            {
                REQUIRE_EQ(dd2.getNumberOfDimensions(), validNumCoeff.size());
                REQUIRE_EQ(dd2.getNumberOfCoefficients(), validNumCoeff.prod());
                REQUIRE_EQ(dd2.getNumberOfCoefficientsPerDimension(), validNumCoeff);
                REQUIRE_EQ(dd2.getSpacingPerDimension(), validSpacing);
                REQUIRE_EQ(dd2.getLocationOfOrigin(), origin);
            }
        }

        WHEN("using a valid number of coefficients and mismatched spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(VolumeDescriptor(validNumCoeff, invalidSpacing),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({12, 15, 25}, {1.5, 2.5}), InvalidArgumentError);
                REQUIRE_THROWS_AS(VolumeDescriptor({12, 15, 25}, {1.5, 2.5, -4.5}),
                                  InvalidArgumentError);
            }
        }
    }
}

TEST_CASE("VolumeDescriptor: Testing clone()")
{
    GIVEN("1D descriptors")
    {
        VolumeDescriptor dd({1, 17});
        VolumeDescriptor ddWithSpacing({1, 17}, {1, 2.75});

        WHEN("cloning the VolumeDescriptor")
        {
            auto ddClone = dd.clone();
            auto ddWithSpacingClone = ddWithSpacing.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(ddClone.get(), &dd);
                REQUIRE_EQ(*ddClone, dd);

                REQUIRE_NE(ddWithSpacingClone.get(), &ddWithSpacing);
                REQUIRE_EQ(*ddWithSpacingClone, ddWithSpacing);
            }
        }
    }

    GIVEN("2D descriptors")
    {
        VolumeDescriptor dd({20, 25});
        VolumeDescriptor ddWithSpacing({20, 25}, {1.5, 3.5});

        WHEN("cloning the VolumeDescriptor")
        {
            auto ddClone = dd.clone();
            auto ddWithSpacingClone = ddWithSpacing.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(ddClone.get(), &dd);
                REQUIRE_EQ(*ddClone, dd);

                REQUIRE_NE(ddWithSpacingClone.get(), &ddWithSpacing);
                REQUIRE_EQ(*ddWithSpacingClone, ddWithSpacing);
            }
        }
    }

    GIVEN("3D descriptors")
    {
        VolumeDescriptor dd({20, 25, 30});
        VolumeDescriptor ddWithSpacing({20, 25, 30}, {1.5, 3.5, 5.5});

        WHEN("cloning the VolumeDescriptor")
        {
            auto ddClone = dd.clone();
            auto ddWithSpacingClone = ddWithSpacing.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(ddClone.get(), &dd);
                REQUIRE_EQ(*ddClone, dd);

                REQUIRE_NE(ddWithSpacingClone.get(), &ddWithSpacing);
                REQUIRE_EQ(*ddWithSpacingClone, ddWithSpacing);
            }
        }
    }
}

TEST_CASE("VolumeDescriptor: Testing calculation of Coordinates and indices")
{
    GIVEN("1D descriptors")
    {
        IndexVector_t numCoeffs{{11}};

        VolumeDescriptor dd(numCoeffs);

        WHEN("converting coordinates to indices")
        {
            IndexVector_t coordinate1(1);
            coordinate1 << 0;
            IndexVector_t coordinate2(1);
            coordinate2 << numCoeffs(0) - 1;

            IndexVector_t coordinateInvalid(2);
            coordinateInvalid << 2, 1;

            THEN("the index is correct")
            {
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate1), 0);
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate2), numCoeffs(0) - 1);
                REQUIRE_THROWS_AS(dd.getIndexFromCoordinate(coordinateInvalid),
                                  InvalidArgumentError);
            }
        }

        WHEN("converting indices to coordinates")
        {
            index_t index1 = 0;
            index_t index2 = numCoeffs(0) - 1;
            index_t indexInvalid1 = -2;
            index_t indexInvalid2 = numCoeffs(0);

            THEN("the coordinate is correct")
            {
                REQUIRE_EQ(dd.getCoordinateFromIndex(index1), IndexVector_t::Constant(1, 0));
                REQUIRE_EQ(dd.getCoordinateFromIndex(index2),
                           IndexVector_t::Constant(1, numCoeffs(0) - 1));
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid1), InvalidArgumentError);
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid2), InvalidArgumentError);
            }
        }
    }

    GIVEN("2D descriptors")
    {
        IndexVector_t numCoeffs{{11, 15}};
        VolumeDescriptor dd(numCoeffs);

        WHEN("converting coordinates to indices")
        {
            IndexVector_t coordinate1(2);
            coordinate1 << 0, 0;
            IndexVector_t coordinate2(2);
            coordinate2 << 0, numCoeffs(1) - 1;
            IndexVector_t coordinate3(2);
            coordinate3 << numCoeffs(0) - 1, numCoeffs(1) - 1;

            IndexVector_t coordinateInvalid(1);
            coordinateInvalid << 5;

            THEN("the index is correct")
            {
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate1), 0);
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate2),
                           numCoeffs(0) * (numCoeffs(1) - 1));
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate3),
                           numCoeffs(0) - 1 + numCoeffs(0) * (numCoeffs(1) - 1));
                REQUIRE_THROWS_AS(dd.getIndexFromCoordinate(coordinateInvalid),
                                  InvalidArgumentError);
            }
        }

        WHEN("converting indices to coordinates")
        {
            index_t index1 = 0;
            index_t index2 = numCoeffs(0) - 1;
            index_t index3 = numCoeffs(0) * (numCoeffs(1) - 1) + (numCoeffs(0) - 3);
            index_t indexInvalid1 = -1;
            index_t indexInvalid2 = numCoeffs(0) * numCoeffs(1);

            THEN("the coordinate is correct")
            {
                IndexVector_t coordinate1(2);
                coordinate1 << 0, 0;
                REQUIRE_EQ(dd.getCoordinateFromIndex(index1), coordinate1);

                IndexVector_t coordinate2(2);
                coordinate2 << numCoeffs(0) - 1, 0;
                REQUIRE_EQ(dd.getCoordinateFromIndex(index2), coordinate2);

                IndexVector_t coordinate3(2);
                coordinate3 << numCoeffs(0) - 3, numCoeffs(1) - 1;
                REQUIRE_EQ(dd.getCoordinateFromIndex(index3), coordinate3);

                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid1), InvalidArgumentError);
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid2), InvalidArgumentError);
            }
        }
    }

    GIVEN("3D descriptors")
    {
        IndexVector_t numCoeffs{{9, 13, 17}};
        VolumeDescriptor dd(numCoeffs);

        WHEN("converting coordinates to indices")
        {
            IndexVector_t coordinate1(3);
            coordinate1 << 0, 0, 0;
            IndexVector_t coordinate2(3);
            coordinate2 << numCoeffs(0) - 2, 0, 0;
            IndexVector_t coordinate3(3);
            coordinate3 << numCoeffs(0) - 5, numCoeffs(1) - 3, 0;
            IndexVector_t coordinate4(3);
            coordinate4 << numCoeffs(0) - 4, numCoeffs(1) - 2, numCoeffs(2) - 1;

            IndexVector_t coordinateInvalid(2);
            coordinateInvalid << 2, 2;

            THEN("the index is correct")
            {
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate1), 0);
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate2), numCoeffs(0) - 2);
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate3),
                           numCoeffs(0) - 5 + numCoeffs(0) * (numCoeffs(1) - 3));
                REQUIRE_EQ(dd.getIndexFromCoordinate(coordinate4),
                           numCoeffs(0) - 4 + numCoeffs(0) * (numCoeffs(1) - 2)
                               + numCoeffs(0) * numCoeffs(1) * (numCoeffs(2) - 1));
                REQUIRE_THROWS_AS(dd.getIndexFromCoordinate(coordinateInvalid),
                                  InvalidArgumentError);
            }
        }

        WHEN("converting indices to coordinates")
        {
            index_t index1 = 0;
            index_t index2 = numCoeffs(0) - 7;
            index_t index3 = numCoeffs(0) - 6 + numCoeffs(0) * (numCoeffs(1) - 8);
            index_t index4 = numCoeffs(0) - 5 + numCoeffs(0) * (numCoeffs(1) - 7)
                             + numCoeffs(0) * numCoeffs(1) * (numCoeffs(2) - 3);
            index_t indexInvalid1 = -3;
            index_t indexInvalid2 = numCoeffs(0) * numCoeffs(1) * numCoeffs(2);

            THEN("the coordinate is correct")
            {
                IndexVector_t coordinate1(3);
                coordinate1 << 0, 0, 0;
                REQUIRE_EQ(dd.getCoordinateFromIndex(index1), coordinate1);

                IndexVector_t coordinate2(3);
                coordinate2 << numCoeffs(0) - 7, 0, 0;
                REQUIRE_EQ(dd.getCoordinateFromIndex(index2), coordinate2);

                IndexVector_t coordinate3(3);
                coordinate3 << numCoeffs(0) - 6, numCoeffs(1) - 8, 0;
                REQUIRE_EQ(dd.getCoordinateFromIndex(index3), coordinate3);

                IndexVector_t coordinate4(3);
                coordinate4 << numCoeffs(0) - 5, numCoeffs(1) - 7, numCoeffs(2) - 3;
                REQUIRE_EQ(dd.getCoordinateFromIndex(index4), coordinate4);

                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid1), InvalidArgumentError);
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid2), InvalidArgumentError);
            }
        }
    }
}

TEST_CASE("VolumeDescriptor: Finding the best common descriptor")
{
    IndexVector_t numCoeffs{{9, 13, 17}};
    VolumeDescriptor dd{numCoeffs};

    GIVEN("an empty descriptor list")
    {
        THEN("trying to determine the best common descriptor throws an error")
        {
            REQUIRE_THROWS_AS(bestCommon(std::vector<const DataDescriptor*>{}),
                              InvalidArgumentError);
        }
    }

    GIVEN("a single descriptor")
    {
        PartitionDescriptor pd{dd, 5};
        THEN("the best common descriptor is the descriptor itself")
        {
            auto common1 = bestCommon(dd);
            auto common2 = bestCommon(pd);

            REQUIRE_EQ(*common1, dd);
            REQUIRE_EQ(*common2, pd);
        }
    }

    GIVEN("two equal PartitionDescriptors")
    {
        PartitionDescriptor pd1{dd, 5};
        PartitionDescriptor pd2{dd, 5};

        THEN("the best common descriptor is the same as the input descriptors")
        {
            auto common = bestCommon(pd1, pd2);
            REQUIRE_EQ(pd1, *common);
            REQUIRE_EQ(pd2, *common);
        }
    }

    GIVEN("a PartitionDescriptor and its base")
    {
        PartitionDescriptor pd{dd, 5};

        THEN("the best common descriptor is the base descriptor")
        {
            auto common1 = bestCommon(pd, dd);
            auto common2 = bestCommon(dd, pd);

            REQUIRE_EQ(*common1, dd);
            REQUIRE_EQ(*common2, dd);
        }
    }

    GIVEN("a PartitionDescriptor and its base but with different spacing")
    {
        VolumeDescriptor dds2{numCoeffs, dd.getSpacingPerDimension() * 2};
        PartitionDescriptor pd{dd, 5};
        PartitionDescriptor pds2{dds2, 5};

        THEN("the best common descriptor is the base descriptor with default spacing")
        {
            auto common1 = bestCommon(pd, dds2);
            auto common2 = bestCommon(dds2, pd);
            auto common3 = bestCommon(pds2, dd);
            auto common4 = bestCommon(dd, pds2);

            REQUIRE_EQ(*common1, dd);
            REQUIRE_EQ(*common2, dd);
            REQUIRE_EQ(*common3, dd);
            REQUIRE_EQ(*common4, dd);
        }
    }

    GIVEN("two equal non-block descriptors")
    {
        VolumeDescriptor dd2{numCoeffs};

        THEN("the best common descriptor is the same as the input descriptors")
        {
            auto common = bestCommon(dd, dd2);

            REQUIRE_EQ(*common, dd);
        }
    }

    GIVEN("two non-block descriptors that differ only in spacing")
    {
        VolumeDescriptor dds2{numCoeffs, dd.getSpacingPerDimension() * 2};
        VolumeDescriptor dds3{numCoeffs, dd.getSpacingPerDimension() * 3};

        THEN("the best common descriptor is the base descriptor with default spacing")
        {
            auto common1 = bestCommon(dds2, dds3);
            auto common2 = bestCommon(dds3, dds2);

            REQUIRE_EQ(*common1, dd);
            REQUIRE_EQ(*common2, dd);
        }
    }

    GIVEN("two descriptors with same number of dimensions and size but different number of "
          "coefficients per dimensions")
    {
        IndexVector_t numCoeffs2 = numCoeffs.reverse();

        VolumeDescriptor dd2{numCoeffs2};
        VolumeDescriptor dds2{numCoeffs, dd.getSpacingPerDimension() * 2};
        VolumeDescriptor dd2s2{numCoeffs2, dd2.getSpacingPerDimension() * 2};

        THEN("the best common descriptor is the linearized descriptor with default spacing")
        {
            auto common1 = bestCommon(dd2, dd);
            auto common2 = bestCommon(dd, dd2);
            auto common3 = bestCommon(dds2, dd2s2);
            auto common4 = bestCommon(dd2s2, dds2);

            VolumeDescriptor expected{IndexVector_t::Constant(1, dd.getNumberOfCoefficients())};
            REQUIRE_EQ(*common1, expected);
            REQUIRE_EQ(*common2, expected);
            REQUIRE_EQ(*common3, expected);
            REQUIRE_EQ(*common4, expected);
        }
    }

    GIVEN("two descriptors with different number of dimensions but same size")
    {
        IndexVector_t numCoeffs2 = numCoeffs.head(numCoeffs.size() - 1);
        numCoeffs2[numCoeffs2.size() - 1] *= numCoeffs[numCoeffs.size() - 1];
        VolumeDescriptor dd2{numCoeffs2};

        THEN("the best common descriptor is the linearized descriptor with default spacing")
        {
            auto common1 = bestCommon(dd2, dd);
            auto common2 = bestCommon(dd, dd2);

            VolumeDescriptor expected{IndexVector_t::Constant(1, dd.getNumberOfCoefficients())};
            REQUIRE_EQ(*common1, expected);
            REQUIRE_EQ(*common2, expected);
        }
    }

    GIVEN("two descriptors with different sizes")
    {
        IndexVector_t numCoeffs2 = numCoeffs;
        numCoeffs2[0] += 1;
        VolumeDescriptor dd2{numCoeffs2};

        THEN("trying to determine the best common descriptor throws an error")
        {
            REQUIRE_THROWS_AS(bestCommon(dd2, dd), InvalidArgumentError);
            REQUIRE_THROWS_AS(bestCommon(dd, dd2), InvalidArgumentError);
        }
    }
}

TEST_SUITE_END();
