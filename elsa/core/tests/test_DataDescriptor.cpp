/**
 * \file test_DataDescriptor.cpp
 *
 * \brief Tests for DataDescriptor class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite to use Catch and BDD
 * \author Tobias Lasser - rewrite and added code coverage
 * \author Nikola Dinev - tests for automatic descriptor generation
 */

#include <catch2/catch.hpp>
#include "DataDescriptor.h"
#include "PartitionDescriptor.h"
#include <stdexcept>
#include <iostream>

using namespace elsa;

SCENARIO("Constructing DataDescriptors")
{
    GIVEN("various 1D descriptor sizes")
    {
        IndexVector_t validNumCoeff(1);
        validNumCoeff << 20;

        RealVector_t validSpacing(1);
        validSpacing << 2.5;

        RealVector_t invalidSpacing(2);
        invalidSpacing << 3.5, 1.5;

        IndexVector_t invalidNumCoeff(1);
        invalidNumCoeff << -10;

        WHEN("using a valid number of coefficients and no spacing")
        {
            DataDescriptor dd(validNumCoeff);

            THEN("everything is set correctly")
            {
                REQUIRE(dd.getNumberOfDimensions() == validNumCoeff.size());
                REQUIRE(dd.getNumberOfCoefficients() == validNumCoeff.prod());
                REQUIRE(dd.getNumberOfCoefficientsPerDimension() == validNumCoeff);
                REQUIRE(dd.getSpacingPerDimension() == RealVector_t::Ones(1));
                RealVector_t origin = 0.5 * (validNumCoeff.cast<real_t>().array());
                REQUIRE(dd.getLocationOfOrigin() == origin);
            }
        }

        WHEN("using an invalid number of coefficients and no spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(invalidNumCoeff), std::invalid_argument);
            }
        }

        WHEN("using an invalid number of coefficients and valid spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(invalidNumCoeff, validSpacing),
                                  std::invalid_argument);
            }
        }

        WHEN("using a valid number of coefficients and spacing")
        {
            DataDescriptor dd(validNumCoeff, validSpacing);

            THEN("everything is set correctly")
            {
                REQUIRE(dd.getNumberOfDimensions() == validNumCoeff.size());
                REQUIRE(dd.getNumberOfCoefficients() == validNumCoeff.prod());
                REQUIRE(dd.getNumberOfCoefficientsPerDimension() == validNumCoeff);
                REQUIRE(dd.getSpacingPerDimension() == validSpacing);
                RealVector_t origin =
                    0.5 * (validNumCoeff.cast<real_t>().array() * validSpacing.array());
                REQUIRE(dd.getLocationOfOrigin() == origin);
            }
        }

        WHEN("using a valid number of coefficients and mismatched spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(validNumCoeff, invalidSpacing),
                                  std::invalid_argument);
            }
        }
    }

    GIVEN("various 2D descriptor sizes")
    {
        IndexVector_t validNumCoeff(2);
        validNumCoeff << 12, 15;

        RealVector_t validSpacing(2);
        validSpacing << 1.5, 2.5;

        RealVector_t invalidSpacing(1);
        invalidSpacing << 1.5;

        IndexVector_t invalidNumCoeff(3);
        invalidNumCoeff << 12, -1, 18;

        WHEN("using a valid number of coefficients and no spacing")
        {
            DataDescriptor dd(validNumCoeff);

            THEN("everything is set correctly")
            {
                REQUIRE(dd.getNumberOfDimensions() == validNumCoeff.size());
                REQUIRE(dd.getNumberOfCoefficients() == validNumCoeff.prod());
                REQUIRE(dd.getNumberOfCoefficientsPerDimension() == validNumCoeff);
                REQUIRE(dd.getSpacingPerDimension() == RealVector_t::Ones(2));
                RealVector_t origin = 0.5 * (validNumCoeff.cast<real_t>().array());
                REQUIRE(dd.getLocationOfOrigin() == origin);
            }
        }

        WHEN("using an invalid number of coefficients and no spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(invalidNumCoeff), std::invalid_argument);
            }
        }

        WHEN("using an invalid number of coefficients and valid spacing")
        {
            IndexVector_t invalidNumCoeff2 = validNumCoeff;
            invalidNumCoeff2[0] = -1;

            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(invalidNumCoeff2, validSpacing),
                                  std::invalid_argument);
            }
        }

        WHEN("using a valid number of coefficients and spacing")
        {
            DataDescriptor dd(validNumCoeff, validSpacing);

            THEN("everything is set correctly")
            {
                REQUIRE(dd.getNumberOfDimensions() == validNumCoeff.size());
                REQUIRE(dd.getNumberOfCoefficients() == validNumCoeff.prod());
                REQUIRE(dd.getNumberOfCoefficientsPerDimension() == validNumCoeff);
                REQUIRE(dd.getSpacingPerDimension() == validSpacing);
                RealVector_t origin =
                    0.5 * (validNumCoeff.cast<real_t>().array() * validSpacing.array());
                REQUIRE(dd.getLocationOfOrigin() == origin);
            }
        }

        WHEN("using a valid number of coefficients and mismatched spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(validNumCoeff, invalidSpacing),
                                  std::invalid_argument);
            }
        }
    }

    GIVEN("various 3D descriptor sizes")
    {
        IndexVector_t validNumCoeff(3);
        validNumCoeff << 12, 15, 25;

        RealVector_t validSpacing(3);
        validSpacing << 1.5, 2.5, 4.5;

        RealVector_t invalidSpacing(2);
        invalidSpacing << 1.5, 2.5;

        IndexVector_t invalidNumCoeff(3);
        invalidNumCoeff << 12, 15, -1;

        WHEN("using a valid number of coefficients and no spacing")
        {
            DataDescriptor dd(validNumCoeff);

            THEN("everything is set correctly")
            {
                REQUIRE(dd.getNumberOfDimensions() == validNumCoeff.size());
                REQUIRE(dd.getNumberOfCoefficients() == validNumCoeff.prod());
                REQUIRE(dd.getNumberOfCoefficientsPerDimension() == validNumCoeff);
                REQUIRE(dd.getSpacingPerDimension() == RealVector_t::Ones(3));
                RealVector_t origin = 0.5 * (validNumCoeff.cast<real_t>().array());
                REQUIRE(dd.getLocationOfOrigin() == origin);
            }
        }

        WHEN("using an invalid number of coefficients and no spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(invalidNumCoeff), std::invalid_argument);
            }
        }

        WHEN("using an invalid number of coefficients and valid spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(invalidNumCoeff, validSpacing),
                                  std::invalid_argument);
            }
        }

        WHEN("using a valid number of coefficients and spacing")
        {
            DataDescriptor dd(validNumCoeff, validSpacing);

            THEN("everything is set correctly")
            {
                REQUIRE(dd.getNumberOfDimensions() == validNumCoeff.size());
                REQUIRE(dd.getNumberOfCoefficients() == validNumCoeff.prod());
                REQUIRE(dd.getNumberOfCoefficientsPerDimension() == validNumCoeff);
                REQUIRE(dd.getSpacingPerDimension() == validSpacing);
                RealVector_t origin =
                    0.5 * (validNumCoeff.cast<real_t>().array() * validSpacing.array());
                REQUIRE(dd.getLocationOfOrigin() == origin);
            }
        }

        WHEN("using a valid number of coefficients and mismatched spacing")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataDescriptor(validNumCoeff, invalidSpacing),
                                  std::invalid_argument);
            }
        }
    }
}

SCENARIO("Cloning DataDescriptors")
{
    GIVEN("1D descriptors")
    {
        IndexVector_t numCoeffs = IndexVector_t::Constant(1, 17);
        RealVector_t spacing = RealVector_t::Constant(1, 2.75);

        DataDescriptor dd(numCoeffs);
        DataDescriptor ddWithSpacing(numCoeffs, spacing);

        WHEN("cloning the DataDescriptor")
        {
            auto ddClone = dd.clone();
            auto ddWithSpacingClone = ddWithSpacing.clone();

            THEN("everything matches")
            {
                REQUIRE(ddClone.get() != &dd);
                REQUIRE(*ddClone == dd);

                REQUIRE(ddWithSpacingClone.get() != &ddWithSpacing);
                REQUIRE(*ddWithSpacingClone == ddWithSpacing);
            }
        }
    }

    GIVEN("2D descriptors")
    {
        IndexVector_t numCoeffs(2);
        numCoeffs << 20, 25;

        RealVector_t spacing(2);
        spacing << 1.5, 3.5;

        DataDescriptor dd(numCoeffs);
        DataDescriptor ddWithSpacing(numCoeffs, spacing);

        WHEN("cloning the DataDescriptor")
        {
            auto ddClone = dd.clone();
            auto ddWithSpacingClone = ddWithSpacing.clone();

            THEN("everything matches")
            {
                REQUIRE(ddClone.get() != &dd);
                REQUIRE(*ddClone == dd);

                REQUIRE(ddWithSpacingClone.get() != &ddWithSpacing);
                REQUIRE(*ddWithSpacingClone == ddWithSpacing);
            }
        }
    }

    GIVEN("3D descriptors")
    {
        IndexVector_t numCoeffs(3);
        numCoeffs << 20, 25, 30;

        RealVector_t spacing(3);
        spacing << 1.5, 3.5, 5.5;

        DataDescriptor dd(numCoeffs);
        DataDescriptor ddWithSpacing(numCoeffs, spacing);

        WHEN("cloning the DataDescriptor")
        {
            auto ddClone = dd.clone();
            auto ddWithSpacingClone = ddWithSpacing.clone();

            THEN("everything matches")
            {
                REQUIRE(ddClone.get() != &dd);
                REQUIRE(*ddClone == dd);

                REQUIRE(ddWithSpacingClone.get() != &ddWithSpacing);
                REQUIRE(*ddWithSpacingClone == ddWithSpacing);
            }
        }
    }
}

SCENARIO("Coordinates and indices")
{
    GIVEN("1D descriptors")
    {
        IndexVector_t numCoeffs(1);
        numCoeffs << 11;

        DataDescriptor dd(numCoeffs);

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
                REQUIRE(dd.getIndexFromCoordinate(coordinate1) == 0);
                REQUIRE(dd.getIndexFromCoordinate(coordinate2) == numCoeffs(0) - 1);
                REQUIRE_THROWS_AS(dd.getIndexFromCoordinate(coordinateInvalid),
                                  std::invalid_argument);
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
                REQUIRE(dd.getCoordinateFromIndex(index1) == IndexVector_t::Constant(1, 0));
                REQUIRE(dd.getCoordinateFromIndex(index2)
                        == IndexVector_t::Constant(1, numCoeffs(0) - 1));
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid1), std::invalid_argument);
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid2), std::invalid_argument);
            }
        }
    }

    GIVEN("2D descriptors")
    {
        IndexVector_t numCoeffs(2);
        numCoeffs << 11, 15;

        DataDescriptor dd(numCoeffs);

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
                REQUIRE(dd.getIndexFromCoordinate(coordinate1) == 0);
                REQUIRE(dd.getIndexFromCoordinate(coordinate2)
                        == numCoeffs(0) * (numCoeffs(1) - 1));
                REQUIRE(dd.getIndexFromCoordinate(coordinate3)
                        == numCoeffs(0) - 1 + numCoeffs(0) * (numCoeffs(1) - 1));
                REQUIRE_THROWS_AS(dd.getIndexFromCoordinate(coordinateInvalid),
                                  std::invalid_argument);
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
                REQUIRE(dd.getCoordinateFromIndex(index1) == coordinate1);

                IndexVector_t coordinate2(2);
                coordinate2 << numCoeffs(0) - 1, 0;
                REQUIRE(dd.getCoordinateFromIndex(index2) == coordinate2);

                IndexVector_t coordinate3(2);
                coordinate3 << numCoeffs(0) - 3, numCoeffs(1) - 1;
                REQUIRE(dd.getCoordinateFromIndex(index3) == coordinate3);

                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid1), std::invalid_argument);
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid2), std::invalid_argument);
            }
        }
    }

    GIVEN("3D descriptors")
    {
        IndexVector_t numCoeffs(3);
        numCoeffs << 9, 13, 17;

        DataDescriptor dd(numCoeffs);

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
                REQUIRE(dd.getIndexFromCoordinate(coordinate1) == 0);
                REQUIRE(dd.getIndexFromCoordinate(coordinate2) == numCoeffs(0) - 2);
                REQUIRE(dd.getIndexFromCoordinate(coordinate3)
                        == numCoeffs(0) - 5 + numCoeffs(0) * (numCoeffs(1) - 3));
                REQUIRE(dd.getIndexFromCoordinate(coordinate4)
                        == numCoeffs(0) - 4 + numCoeffs(0) * (numCoeffs(1) - 2)
                               + numCoeffs(0) * numCoeffs(1) * (numCoeffs(2) - 1));
                REQUIRE_THROWS_AS(dd.getIndexFromCoordinate(coordinateInvalid),
                                  std::invalid_argument);
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
                REQUIRE(dd.getCoordinateFromIndex(index1) == coordinate1);

                IndexVector_t coordinate2(3);
                coordinate2 << numCoeffs(0) - 7, 0, 0;
                REQUIRE(dd.getCoordinateFromIndex(index2) == coordinate2);

                IndexVector_t coordinate3(3);
                coordinate3 << numCoeffs(0) - 6, numCoeffs(1) - 8, 0;
                REQUIRE(dd.getCoordinateFromIndex(index3) == coordinate3);

                IndexVector_t coordinate4(3);
                coordinate4 << numCoeffs(0) - 5, numCoeffs(1) - 7, numCoeffs(2) - 3;
                REQUIRE(dd.getCoordinateFromIndex(index4) == coordinate4);

                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid1), std::invalid_argument);
                REQUIRE_THROWS_AS(dd.getCoordinateFromIndex(indexInvalid2), std::invalid_argument);
            }
        }
    }
}

SCENARIO("Finding the best common descriptor")
{
    IndexVector_t numCoeffs(3);
    numCoeffs << 9, 13, 17;

    DataDescriptor dd{numCoeffs};

    GIVEN("an empty descriptor list")
    {
        THEN("trying to determine the best common descriptor throws an error")
        {
            REQUIRE_THROWS_AS(DataDescriptor::bestCommon(std::vector<const DataDescriptor*>{}),
                              std::invalid_argument);
        }
    }

    GIVEN("a single descriptor")
    {
        PartitionDescriptor pd{dd, 5};
        THEN("the best common descriptor is the descriptor itself")
        {
            auto common1 = DataDescriptor::bestCommon(dd);
            auto common2 = DataDescriptor::bestCommon(pd);

            REQUIRE(*common1 == dd);
            REQUIRE(*common2 == pd);
        }
    }

    GIVEN("two equal PartitionDescriptors")
    {
        PartitionDescriptor pd1{dd, 5};
        PartitionDescriptor pd2{dd, 5};

        THEN("the best common descriptor is the same as the input descriptors")
        {
            auto common = DataDescriptor::bestCommon(pd1, pd2);
            REQUIRE(pd1 == *common);
            REQUIRE(pd2 == *common);
        }
    }

    GIVEN("a PartitionDescriptor and its base")
    {
        PartitionDescriptor pd{dd, 5};

        THEN("the best common descriptor is the base descriptor")
        {
            auto common1 = DataDescriptor::bestCommon(pd, dd);
            auto common2 = DataDescriptor::bestCommon(dd, pd);

            REQUIRE(*common1 == dd);
            REQUIRE(*common2 == dd);
        }
    }

    GIVEN("a PartitionDescriptor and its base but with different spacing")
    {
        DataDescriptor dds2{numCoeffs, dd.getSpacingPerDimension() * 2};
        PartitionDescriptor pd{dd, 5};
        PartitionDescriptor pds2{dds2, 5};

        THEN("the best common descriptor is the base descriptor with default spacing")
        {
            auto common1 = DataDescriptor::bestCommon(pd, dds2);
            auto common2 = DataDescriptor::bestCommon(dds2, pd);
            auto common3 = DataDescriptor::bestCommon(pds2, dd);
            auto common4 = DataDescriptor::bestCommon(dd, pds2);

            REQUIRE(*common1 == dd);
            REQUIRE(*common2 == dd);
            REQUIRE(*common3 == dd);
            REQUIRE(*common4 == dd);
        }
    }

    GIVEN("two equal non-block descriptors")
    {
        DataDescriptor dd2{numCoeffs};

        THEN("the best common descriptor is the same as the input descriptors")
        {
            auto common = DataDescriptor::bestCommon(dd, dd2);

            REQUIRE(*common == dd);
        }
    }

    GIVEN("two non-block descriptors that differ only in spacing")
    {
        DataDescriptor dds2{numCoeffs, dd.getSpacingPerDimension() * 2};
        DataDescriptor dds3{numCoeffs, dd.getSpacingPerDimension() * 3};

        THEN("the best common descriptor is the base descriptor with default spacing")
        {
            auto common1 = DataDescriptor::bestCommon(dds2, dds3);
            auto common2 = DataDescriptor::bestCommon(dds3, dds2);

            REQUIRE(*common1 == dd);
            REQUIRE(*common2 == dd);
        }
    }

    GIVEN("two descriptors with same number of dimensions and size but different number of "
          "coefficients per dimensions")
    {
        IndexVector_t numCoeffs2 = numCoeffs.reverse();

        std::cout << numCoeffs2.transpose() << std::endl;
        DataDescriptor dd2{numCoeffs2};
        DataDescriptor dds2{numCoeffs, dd.getSpacingPerDimension() * 2};
        DataDescriptor dd2s2{numCoeffs2, dd2.getSpacingPerDimension() * 2};

        THEN("the best common descriptor is the linearized descriptor with default spacing")
        {
            auto common1 = DataDescriptor::bestCommon(dd2, dd);
            auto common2 = DataDescriptor::bestCommon(dd, dd2);
            auto common3 = DataDescriptor::bestCommon(dds2, dd2s2);
            auto common4 = DataDescriptor::bestCommon(dd2s2, dds2);

            DataDescriptor expected{IndexVector_t::Constant(1, dd.getNumberOfCoefficients())};
            REQUIRE(*common1 == expected);
            REQUIRE(*common2 == expected);
            REQUIRE(*common3 == expected);
            REQUIRE(*common4 == expected);
        }
    }

    GIVEN("two descriptors with different number of dimensions but same size")
    {
        IndexVector_t numCoeffs2 = numCoeffs.head(numCoeffs.size() - 1);
        numCoeffs2[numCoeffs2.size() - 1] *= numCoeffs[numCoeffs.size() - 1];
        DataDescriptor dd2{numCoeffs2};

        THEN("the best common descriptor is the linearized descriptor with default spacing")
        {
            auto common1 = DataDescriptor::bestCommon(dd2, dd);
            auto common2 = DataDescriptor::bestCommon(dd, dd2);

            DataDescriptor expected{IndexVector_t::Constant(1, dd.getNumberOfCoefficients())};
            REQUIRE(*common1 == expected);
            REQUIRE(*common2 == expected);
        }
    }

    GIVEN("two descriptors with different sizes")
    {
        IndexVector_t numCoeffs2 = numCoeffs;
        numCoeffs2[0] += 1;
        DataDescriptor dd2{numCoeffs2};

        THEN("trying to determine the best common descriptor throws an error")
        {
            REQUIRE_THROWS_AS(DataDescriptor::bestCommon(dd2, dd), std::invalid_argument);
            REQUIRE_THROWS_AS(DataDescriptor::bestCommon(dd, dd2), std::invalid_argument);
        }
    }
}