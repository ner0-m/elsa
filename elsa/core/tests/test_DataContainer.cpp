/**
 * \file test_DataContainer.cpp
 *
 * \brief Tests for DataContainer class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite to use Catch and BDD
 * \author Tobias Lasser - rewrite and added code coverage
 */

#include <catch2/catch.hpp>
#include "DataContainer.h"

using namespace elsa;
using namespace Catch::literals; // to enable 0.0_a approximate floats

SCENARIO("Constructing DataContainers") {
    GIVEN("a DataDescriptor") {
        IndexVector_t numCoeff(3);
        numCoeff << 17, 47, 91;
        DataDescriptor desc(numCoeff);

        WHEN("constructing an empty DataContainer") {
            DataContainer dc(desc);

            THEN("it has the correct DataDescriptor") {
                REQUIRE(dc.getDataDescriptor() == desc);
            }

            THEN("it has a zero data vector of correct size") {
                REQUIRE(dc.getSize() == desc.getNumberOfCoefficients());

                for (index_t i = 0; i < desc.getNumberOfCoefficients(); ++i)
                    REQUIRE(dc[i] == 0.0);
            }
        }

        WHEN("constructing an initialized DataContainer") {
            RealVector_t data(desc.getNumberOfCoefficients());
            data.setRandom();

            DataContainer dc(desc, data);

            THEN("it has the correct DataDescriptor") {
                REQUIRE(dc.getDataDescriptor() == desc);
            }

            THEN("it has correctly initialized data") {
                REQUIRE(dc.getSize() == desc.getNumberOfCoefficients());

                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == data[i]);
            }
        }
    }

    GIVEN("another DataContainer") {
        IndexVector_t numCoeff(2);
        numCoeff << 32, 57;
        DataDescriptor desc(numCoeff);

        DataContainer otherDc(desc);
        Eigen::VectorXf randVec = Eigen::VectorXf::Random(otherDc.getSize());
        for (index_t i = 0; i < otherDc.getSize(); ++i)
            otherDc[i] = randVec(i);

        WHEN("copy constructing") {
            DataContainer dc(otherDc);

            THEN("it copied correctly") {
                REQUIRE(dc.getDataDescriptor() == otherDc.getDataDescriptor());
                REQUIRE(&dc.getDataDescriptor() != &otherDc.getDataDescriptor());

                REQUIRE(dc == otherDc);
            }
        }

        WHEN("copy assigning") {
            DataContainer dc(desc);
            dc = otherDc;

            THEN("it copied correctly") {
                REQUIRE(dc.getDataDescriptor() == otherDc.getDataDescriptor());
                REQUIRE(&dc.getDataDescriptor() != &otherDc.getDataDescriptor());

                REQUIRE(dc == otherDc);
            }
        }

        WHEN("move constructing") {
            DataContainer oldOtherDc(otherDc);

            DataContainer dc(std::move(otherDc));

            THEN("it moved correctly") {
                REQUIRE(dc.getDataDescriptor() == oldOtherDc.getDataDescriptor());

                REQUIRE(dc == oldOtherDc);
            }

            THEN("the moved from object is still valid (but empty)") {
                REQUIRE(otherDc.getSize() == 1);
                REQUIRE(otherDc[0] == 0);
            }
        }

        WHEN("move assigning") {
            DataContainer oldOtherDc(otherDc);

            DataContainer dc(desc);
            dc = std::move(otherDc);

            THEN("it moved correctly") {
                REQUIRE(dc.getDataDescriptor() == oldOtherDc.getDataDescriptor());

                REQUIRE(dc == oldOtherDc);
            }

            THEN("the moved from object is still valid (but empty)") {
                REQUIRE(otherDc.getSize() == 1);
                REQUIRE(otherDc[0] == 0);
            }
        }
    }
}


SCENARIO("Element-wise access of DataContainers") {
    GIVEN("a DataContainer") {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        DataDescriptor desc(numCoeff);
        DataContainer dc(desc);

        WHEN("accessing the elements") {
            IndexVector_t coord(2);
            coord << 17, 4;
            index_t index = desc.getIndexFromCoordinate(coord);

            THEN("it works as expected when using indices/coordinates") {
                REQUIRE(dc[index] == 0.0_a);
                REQUIRE(dc(coord) == 0.0_a);

                dc[index] = 2.2;
                REQUIRE(dc[index] == 2.2_a);
                REQUIRE(dc(coord) == 2.2_a);

                dc(coord) = 3.3;
                REQUIRE(dc[index] == 3.3_a);
                REQUIRE(dc(coord) == 3.3_a);
            }
        }
    }
}


SCENARIO("Testing the reduction operations of DataContainer") {
    GIVEN("a DataContainer") {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 73, 45;
        DataDescriptor desc(numCoeff);
        DataContainer dc(desc);

        WHEN("putting in some random data") {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(dc.getSize());
            for (index_t i = 0; i < dc.getSize(); ++i)
                dc[i] = randVec(i);

            THEN("the reductions work as expected") {
                REQUIRE(dc.sum() == Approx(randVec.sum()) );
                REQUIRE(dc.l1Norm() == Approx(randVec.array().abs().sum()));
                REQUIRE(dc.lInfNorm() == Approx(randVec.array().abs().maxCoeff()));
                REQUIRE(dc.squaredL2Norm() == Approx(randVec.squaredNorm()) );

                Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(dc.getSize());
                DataContainer dc2(desc);
                for (index_t i = 0; i < dc2.getSize(); ++i)
                    dc2[i] = randVec2(i);

                REQUIRE(dc.dot(dc2) == Approx(randVec.dot(randVec2)) );
            }
        }


    }
}


SCENARIO("Testing the element-wise operations of DataContainer") {
    GIVEN("a DataContainer") {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        DataDescriptor desc(numCoeff);
        DataContainer dc(desc);

        WHEN("putting in some random data") {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(dc.getSize());
            for (index_t i = 0; i < dc.getSize(); ++i)
                dc[i] = randVec(i);

            THEN("the element-wise unary operations work as expected") {
                auto dcSquare = dc.square();
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dcSquare[i] == Approx(randVec(i) * randVec(i)));

                auto dcSqrt = dc.sqrt();
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (randVec(i) >= 0)
                        REQUIRE(dcSqrt[i] == Approx(std::sqrt(randVec(i))));

                auto dcExp = dc.exp();
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dcExp[i] == Approx(std::exp(randVec(i))));

                auto dcLog = dc.log();
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (randVec(i) > 0)
                        REQUIRE(dcLog[i] == Approx(std::log(randVec(i))));
            }

            THEN("the binary in-place addition of a scalar work as expected") {
                float scalar = 923.41;
                dc += scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) + scalar);
            }

            THEN("the binary in-place subtraction of a scalar work as expected") {
                float scalar = 74.165;
                dc -= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) - scalar);
            }

            THEN("the binary in-place multiplication with a scalar work as expected") {
                float scalar = 12.69;
                dc *= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) * scalar);
            }

            THEN("the binary in-place division by a scalar work as expected") {
                float scalar = 82.61;
                dc /= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) / scalar);
            }

            THEN("the element-wise assignment of a scalar works as expected") {
                float scalar = 123.45;
                dc = scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == scalar);
            }

        }

        WHEN("having two containers with random data") {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(dc.getSize());
            for (index_t i = 0; i < dc.getSize(); ++i)
                dc[i] = randVec(i);

            Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(dc.getSize());
            DataContainer dc2(desc);
            for (index_t i = 0; i < dc2.getSize(); ++i)
                dc2[i] = randVec2[i];

            THEN("the element-wise in-place addition works as expected") {
                dc += dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) + randVec2(i));
            }

            THEN("the element-wise in-place subtraction works as expected") {
                dc -= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) - randVec2(i));
            }

            THEN("the element-wise in-place multiplication works as expected") {
                dc *= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) * randVec2(i));
            }

            THEN("the element-wise in-place division works as expected") {
                dc /= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (dc2[i] != 0)
                        REQUIRE(dc[i] == randVec(i) / randVec2(i));
            }

        }
    }
}


SCENARIO("Testing the arithmetic operations with DataContainer arguments") {
    GIVEN("some DataContainers") {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        DataDescriptor desc(numCoeff);

        DataContainer dc(desc);
        DataContainer dc2(desc);

        Eigen::VectorXf randVec  = Eigen::VectorXf::Random(dc.getSize());
        Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(dc.getSize());

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i]  = randVec(i);
            dc2[i] = randVec2(i);
        }

        THEN("the binary element-wise operations work as expected") {
            auto resultPlus = dc + dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultPlus[i] == dc[i] + dc2[i]);

            auto resultMinus = dc - dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMinus[i] == dc[i] - dc2[i]);

            auto resultMult = dc * dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMult[i] == dc[i] * dc2[i]);

            auto resultDiv = dc / dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc2[i] != 0)
                    REQUIRE(resultDiv[i] == dc[i] / dc2[i]);
        }

        THEN("the operations with a scalar work as expected") {
            float scalar = 4.92;

            auto resultScalarPlus = scalar + dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarPlus[i] == scalar + dc[i]);

            auto resultPlusScalar = dc + scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultPlusScalar[i] == dc[i] + scalar);

            auto resultScalarMinus = scalar - dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarMinus[i] == scalar - dc[i]);

            auto resultMinusScalar = dc - scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMinusScalar[i] == dc[i] - scalar);

            auto resultScalarMult = scalar * dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarMult[i] == scalar * dc[i]);

            auto resultMultScalar = dc * scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMultScalar[i] == dc[i] * scalar);

            auto resultScalarDiv = scalar / dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc[i] != 0)
                    REQUIRE(resultScalarDiv[i] == scalar / dc[i]);

            auto resultDivScalar = dc / scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultDivScalar[i] == dc[i] / scalar);


        }
    }
}