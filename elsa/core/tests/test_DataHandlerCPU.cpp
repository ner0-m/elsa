/**
 * \file test_DataHandlerCPU.cpp
 *
 * \brief Tests for DataHandlerCPU class
 *
 * \author David Frank - initial code
 * \author Tobias Lasser - rewrite and code coverage
 */

#include <catch2/catch.hpp>
#include "DataHandlerCPU.h"

using namespace elsa;

SCENARIO("Constructing DataHandlerCPU") {
    GIVEN("a certain size") {
        index_t size = 314;

        WHEN("constructing with zeros") {
            DataHandlerCPU dh(size);

            THEN("it has the correct size") {
                REQUIRE(size == dh.getSize());
            }

            THEN("it has a zero vector") {
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == 0.0);
            }
        }

        WHEN("constructing with a given vector") {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(size);
            DataHandlerCPU dh(randVec);

            for (index_t i = 0; i < size; ++i)
                REQUIRE(dh[i] == randVec(i));
        }
    }
}


SCENARIO("Cloning DataHandlerCPU") {
    GIVEN("some DataHandlerCPU") {
        index_t size = 728;
        DataHandlerCPU dh(size);

        WHEN("cloning") {
            auto dhClone = dh.clone();

            THEN("everything matches") {
                REQUIRE(dhClone.get() != &dh);
                REQUIRE(*dhClone == dh);

                REQUIRE(dhClone->getSize() == dh.getSize());
            }
        }
    }
}


SCENARIO("Testing the reduction operations of DataHandlerCPU") {
    GIVEN("some DataHandlerCPU") {
        index_t size = 284;

        WHEN("putting in some random data") {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(size);
            DataHandlerCPU dh(randVec);

            THEN("the reductions work as expected") {
                REQUIRE(dh.sum() == Approx(randVec.sum()) );
                REQUIRE(dh.squaredNorm() == Approx(randVec.squaredNorm()) );

                Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(size);
                DataHandlerCPU dh2(randVec2);

                REQUIRE(dh.dot(dh2) == Approx(randVec.dot(randVec2)) );
            }

            THEN("the dot product expects correctly sized arguments") {
                index_t wrongSize = size - 1;
                Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(wrongSize);
                DataHandlerCPU dh2(randVec2);

                REQUIRE_THROWS_AS(dh.dot(dh2), std::invalid_argument);
            }
        }
    }
}


SCENARIO("Testing the element-wise operations of DataHandlerCPU") {
    GIVEN("some DataHandlerCPU") {
        index_t size = 567;

        WHEN("putting in some random data") {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(size);
            DataHandlerCPU dh(randVec);

            THEN("the element-wise unary operations work as expected") {
                auto dhSquared = dh.square();
                for (index_t i = 0; i < size; ++i)
                    REQUIRE( (*dhSquared)[i] == Approx(randVec(i)*randVec(i)) );

                auto dhSqrt = dh.sqrt();
                for (index_t i = 0; i < size; ++i)
                    if (randVec(i) >= 0)
                        REQUIRE( (*dhSqrt)[i] == Approx(std::sqrt(randVec(i))) );

                auto dhExp = dh.exp();
                for (index_t i = 0; i < size; ++i)
                    REQUIRE( (*dhExp)[i] == Approx(std::exp(randVec(i))) );

                auto dhLog = dh.log();
                for (index_t i = 0; i < size; ++i)
                    if (randVec(i) > 0)
                        REQUIRE( (*dhLog)[i] == Approx(log(randVec(i))) );
            }

            THEN("the element-wise binary vector operations work as expected") {
                DataHandlerCPU oldDh = dh;

                Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(size);
                DataHandlerCPU dh2(randVec2);

                dh += dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] + dh2[i]);

                dh = oldDh;
                dh -= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] - dh2[i]);

                dh = oldDh;
                dh *= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] * dh2[i]);

                dh = oldDh;
                dh /= dh2;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != 0)
                        REQUIRE(dh[i] == oldDh[i] / dh2[i]);
            }

            THEN("the element-wise binary scalar operations work as expected") {
                DataHandlerCPU oldDh = dh;
                float scalar = 3.5;

                dh += scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] + scalar);

                dh = oldDh;
                dh -= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] - scalar);

                dh = oldDh;
                dh *= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] * scalar);

                dh = oldDh;
                dh /= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] / scalar);
            }

            THEN("the element-wise assignment of a scalar works as expected") {
                float scalar = 47.11;

                dh = scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == scalar);
            }
        }
    }
}


SCENARIO("Testing the arithmetic operations with DataHandler arguments") {
    GIVEN("some DataHandlers") {
        index_t size = 1095;
        Eigen::VectorXf randVec = Eigen::VectorXf::Random(size);
        Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(size);
        DataHandlerCPU dh(randVec);
        DataHandlerCPU dh2(randVec2);

        THEN("the binary element-wise operations work as expected") {
            auto resultPlus = dh + dh2;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultPlus)[i] == dh[i] + dh2[i]);

            auto resultMinus = dh - dh2;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMinus)[i] == dh[i] - dh2[i]);

            auto resultMult = dh * dh2;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMult)[i] == dh[i] * dh2[i]);

            auto resultDiv = dh / dh2;
            for (index_t i = 0; i < size; ++i)
                if (dh2[i] != 0)
                    REQUIRE((*resultDiv)[i] == dh[i] / dh2[i]);
        }

        THEN("the operations with a scalar work as expected") {
            float scalar = 4.7;

            auto resultScalarPlus = scalar + dh;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultScalarPlus)[i] == scalar + dh[i]);

            auto resultPlusScalar = dh + scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultPlusScalar)[i] == dh[i] + scalar);

            auto resultScalarMinus = scalar - dh;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultScalarMinus)[i] == scalar - dh[i]);

            auto resultMinusScalar = dh - scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMinusScalar)[i] == dh[i] - scalar);

            auto resultScalarMult = scalar * dh;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultScalarMult)[i] == scalar * dh[i]);

            auto resultMultScalar = dh * scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMultScalar)[i] == dh[i] * scalar);

            auto resultScalarDiv = scalar / dh;
            for (index_t i = 0; i < size; ++i)
                if (dh[i] != 0)
                REQUIRE((*resultScalarDiv)[i] == scalar / dh[i]);

            auto resultDivScalar = dh / scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultDivScalar)[i] == dh[i] / scalar);
        }
    }
}