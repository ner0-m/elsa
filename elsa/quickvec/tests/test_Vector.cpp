#include "doctest/doctest.h"

#include "Vector.cuh"
#include "Eigen/Dense"
#include "Defines.cuh"

#include <thrust/complex.h>

using namespace quickvec;

using doctest::Approx;

static float margin = 0.000001f;

TEST_CASE_TEMPLATE("Vector", TestType, index_t, float, double, thrust::complex<float>,
                   thrust::complex<double>)
{
    GIVEN("two Eigen matrices with four elements")
    {
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vec(4);
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vec2(4);
        vec[0] = static_cast<TestType>(1);
        vec[1] = static_cast<TestType>(2.3f);
        vec[2] = static_cast<TestType>(2.2f);
        vec[3] = static_cast<TestType>(-3.4f);
        vec2[0] = static_cast<TestType>(2);
        vec2[1] = static_cast<TestType>(3.3f);
        vec2[2] = static_cast<TestType>(3.2f);
        vec2[3] = static_cast<TestType>(4.4f);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vecResult(4);

        WHEN("constructing Vector and testing ctor etc.")
        {
            Vector dc(vec);
            Vector dc2(vec2);
            Vector result(vecResult);

            THEN("the elements have to be correct")
            {
                REQUIRE(dc[0] == TestType(1.f));
                REQUIRE(dc[1] == TestType(2.3f));
                REQUIRE(dc[2] == TestType(2.2f));
                REQUIRE(dc[3] == TestType(-3.4f));
                REQUIRE(dc2[0] == TestType(2.f));
                REQUIRE(dc2[1] == TestType(3.3f));
                REQUIRE(dc2[2] == TestType(3.2f));
                REQUIRE(dc2[3] == TestType(4.4f));
            }

            THEN("copy constructor creates a shallow copy")
            {
                Vector copy = dc;
                copy[0] = TestType(12);
                REQUIRE(copy[0] == dc[0]);
            }

            THEN("copy assignment creates a deep copy")
            {
                Vector<TestType> copy(dc.size());
                copy = dc;
                copy[0] = TestType(12);
                REQUIRE(copy[0] != dc[0]);
            }

            THEN("move constructor copy")
            {
                Vector copy = std::move(dc);
                REQUIRE(copy[0] == TestType(1.f));
            }

            THEN("move assignment moves data")
            {
                Vector<TestType> copy(dc.size());
                copy = std::move(dc);
                REQUIRE(copy[0] == TestType(1.f));
            }

            THEN("cloning creates a deep copy")
            {
                Vector copy = dc.clone();
                copy[0] = TestType(12);
                REQUIRE(copy[0] != dc[0]);
            }
        }

        WHEN("constructing an Vector")
        {
            Vector dc(vec);
            Vector dc2(vec2);
            Vector result(vecResult);

            THEN("checking compile time predicates")
            {
                static_assert(!isVector<float>);
                static_assert(!isExpression<float>);
                static_assert(isVector<decltype(dc)>);
                static_assert(isVector<decltype(dc2)>);
            }

            THEN("the elements have to be correct")
            {
                REQUIRE(dc[0] == TestType(1.f));
                REQUIRE(dc[1] == TestType(2.3f));
                REQUIRE(dc[2] == TestType(2.2f));
                REQUIRE(dc[3] == TestType(-3.4f));
                REQUIRE(dc2[0] == TestType(2.f));
                REQUIRE(dc2[1] == TestType(3.3f));
                REQUIRE(dc2[2] == TestType(3.2f));
                REQUIRE(dc2[3] == TestType(4.4f));
            }

            THEN("element-wise write has to work")
            {
                dc[0] = TestType(5);
                REQUIRE(dc[0] == TestType(5));
            }

            THEN("In-place scalar assign works as expected")
            {
                dc = 1;

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (TestType(1.f)) - dc[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }

            THEN("In-place scalar minus works as expected")
            {
                dc -= 1;

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] - TestType(1.f)) - dc[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }

            THEN("In-place scalar plus works as expected")
            {
                dc += 1;

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] + TestType(1.f)) - dc[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }

            THEN("In-place scalar multiply works as expected")
            {
                dc *= 2;

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] * TestType(2.f)) - dc[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }

            THEN("In-place scalar divide works as expected")
            {
                dc /= 2;

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] / TestType(2.f)) - dc[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }

            THEN("Multiplication works as expected")
            {
                Expression expr = dc * dc2;
                result.eval(expr);

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] * vec2[i]) - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }

            THEN("Subtraction works as expected")
            {
                Expression expr = dc - dc2;
                result.eval(expr);

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] - vec2[i]) - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0);
                }
            }

            THEN("Addition works as expected")
            {
                Expression expr = dc + dc2;
                result.eval(expr);

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] + vec2[i]) - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0);
                }
            }

            THEN("Division works as expected")
            {
                Expression expr = dc / dc2;
                result.eval(expr);

                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = (vec[i] / vec2[i]) - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }

            THEN("Summing all elements works as expected")
            {
                TestType difference = dc.sum() - vec.sum();
                auto diffAbs = quickvec::abs(difference);
                REQUIRE(Approx(diffAbs) == 0.f);
            }

            THEN("L2-norm works as expected")
            {
                // using thrust directly in norm is unsafe with eigen
                auto castVec = vec.template cast<std::complex<double>>();
                REQUIRE(Approx(dc.squaredl2Norm()) == castVec.squaredNorm());
            }

            THEN("L-infinity-norm works as expected")
            {
                // using thrust directly in norm does not work as expected
                auto castVec = vec.template cast<std::complex<double>>();
                REQUIRE(dc.lInfNorm() == castVec.template lpNorm<Eigen::Infinity>());
            }

            THEN("L1-norm works as expected")
            {
                // using thrust directly in norm does not work as expected
                auto castVec = vec.template cast<std::complex<double>>();
                REQUIRE(Approx(dc.l1Norm()) == castVec.template lpNorm<1>());
            }

            THEN("L0-norm works as expected")
            {
                // using thrust directly in norm does not work as expected
                auto castVec = vec.template cast<std::complex<double>>();
                REQUIRE(Approx(dc.l0PseudoNorm())
                        == (castVec.array().cwiseAbs() >= margin).count());
            }

            THEN("dot product works as expected")
            {
                TestType difference = dc.dot(dc2) - (vec.array() * vec2.array()).sum();
                auto diffAbs = quickvec::abs(difference);
                REQUIRE(Approx(diffAbs) == 0.f);
            }

            THEN("copy constructor creates a shallow copy")
            {
                Vector copy = dc;
                copy[0] = TestType(12);
                REQUIRE(copy[0] == dc[0]);
            }

            THEN("copy assignment creates a deep copy")
            {
                Vector<TestType> copy(dc.size());
                copy = dc;
                copy[0] = TestType(12);
                REQUIRE(copy[0] != dc[0]);
            }

            THEN("cloning creates a deep copy")
            {
                Vector copy = dc.clone();
                copy[0] = TestType(12);
                REQUIRE(copy[0] != dc[0]);
            }
        }

        WHEN("using complex nested expression")
        {
            Vector dc(vec);
            Vector dc2(vec2);
            Vector result(vecResult);

            Expression expr = dc * TestType(1.2f) / dc2 - dc + TestType(1.2f) * dc2;

            result.eval(expr);

            THEN("the results have to be correct")
            {
                for (long i = 0; i < vec.size(); i++) {

                    TestType directResult =
                        vec[i] * TestType(1.2f) / vec2[i] - vec[i] + TestType(1.2f) * vec2[i];
                    TestType difference = directResult - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }
        }

        WHEN("Using unary sqrt operation")
        {
            Vector dc(vec);
            Vector result(vecResult);

            auto expr = sqrt(square(dc));

            result.eval(expr);

            THEN("the results have to be correct")
            {
                for (long i = 0; i < vec.size(); i++) {
                    TestType difference =
                        vec.array().square().sqrt()[i] - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }
        }

        WHEN("Using unary exp operation")
        {
            Vector dc(vec);
            Vector result(vecResult);

            auto expr = exp(dc);

            result.eval(expr);

            THEN("the results have to be correct")
            {
                for (long i = 0; i < vec.size(); i++) {
                    TestType difference = vec.array().exp()[i] - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }
        }

        WHEN("Using unary log operation")
        {
            Vector dc(vec);
            Vector result(vecResult);

            auto expr = log(square(dc));

            result.eval(expr);

            THEN("the results have to be correct")
            {
                for (long i = 0; i < vec.size(); i++) {
                    TestType difference =
                        vec.array().square().log()[i] - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }
        }
    }

    GIVEN("two random constructed Vector")
    {
        long size = 5;

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec(size);
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec2(size);
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> resultVec(size);

        randVec.setRandom();
        randVec2.setRandom();

        Vector dc(randVec);
        Vector dc2(randVec2);
        Vector result(resultVec);

        WHEN("Multiplying them together")
        {

            auto expr = dc * dc2;

            result.eval(expr);

            THEN("the results have to be correct")
            {
                for (long i = 0; i < randVec.size(); i++) {
                    TestType difference =
                        (randVec[i] * randVec2[i]) - result[static_cast<size_t>(i)];
                    auto diffAbs = quickvec::abs(difference);
                    REQUIRE(Approx(diffAbs) == 0.f);
                }
            }
        }
    }

    cudaDeviceReset();
}

TEST_CASE_TEMPLATE("Vector memory test simple", TestType, float, double, index_t)
{
    GIVEN("Eigen matrix")
    {
        long size = 256;

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec(size);
        randVec.setRandom();

        WHEN("Constructing and destructing Vector") { Vector dc(randVec); }

        THEN("Memory should not leak")
        {
            // test this with cuda-memcheck --leak-check full ./binary
            REQUIRE(true);
        }

        cudaDeviceReset();
    }
}
