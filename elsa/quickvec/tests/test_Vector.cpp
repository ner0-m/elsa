#include "doctest/doctest.h"

#include "Vector.cuh"
#include "Eigen/Dense"
#include "Defines.cuh"
#include "Complex.h"

#include <thrust/complex.h>

using namespace quickvec;

using doctest::Approx;

static float margin = 0.000001f;

TEST_CASE_TEMPLATE("Vector, real() and imag()", TestType, thrust::complex<float>,
                   thrust::complex<double>)
{
    using data_t = TestType;
    using inner_t = GetFloatingPointType_t<data_t>;

    constexpr index_t size = 4;

    GIVEN("A complex vector")
    {
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> vec(size);

        vec[0] = data_t(1.f, 1.f);
        vec[1] = data_t(2.3f, 2.3f);
        vec[2] = data_t(2.2f, 2.2f);
        vec[3] = data_t(-3.4f, -3.4f);

        Vector<data_t> x(vec);

        WHEN("getting the real part of the complex vector")
        {
            Vector<inner_t> y(size);
            y.eval(quickvec::real(x));

            THEN("It's extracted correctly")
            {
                for (int i = 0; i < x.size(); ++i) {
                    REQUIRE_EQ(y[i], x[i].real());
                }
            }
        }

        WHEN("getting the imaginary part of the complex vector")
        {
            Vector<inner_t> y(size);
            y.eval(quickvec::imag(x));

            THEN("It's extracted correctly")
            {
                for (int i = 0; i < x.size(); ++i) {
                    REQUIRE_EQ(y[i], x[i].imag());
                }
            }
        }
    }

    GIVEN("A real vector")
    {
        Eigen::Matrix<inner_t, Eigen::Dynamic, 1> vec(size);

        vec[0] = inner_t(1.f);
        vec[1] = inner_t(2.3f);
        vec[2] = inner_t(2.2f);
        vec[3] = inner_t(-3.4f);

        Vector<inner_t> x(vec);

        WHEN("getting the real part of the real vector")
        {
            Vector<inner_t> y(size);
            y.eval(quickvec::real(x));

            THEN("It's extracted correctly")
            {
                for (int i = 0; i < x.size(); ++i) {
                    REQUIRE_EQ(y[i], std::real(x[i]));
                }
            }
        }

        WHEN("getting the imaginary part of the real vector")
        {
            Vector<inner_t> y(size);
            y.eval(quickvec::imag(x));

            THEN("It's extracted correctly")
            {
                for (int i = 0; i < x.size(); ++i) {
                    REQUIRE_EQ(y[i], std::imag(x[i]));
                    REQUIRE_EQ(y[i], 0.f);
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("Vector, dot product", TestType, float, double, thrust::complex<float>,
                   thrust::complex<double>)
{
    using data_t = TestType;
    using inner_t = GetFloatingPointType_t<data_t>;

    GIVEN("Two vectors")
    {

        Eigen::Matrix<data_t, Eigen::Dynamic, 1> v1(4);
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> v2(4);

        if constexpr (isComplex<data_t>) {
            v1[0] = data_t{1, 2};
            v1[1] = data_t{2.3f, 4};
            v1[2] = data_t{2.2f, -3};
            v1[3] = data_t{-3.4f, 3};

            v2[0] = data_t{2, 1};
            v2[1] = data_t{3.3f, 29};
            v2[2] = data_t{3.2f, 2};
            v2[3] = data_t{4.4f, 19};
        } else {
            v1[0] = data_t{1};
            v1[1] = data_t{2.3f};
            v1[2] = data_t{2.2f};
            v1[3] = data_t{-3.4f};

            v2[0] = data_t{2};
            v2[1] = data_t{3.3f};
            v2[2] = data_t{3.2f};
            v2[3] = data_t{4.4f};
        }

        Vector<data_t> x(v1);
        Vector<data_t> y(v2);

        WHEN("Computing the dot product")
        {
            data_t mydot = x.dot(y);
            if constexpr (isComplex<data_t>) {
                auto otherdot = v1.template cast<std::complex<inner_t>>().dot(
                    v2.template cast<std::complex<inner_t>>());

                THEN("It's computed correctly")
                {
                    CHECK(Approx(mydot.real()) == std::real(otherdot));
                    CHECK(Approx(mydot.imag()) == std::imag(otherdot));
                }
            } else {
                auto otherdot = v1.dot(v2);

                THEN("It's computed correctly")
                {
                    CHECK(Approx(std::real(mydot)) == std::real(otherdot));
                    CHECK(Approx(std::imag(mydot)) == std::imag(otherdot));
                }
            }
        }
    }
}

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

            THEN("Max reduction works as expected")
            {
                if constexpr (isComplex<TestType>) {
                    REQUIRE_THROWS(dc.maxElement());
                } else {
                    REQUIRE(Approx(dc.maxElement()) == vec.maxCoeff());
                }
            }
            THEN("Min reduction works as expected")
            {
                if constexpr (isComplex<TestType>) {
                    REQUIRE_THROWS(dc.minElement());
                } else {
                    REQUIRE(Approx(dc.minElement()) == vec.minCoeff());
                }
            }

            THEN("L2-norm works as expected")
            {
                // using thrust directly in norm is unsafe with eigen
                auto castVec = vec.template cast<std::complex<double>>();
                REQUIRE(Approx(dc.squaredl2Norm()) == castVec.squaredNorm());
                REQUIRE(Approx(dc.l2Norm()) == castVec.norm());
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

        WHEN("Constructing and destructing Vector")
        {
            Vector dc(randVec);
        }

        THEN("Memory should not leak")
        {
            // test this with cuda-memcheck --leak-check full ./binary
            REQUIRE(true);
        }

        cudaDeviceReset();
    }
}
