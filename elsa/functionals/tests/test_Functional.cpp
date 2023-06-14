#include <doctest/doctest.h>
#include <unistd.h>

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "L1Norm.h"
#include "Scaling.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "Functional.h"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

template <class data_t>
class MockFunctional1 : public Functional<data_t>
{
public:
    MockFunctional1(const DataDescriptor& desc) : Functional<data_t>(desc) {}

    data_t evaluateImpl(const DataContainer<data_t>& Rx) override { return 1; }

    void getGradientImpl(const DataContainer<data_t>&, DataContainer<data_t>& out) override
    {
        out = 1;
    }

    LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override
    {
        return Scaling<data_t>(this->getDomainDescriptor(), 1);
    }

    MockFunctional1<data_t>* cloneImpl() const override
    {
        return new MockFunctional1<data_t>(this->getDomainDescriptor());
    }

    bool isEqual(const Functional<data_t>& other) const override
    {
        return Functional<data_t>::isEqual(other) && is<MockFunctional1<data_t>>(other);
    }
};

template <class data_t>
class MockFunctional2 : public Functional<data_t>
{
public:
    MockFunctional2(const DataDescriptor& desc) : Functional<data_t>(desc) {}

    data_t evaluateImpl(const DataContainer<data_t>& Rx) override { return 2; }

    void getGradientImpl(const DataContainer<data_t>&, DataContainer<data_t>& out) override
    {
        out = 2;
    }

    LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override
    {
        return Scaling<data_t>(this->getDomainDescriptor(), 2);
    }

    MockFunctional2<data_t>* cloneImpl() const override
    {
        return new MockFunctional2<data_t>(this->getDomainDescriptor());
    }

    bool isEqual(const Functional<data_t>& other) const override
    {
        return Functional<data_t>::isEqual(other) && is<MockFunctional2<data_t>>(other);
    }
};

template <class data_t>
class MockFunctional3 : public Functional<data_t>
{
public:
    MockFunctional3(const DataDescriptor& desc) : Functional<data_t>(desc) {}

    data_t evaluateImpl(const DataContainer<data_t>& Rx) override { return 3; }

    void getGradientImpl(const DataContainer<data_t>&, DataContainer<data_t>& out) override
    {
        out = 3;
    }

    LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override
    {
        return Scaling<data_t>(this->getDomainDescriptor(), 3);
    }

    MockFunctional3<data_t>* cloneImpl() const override
    {
        return new MockFunctional3<data_t>(this->getDomainDescriptor());
    }

    bool isEqual(const Functional<data_t>& other) const override
    {
        return Functional<data_t>::isEqual(other) && is<MockFunctional3<data_t>>(other);
    }
};

TEST_CASE_TEMPLATE("FunctionalSum: Testing", data_t, float, double, complex<float>, complex<double>)
{
    VolumeDescriptor desc({4, 6});

    MockFunctional1<data_t> fn1(desc);
    MockFunctional2<data_t> fn2(desc);
    MockFunctional3<data_t> fn3(desc);

    GIVEN("A sum of two functionals")
    {
        auto sum = fn1 + fn2;

        DataContainer<data_t> x(desc);
        x = 1234;

        CHECK_EQ(sum.evaluate(x), 3);

        DataContainer<data_t> expectedGrad(desc);
        expectedGrad = 3;

        auto grad = sum.getGradient(x);
        CHECK_UNARY(isApprox(grad, expectedGrad));

        THEN("Clone behaves as original")
        {
            auto clone = sum.clone();

            CHECK_EQ(clone->evaluate(x), 3);
            CHECK_UNARY(isApprox(clone->getGradient(x), expectedGrad));

            CHECK_EQ(*clone, sum);
            CHECK_NE(fn1, sum);
        }
    }

    GIVEN("A sum of three functionals")
    {
        auto sum = fn1 + fn2 + fn3;

        DataContainer<data_t> x(desc);
        x = 1234;

        CHECK_EQ(sum.evaluate(x), 6);

        DataContainer<data_t> expectedGrad(desc);
        expectedGrad = 6;

        auto grad = sum.getGradient(x);
        CHECK_UNARY(isApprox(grad, expectedGrad));

        THEN("Clone behaves as original")
        {
            auto clone = sum.clone();

            CHECK_EQ(clone->evaluate(x), 6);
            CHECK_UNARY(isApprox(clone->getGradient(x), expectedGrad));

            CHECK_EQ(*clone, sum);
            CHECK_NE(fn1, sum);
        }
    }
}

TEST_CASE_TEMPLATE("FunctionalScalarMul: Testing", data_t, float, double, complex<float>,
                   complex<double>)
{
    VolumeDescriptor desc({4, 6});

    MockFunctional1<data_t> fn1(desc);

    GIVEN("Given scalar * functional")
    {
        auto sum = 0.5 * fn1;

        DataContainer<data_t> x(desc);
        x = 1234;

        CHECK_EQ(sum.evaluate(x), 0.5);

        DataContainer<data_t> expectedGrad(desc);
        expectedGrad = 0.5;

        auto grad = sum.getGradient(x);
        CHECK_UNARY(isApprox(grad, expectedGrad));

        THEN("Clone behaves as original")
        {
            auto clone = sum.clone();

            CHECK_EQ(clone->evaluate(x), 0.5);
            CHECK_UNARY(isApprox(clone->getGradient(x), expectedGrad));

            CHECK_EQ(*clone, sum);
            CHECK_NE(fn1, sum);
        }
    }

    GIVEN("Given functional * scalar")
    {
        auto sum = fn1 * 0.5;

        DataContainer<data_t> x(desc);
        x = 1234;

        CHECK_EQ(sum.evaluate(x), 0.5);

        DataContainer<data_t> expectedGrad(desc);
        expectedGrad = 0.5;

        auto grad = sum.getGradient(x);
        CHECK_UNARY(isApprox(grad, expectedGrad));

        THEN("Clone behaves as original")
        {
            auto clone = sum.clone();

            CHECK_EQ(clone->evaluate(x), 0.5);
            CHECK_UNARY(isApprox(clone->getGradient(x), expectedGrad));

            CHECK_EQ(*clone, sum);
            CHECK_NE(fn1, sum);
        }
    }
}
