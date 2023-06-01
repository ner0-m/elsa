/**
 * @file test_LinearOperator.cpp
 *
 * @brief Tests for LinearOperator class
 *
 * @author Tobias Lasser - main code
 * @author David Frank - composite tests
 * @author Nikola Dinev - fixes
 */

#include "doctest/doctest.h"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

// mock operator, which outputs 1 for apply and 2 for applyAdjoint
template <typename data_t = real_t>
class MockOperator : public LinearOperator<data_t>
{
public:
    MockOperator(const DataDescriptor& domain, const DataDescriptor& range)
        : LinearOperator<data_t>(domain, range)
    {
    }

protected:
    void applyImpl([[maybe_unused]] const DataContainer<data_t>& x,
                   DataContainer<data_t>& Ax) const override
    {
        Ax = 1;
    }

    void applyAdjointImpl([[maybe_unused]] const DataContainer<data_t>& y,
                          DataContainer<data_t>& Aty) const override
    {
        Aty = 3;
    }

protected:
    MockOperator<data_t>* cloneImpl() const override
    {
        return new MockOperator<data_t>(this->getDomainDescriptor(), this->getRangeDescriptor());
    }

    bool isEqual(const LinearOperator<data_t>& other) const override
    {
        auto tmp = downcast_safe<MockOperator<data_t>>(&other);
        if (!tmp)
            return false;

        return this->getDomainDescriptor() == tmp->getDomainDescriptor()
               && this->getRangeDescriptor() == tmp->getRangeDescriptor();
    }
};

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("LinearOperator: Testing construction", TestType, float, double, complex<float>,
                   complex<double>)
{
    GIVEN("DataDescriptors")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 31, 23;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);

        WHEN("instantiating a LinearOperator")
        {
            LinearOperator<TestType> linOp(ddDomain, ddRange);

            THEN("the DataDescriptors are as expected")
            {
                REQUIRE_EQ(linOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(linOp.getRangeDescriptor(), ddRange);
            }

            THEN("the apply* operations throw")
            {
                DataContainer<TestType> dc(ddDomain);
                REQUIRE_THROWS_AS(linOp.apply(dc), LogicError);
                REQUIRE_THROWS_AS(linOp.applyAdjoint(dc), LogicError);
            }

            THEN("copies are good")
            {
                auto newOp = linOp;
                REQUIRE_EQ(newOp, linOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("LinearOperator: Testing clone()", TestType, float, double, complex<float>,
                   complex<double>)
{
    GIVEN("a LinearOperator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 23, 45, 67;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 78, 90;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        LinearOperator<TestType> linOp(ddDomain, ddRange);

        WHEN("cloning the LinearOperator")
        {
            auto linOpClone = linOp.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(linOpClone.get(), &linOp);
                REQUIRE_EQ(*linOpClone, linOp);
            }

            THEN("copies are also identical")
            {
                auto newOp = *linOpClone;
                REQUIRE_EQ(newOp, linOp);
            }
        }
    }

    GIVEN("a scalar multiplicative composite LinearOperator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 50, 41, 22;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 4, 88;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        LinearOperator<TestType> linOp(ddDomain, ddRange);
        TestType scalar = 42;

        LinearOperator<TestType> scalarMultLinOp = scalar * linOp;

        WHEN("cloning the LinearOperator")
        {
            auto linOpClone = scalarMultLinOp.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(linOpClone.get(), &scalarMultLinOp);
                REQUIRE_EQ(*linOpClone, scalarMultLinOp);
            }

            THEN("copies are also identical")
            {
                auto newOp = *linOpClone;
                REQUIRE_EQ(newOp, scalarMultLinOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("LinearOperator: Testing a leaf LinearOperator", TestType, float, double,
                   complex<float>, complex<double>)
{
    GIVEN("a non-adjoint leaf linear operator")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 12, 23;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 34, 45;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        MockOperator<TestType> mockOp(ddDomain, ddRange);

        auto leafOp = leaf(mockOp);

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE_EQ(leafOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(leafOp.getRangeDescriptor(), ddRange);
            }
        }

        WHEN("given data")
        {
            DataContainer<TestType> dcDomain(ddDomain);
            DataContainer<TestType> dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = leafOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE_EQ(resultApply[i], static_cast<TestType>(1));

                auto resultApplyAdjoint = leafOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<TestType>(3));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(leafOp.apply(dcRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(leafOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = leafOp;
            auto assignedOp = leaf(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, leafOp);
                REQUIRE_EQ(assignedOp, leaf(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }

    GIVEN("an adjoint linear operator")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 12, 23;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 34, 45;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        MockOperator<TestType> mockOp(ddDomain, ddRange);

        auto adjointOp = adjoint(mockOp);

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE_EQ(adjointOp.getDomainDescriptor(), ddRange);
                REQUIRE_EQ(adjointOp.getRangeDescriptor(), ddDomain);
            }
        }

        WHEN("given data")
        {
            DataContainer<TestType> dcDomain(ddDomain);
            DataContainer<TestType> dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = adjointOp.apply(dcRange);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE_EQ(resultApply[i], static_cast<TestType>(3));

                auto resultApplyAdjoint = adjointOp.applyAdjoint(dcDomain);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<TestType>(1));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(adjointOp.apply(dcDomain), InvalidArgumentError);
                REQUIRE_THROWS_AS(adjointOp.applyAdjoint(dcRange), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = adjointOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, adjointOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("LinearOperator: Testing composite LinearOperator", TestType, float, double,
                   complex<float>, complex<double>)
{
    GIVEN("an additive composite linear operator")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 45, 67;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 13, 48;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);

        MockOperator<TestType> op1(ddDomain, ddRange);
        MockOperator<TestType> op2(ddDomain, ddRange);

        auto addOp = op1 + op2;

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE_EQ(addOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(addOp.getRangeDescriptor(), ddRange);
            }
        }

        WHEN("given data")
        {
            DataContainer<TestType> dcDomain(ddDomain);
            DataContainer<TestType> dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = addOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE_EQ(resultApply[i], static_cast<TestType>(2));

                auto resultApplyAdjoint = addOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<TestType>(6));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(addOp.apply(dcRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(addOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = addOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, addOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }

    GIVEN("a multiplicative composite linear operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 47, 69;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 15, 28;
        IndexVector_t numCoeff3(4);
        numCoeff3 << 7, 30, 83, 13;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddMiddle(numCoeff2);
        VolumeDescriptor ddRange(numCoeff3);

        MockOperator<TestType> op1(ddDomain, ddMiddle);
        MockOperator<TestType> op2(ddMiddle, ddRange);

        auto multOp = op2 * op1;

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE_EQ(multOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(multOp.getRangeDescriptor(), ddRange);
            }
        }

        WHEN("given data")
        {
            DataContainer<TestType> dcDomain(ddDomain);
            DataContainer<TestType> dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = multOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE_EQ(resultApply[i], static_cast<TestType>(1));

                auto resultApplyAdjoint = multOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<TestType>(3));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(multOp.apply(dcRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(multOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = multOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, multOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }

    GIVEN("a scalar multiplicative composite linear operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 47, 69;
        IndexVector_t otherNumCoeff(2);
        otherNumCoeff << 15, 28;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(otherNumCoeff);

        MockOperator op(ddDomain, ddRange);
        real_t scalar = 8;

        auto scalarMultOp = scalar * op;

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE_EQ(scalarMultOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(scalarMultOp.getRangeDescriptor(), ddRange);
            }
        }

        WHEN("given data")
        {
            DataContainer dcDomain(ddDomain);
            DataContainer dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = scalarMultOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE_EQ(resultApply[i], static_cast<real_t>(8));

                auto resultApplyAdjoint = scalarMultOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<real_t>(24));
            }

            THEN("the apply operations account for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(scalarMultOp.apply(dcRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(scalarMultOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            const auto& newOp = scalarMultOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, scalarMultOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }

    GIVEN("a complex composite with multiple leafs and levels")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 38;
        IndexVector_t numCoeff2(1);
        numCoeff2 << 16;
        IndexVector_t numCoeff3(3);
        numCoeff3 << 17, 38, 15;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        VolumeDescriptor ddFinalRange(numCoeff3);

        MockOperator<TestType> op1(ddDomain, ddRange);
        MockOperator<TestType> op2(ddFinalRange, ddRange);
        MockOperator<TestType> op3(ddRange, ddFinalRange);

        auto compositeOp = (op3 + adjoint(op2)) * op1;

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE_EQ(compositeOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(compositeOp.getRangeDescriptor(), ddFinalRange);
            }
        }

        WHEN("given data")
        {
            DataContainer<TestType> dcDomain(ddDomain);
            DataContainer<TestType> dcFinalRange(ddFinalRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = compositeOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE_EQ(resultApply[i], static_cast<TestType>(4));

                auto resultApplyAdjoint = compositeOp.applyAdjoint(dcFinalRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<TestType>(3));
            }

            THEN("the apply operations expect appropriately sized containers")
            {
                REQUIRE_THROWS_AS(compositeOp.apply(dcFinalRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(compositeOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = compositeOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, compositeOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }
}

template <typename data_t = real_t>
class UniqueMockOperator : public LinearOperator<data_t>
{
public:
    UniqueMockOperator(index_t id, const DataDescriptor& domain, const DataDescriptor& range)
        : LinearOperator<data_t>(domain, range), id_(id)
    {
    }

protected:
    void applyImpl([[maybe_unused]] const DataContainer<data_t>& x,
                   DataContainer<data_t>& Ax) const override
    {
        Ax = id_;
    }

    void applyAdjointImpl([[maybe_unused]] const DataContainer<data_t>& y,
                          DataContainer<data_t>& Aty) const override
    {
        Aty = 1. / id_;
    }

protected:
    UniqueMockOperator<data_t>* cloneImpl() const override
    {
        return new UniqueMockOperator<data_t>(id_, this->getDomainDescriptor(),
                                              this->getRangeDescriptor());
    }

    bool isEqual(const LinearOperator<data_t>& other) const override
    {
        auto tmp = downcast_safe<UniqueMockOperator<data_t>>(&other);
        if (!tmp)
            return false;

        return id_ == tmp->id_ && this->getDomainDescriptor() == tmp->getDomainDescriptor()
               && this->getRangeDescriptor() == tmp->getRangeDescriptor();
    }

private:
    index_t id_;
};

TEST_CASE_TEMPLATE("LinearOperator: Testing OperatorList", data_t, float, double)
{
    IndexVector_t numCoeff(2);
    numCoeff << 47, 11;
    IndexVector_t numCoeff2(2);
    numCoeff2 << 31, 23;
    VolumeDescriptor domain(numCoeff);
    VolumeDescriptor range(numCoeff2);

    UniqueMockOperator<data_t> op1(1, domain, range);
    UniqueMockOperator<data_t> op2(2, domain, range);
    UniqueMockOperator<data_t> op3(3, domain, range);

    GIVEN("A list of three operators")
    {
        LinearOperatorList<data_t> oplist(op1, op2, op3);

        THEN("Copy works")
        {
            auto copy = oplist;

            CHECK_EQ(oplist.getSize(), copy.getSize());
            CHECK_EQ(oplist[0], copy[0]);
            CHECK_EQ(oplist[1], copy[1]);
            CHECK_EQ(oplist[2], copy[2]);
        }

        THEN("Move works")
        {
            auto move = std::move(oplist);

            CHECK_EQ(move.getSize(), 3);
            CHECK_EQ(move[0], op1);
            CHECK_EQ(move[1], op2);
            CHECK_EQ(move[2], op3);
        }

        THEN("Iterators work")
        {
            auto begin = oplist.begin();
            auto end = oplist.end();

            CHECK_EQ(*begin, op1);
            ++begin;
            CHECK_EQ(*begin, op2);
            ++begin;
            CHECK_EQ(*begin, op3);
            ++begin;
            CHECK_EQ(begin, end);
        }
    }
}

TEST_CASE_TEMPLATE("LinearOperator: Testing const OperatorList", data_t, float, double)
{
    IndexVector_t numCoeff(2);
    numCoeff << 47, 11;
    IndexVector_t numCoeff2(2);
    numCoeff2 << 31, 23;
    VolumeDescriptor domain(numCoeff);
    VolumeDescriptor range(numCoeff2);

    UniqueMockOperator<data_t> op1(1, domain, range);
    UniqueMockOperator<data_t> op2(2, domain, range);
    UniqueMockOperator<data_t> op3(3, domain, range);

    GIVEN("A list of three operators")
    {
        const LinearOperatorList<data_t> oplist(op1, op2, op3);

        THEN("Copy works")
        {
            auto copy = oplist;

            CHECK_EQ(oplist.getSize(), copy.getSize());
            CHECK_EQ(oplist[0], copy[0]);
            CHECK_EQ(oplist[1], copy[1]);
            CHECK_EQ(oplist[2], copy[2]);
        }

        THEN("Iteration works")
        {

            auto begin = oplist.begin();
            auto end = oplist.end();

            CHECK_EQ(*begin, op1);
            ++begin;
            CHECK_EQ(*begin, op2);
            ++begin;
            CHECK_EQ(*begin, op3);
            ++begin;
            CHECK_EQ(begin, end);
        }
    }
}

TEST_SUITE_END();
