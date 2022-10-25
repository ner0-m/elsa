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
    void apply([[maybe_unused]] const DataContainer<data_t>& x,
               DataContainer<data_t>& Ax) const override
    {
        Ax = 1;
    }

    void applyAdjoint([[maybe_unused]] const DataContainer<data_t>& y,
                      DataContainer<data_t>& Aty) const override
    {
        Aty = 3;
    }

    // Pull in apply and applyAdjoint with single argument from base class
    using LinearOperator<data_t>::apply;
    using LinearOperator<data_t>::applyAdjoint;

protected:
    MockOperator<data_t>* cloneImpl() const override
    {
        return new MockOperator<data_t>(this->getDomainDescriptor(), this->getRangeDescriptor());
    }
    bool isEqual(const LinearOperator<data_t>& other) const override
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherOp = downcast_safe<MockOperator<data_t>>(&other);
        return static_cast<bool>(otherOp);
    }
};

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("core");

// TEST_CASE_TEMPLATE("AdjointLinearOperator", data_t, float, double, complex<float>,
// complex<double>)
// {
//     IndexVector_t numDomain(2);
//     numDomain << 12, 23;
//     IndexVector_t numRange(2);
//     numRange << 34, 45;
//     VolumeDescriptor domainDesc(numDomain);
//     VolumeDescriptor rangeDesc(numRange);
//
//     MockOperator<data_t> op(domainDesc, rangeDesc);
//
//     auto adj = adjoint(op);
//
//     THEN("Domain and range descriptors are swapped")
//     {
//         CHECK_EQ(adj.getDomainDescriptor(), rangeDesc);
//         CHECK_EQ(adj.getRangeDescriptor(), domainDesc);
//     }
//
//     THEN("Calling apply() with a wrong descriptor throws")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(domainDesc);
//
//         CHECK_THROWS(adj.apply(dummy));
//     }
//
//     THEN("Calling applyAdjoint() with a wrong descriptor throws")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(rangeDesc);
//
//         CHECK_THROWS(adj.applyAdjoint(dummy));
//     }
//
//     THEN("Apply actually calls applyAdjoint")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(rangeDesc);
//
//         auto Ax = adj.apply(dummy);
//
//         for (auto elem : Ax) {
//             CHECK_EQ(elem, data_t(3));
//         }
//     }
//
//     THEN("applyAdjoint actually calls apply")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(domainDesc);
//
//         auto Aty = adj.applyAdjoint(dummy);
//
//         for (auto elem : Aty) {
//             CHECK_EQ(elem, data_t(1));
//         }
//     }
//
//     THEN("Equality comparison works")
//     {
//         auto adj2 = adjoint(op);
//         CHECK_EQ(adj, adj2);
//
//         CHECK_NE(adj, op);
//     }
//
//     THEN("Clone works")
//     {
//         auto adjClone = adj.clone();
//         CHECK_EQ(adj, *adjClone);
//     }
// }
//
// TEST_CASE_TEMPLATE("ScalarMulLinearOperator", data_t, float, double, complex<float>,
//                    complex<double>)
// {
//     IndexVector_t numDomain(2);
//     numDomain << 12, 23;
//     IndexVector_t numRange(2);
//     numRange << 34, 45;
//     VolumeDescriptor domainDesc(numDomain);
//     VolumeDescriptor rangeDesc(numRange);
//
//     MockOperator<data_t> op(domainDesc, rangeDesc);
//
//     auto comp = 5 * op;
//
//     THEN("Domain and range descriptors are the same")
//     {
//         CHECK_EQ(comp.getDomainDescriptor(), domainDesc);
//         CHECK_EQ(comp.getRangeDescriptor(), rangeDesc);
//     }
//
//     THEN("Calling apply() with a wrong descriptor throws")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(rangeDesc);
//
//         CHECK_THROWS(comp.apply(dummy));
//     }
//
//     THEN("Calling applyAdjoint() with a wrong descriptor throws")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(domainDesc);
//
//         CHECK_THROWS(comp.applyAdjoint(dummy));
//     }
//
//     THEN("Apply scales result correctly by scalar")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(domainDesc);
//
//         auto Ax = comp.apply(dummy);
//
//         for (auto elem : Ax) {
//             CHECK_EQ(elem, data_t(5));
//         }
//     }
//
//     THEN("applyAdjoint scales result correctly by scalar")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(rangeDesc);
//
//         auto Aty = comp.applyAdjoint(dummy);
//
//         for (auto elem : Aty) {
//             CHECK_EQ(elem, data_t(15));
//         }
//     }
//
//     THEN("Equality comparison works")
//     {
//         auto comp2 = data_t(5) * op;
//         CHECK_EQ(comp, comp2);
//         CHECK_NE(comp, op);
//
//         auto comp3 = data_t(6) * op;
//         CHECK_NE(comp, comp3);
//         CHECK_NE(comp, op);
//     }
//
//     THEN("Clone works")
//     {
//         auto compClone = comp.clone();
//         CHECK_EQ(comp, *compClone);
//     }
// }
//
// TEST_CASE_TEMPLATE("CompositeAddLinearOperator", data_t, float, double, complex<float>,
//                    complex<double>)
// {
//     IndexVector_t numDomain(2);
//     numDomain << 12, 23;
//     IndexVector_t numRange(2);
//     numRange << 34, 45;
//     VolumeDescriptor domainDesc(numDomain);
//     VolumeDescriptor rangeDesc(numRange);
//
//     MockOperator<data_t> op1(domainDesc, rangeDesc);
//     MockOperator<data_t> op2(domainDesc, rangeDesc);
//
//     auto comp = op1 + op2;
//
//     THEN("Domain and range descriptors are the same")
//     {
//         CHECK_EQ(comp.getDomainDescriptor(), domainDesc);
//         CHECK_EQ(comp.getRangeDescriptor(), rangeDesc);
//     }
//
//     THEN("Calling apply() with a wrong descriptor throws")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(rangeDesc);
//
//         CHECK_THROWS(comp.apply(dummy));
//     }
//
//     THEN("Calling applyAdjoint() with a wrong descriptor throws")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(domainDesc);
//
//         CHECK_THROWS(comp.applyAdjoint(dummy));
//     }
//
//     THEN("Apply adds the two operators together correctly")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(domainDesc);
//
//         auto Ax = comp.apply(dummy);
//
//         for (auto elem : Ax) {
//             CHECK_EQ(elem, data_t(1 + 1));
//         }
//     }
//
//     THEN("applyAdjoint adds the two operators together correctly")
//     {
//         // Input is not used, so leave it uninnitialized
//         DataContainer<data_t> dummy(rangeDesc);
//
//         auto Aty = comp.applyAdjoint(dummy);
//
//         for (auto elem : Aty) {
//             CHECK_EQ(elem, data_t(3 + 3));
//         }
//     }
//
//     THEN("Equality comparison works")
//     {
//         auto comp2 = op1 + op2;
//         CHECK_EQ(comp, comp2);
//         CHECK_NE(comp2, op1);
//         CHECK_NE(comp2, op2);
//
//         auto comp3 = op2 + op1;
//         CHECK_EQ(comp, comp3);
//         CHECK_NE(comp3, op1);
//         CHECK_NE(comp3, op2);
//     }
//
//     THEN("Clone works")
//     {
//         auto compClone = comp.clone();
//         CHECK_EQ(comp, *compClone);
//     }
// }

// TEST_CASE_TEMPLATE("CompositeMulLinearOperator", data_t, float, double, complex<float>,
//                    complex<double>)
TEST_CASE_TEMPLATE("CompositeMulLinearOperator", data_t, float)
{
    IndexVector_t num1(2);
    num1 << 12, 23;
    IndexVector_t num2(2);
    num2 << 34, 45;
    IndexVector_t num3(2);
    num3 << 59, 24;

    VolumeDescriptor descDomain(num1);
    VolumeDescriptor descMiddle(num2);
    VolumeDescriptor descRange(num3);

    MockOperator<data_t> op1(descDomain, descMiddle);
    MockOperator<data_t> op2(descMiddle, descRange);

    auto comp = op2 * op1;

    THEN("Domain and range descriptors are the same")
    {
        CHECK_EQ(comp.getDomainDescriptor(), descDomain);
        CHECK_EQ(comp.getRangeDescriptor(), descRange);
    }

    THEN("Calling apply() with a wrong descriptor throws")
    {
        // Input is not used, so leave it uninnitialized
        DataContainer<data_t> dummy(descMiddle);

        CHECK_THROWS(comp.apply(dummy));
    }

    THEN("Calling applyAdjoint() with a wrong descriptor throws")
    {
        // Input is not used, so leave it uninnitialized
        DataContainer<data_t> dummy(descMiddle);

        CHECK_THROWS(comp.applyAdjoint(dummy));
    }

    THEN("Apply multiplies the two operators together correctly")
    {
        // Input is not used, so leave it uninnitialized
        DataContainer<data_t> dummy(descDomain);

        auto Ax = comp.apply(dummy);

        for (auto elem : Ax) {
            CHECK_EQ(elem, data_t(1 * 1));
        }
    }

    THEN("applyAdjoint adds the two operators together correctly")
    {
        // Input is not used, so leave it uninnitialized
        DataContainer<data_t> dummy(descRange);

        auto Aty = comp.applyAdjoint(dummy);

        for (auto elem : Aty) {
            CHECK_EQ(elem, data_t(3));
        }
    }

    THEN("Equality comparison works")
    {
        auto comp2 = op2 * op1;
        CHECK_EQ(comp, comp2);
        CHECK_NE(comp2, op1);
        CHECK_NE(comp2, op2);

        auto comp3 = op1 * op2;
        CHECK_NE(comp, comp3);
        CHECK_NE(comp3, op1);
        CHECK_NE(comp3, op2);
    }

    THEN("Clone works")
    {
        auto compClone = comp.clone();
        CHECK_EQ(comp, *compClone);
    }
}

TEST_SUITE_END();
