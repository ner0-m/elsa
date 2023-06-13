/**
* @file test_ZeroOperator.cpp
*
* @brief Tests for ZeroOperator
*
* @author Shen Hu - initial code
*/

#include "doctest/doctest.h"
#include "ZeroOperator.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("ZeroOperator: Testing construction, cloning and comparison", data_t, float, double)
{
   GIVEN("a domain and a range descriptor")
   {
       IndexVector_t numCoeffD(3);
       numCoeffD << 2, 3, 3;
       VolumeDescriptor dd(numCoeffD);

       IndexVector_t numCoeffR(2);
       numCoeffR << 4, 5;
       VolumeDescriptor rd(numCoeffR);

       WHEN("instantiating a ZeroOperator")
       {
           ZeroOperator<data_t> op(dd, rd);

           THEN("descriptors are as expected")
           {
               REQUIRE(op.getDomainDescriptor() == dd);
               REQUIRE(op.getRangeDescriptor() == rd);
           }
       }

       WHEN("cloning a ZeroOperator operator")
       {
           ZeroOperator<data_t> op(dd, rd);
           auto opClone = op.clone();

           THEN("everything matches")
           {
               REQUIRE(opClone.get() != &op);
               REQUIRE(*opClone == op);
           }
       }
   }
}

TEST_CASE_TEMPLATE("ZeroOperator: Test apply and applyAdjoint", data_t, float, double, complex<float>,
                  complex<double>)
{
   GIVEN("some data and an appropriate ZeroOperator")
   {
       IndexVector_t numCoeffD(3);
       numCoeffD << 2, 3, 3;
       VolumeDescriptor dd(numCoeffD);

       IndexVector_t numCoeffR(2);
       numCoeffR << 4, 5;
       VolumeDescriptor rd(numCoeffR);

       Vector_t<data_t> domain_data_raw(dd.getNumberOfCoefficients());
       domain_data_raw.setRandom();
       DataContainer<data_t> domain_data(dd, domain_data_raw);

       Vector_t<data_t> range_data_raw(rd.getNumberOfCoefficients());
       range_data_raw.setRandom();
       DataContainer<data_t> range_data(rd, range_data_raw);

       ZeroOperator<data_t> op(dd, rd);

       WHEN("applying")
       {
           auto output = op.apply(domain_data);

           THEN("the result is as expected")
           {
               REQUIRE(output.getSize() == op.getRangeDescriptor().getNumberOfCoefficients());
               REQUIRE(output.sum() == 0);
           }
       }

       WHEN("applyAdjoint-ing")
       {
           auto output = op.applyAdjoint(range_data);

           THEN("the results is as expected")
           {
               REQUIRE(output.getSize() == op.getDomainDescriptor().getNumberOfCoefficients());
               REQUIRE(output.sum() == 0);
           }
       }
   }
}
TEST_SUITE_END();
