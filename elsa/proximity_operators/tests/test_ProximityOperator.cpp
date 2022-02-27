/**
* @file test_ProximityOperator.cpp
*
* @brief Tests for the ProximityOperator class
*
* @author Andi Braimllari
*/

#include "SoftThresholding.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximity_operators");

TEST_CASE_TEMPLATE("ProximityOperator: Testing valuesToThresholds", data_t, float, double)
{
   GIVEN("a 1D DataDescriptor")
   {
       IndexVector_t numCoeff(1);
       numCoeff << 4;
       VolumeDescriptor volDescr(numCoeff);

       WHEN("using the static valuesToThresholds method")
       {
           DataContainer<data_t> dataCont(volDescr);
           dataCont[0] = 7;
           dataCont[1] = 3;
           dataCont[2] = 4;
           dataCont[3] = 5;

           std::vector<geometry::Threshold<data_t>> res =
               ProximityOperator<data_t>::valuesToThresholds(dataCont);

           THEN("the primitive values are converted to Threshold objects contain that respective "
                "value")
           {
               std::vector<geometry::Threshold<data_t>> expectedRes;
               expectedRes.push_back(geometry::Threshold<data_t>{7});
               expectedRes.push_back(geometry::Threshold<data_t>{3});
               expectedRes.push_back(geometry::Threshold<data_t>{4});
               expectedRes.push_back(geometry::Threshold<data_t>{5});

               for (index_t i = 0; i < dataCont.getSize(); ++i) {
                   REQUIRE_UNARY(dataCont[i] == expectedRes[i]);
               }
           }
       }
   }

   GIVEN("a 2D DataDescriptor")
   {
       IndexVector_t numCoeff(2);
       numCoeff << 3, 2;
       VolumeDescriptor volDescr(numCoeff);

       WHEN("using the static valuesToThresholds method")
       {
           DataContainer<data_t> dataCont(volDescr);
           dataCont[0] = 9;
           dataCont[1] = 8;
           dataCont[2] = 5;
           dataCont[3] = 1;
           dataCont[4] = 2;
           dataCont[5] = 2;

           std::vector<geometry::Threshold<data_t>> res =
               ProximityOperator<data_t>::valuesToThresholds(dataCont);

           THEN("the primitive values are converted to Threshold objects contain that "
                "respective value")
           {
               std::vector<geometry::Threshold<data_t>> expectedRes;
               expectedRes.push_back(geometry::Threshold<data_t>{9});
               expectedRes.push_back(geometry::Threshold<data_t>{8});
               expectedRes.push_back(geometry::Threshold<data_t>{5});
               expectedRes.push_back(geometry::Threshold<data_t>{1});
               expectedRes.push_back(geometry::Threshold<data_t>{2});
               expectedRes.push_back(geometry::Threshold<data_t>{2});

               for (index_t i = 0; i < dataCont.getSize(); ++i) {
                   REQUIRE_UNARY(dataCont[i] == expectedRes[i]);
               }
           }
       }
   }
}

TEST_SUITE_END();
