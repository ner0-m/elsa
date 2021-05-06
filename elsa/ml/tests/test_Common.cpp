/**
 * @file test_common.cpp
 *
 * @brief Tests for common ml functionality
 *
 * @author David Tellenbach
 */

#include "doctest/doctest.h"
#include "DataContainer.h"
#include "VolumeDescriptor.h"
#include "Common.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE("Common")
{
    SECTION("LayerType")
    {
        REQUIRE(ml::detail::getEnumMemberAsString(ml::LayerType::Input) == "Input");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::LayerType::Dense) == "Dense");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::LayerType::Conv1D) == "Conv1D");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::LayerType::Conv2D) == "Conv2D");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::LayerType::Conv3D) == "Conv3D");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::LayerType::Sum) == "Sum");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::LayerType::Concatenate) == "Concatenate");
    }
    SECTION("PropagationKind")
    {
        REQUIRE(ml::detail::getEnumMemberAsString(ml::PropagationKind::Forward) == "Forward");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::PropagationKind::Backward) == "Backward");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::PropagationKind::Full) == "Full");
    }
    SECTION("MlBackend")
    {
        REQUIRE(ml::detail::getEnumMemberAsString(ml::MlBackend::Dnnl) == "Dnnl");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::MlBackend::Cudnn) == "Cudnn");
    }
    SECTION("Initializer")
    {
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::Zeros) == "Zeros");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::Ones) == "Ones");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::Normal) == "Normal");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::Uniform) == "Uniform");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::GlorotNormal) == "GlorotNormal");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::GlorotUniform)
                == "GlorotUniform");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::HeNormal) == "HeNormal");
        REQUIRE(ml::detail::getEnumMemberAsString(ml::Initializer::HeUniform) == "HeUniform");
    }
}
TEST_SUITE_END();
