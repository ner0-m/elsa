#include <catch2/catch.hpp>
#include <random>

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "CudnnDense.h"
#include "CudnnDataContainerInterface.h"

using namespace elsa;
using namespace elsa::ml::detail;
using namespace doctest;

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_SUITE_BEGIN("ml-cudnn");

TEST_SUITE_END();
