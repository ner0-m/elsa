#include <catch2/catch.hpp>
#include <random>

#include "VolumeDescriptor.h"
#include "CudnnMemory.h"

using namespace elsa;

TEST_CASE("CudnnMemory", "[ml][cudnn]")
{
    IndexVector_t dims{{1, 3, 4}};
    VolumeDescriptor desc(dims);

    ml::detail::CudnnMemory<real_t> mem(desc);

    // Dimensions of Cudnn memory are filled until 4 are reached
    REQUIRE(mem.getDimensions().size() == 4);
    REQUIRE(mem.getDimensions()[0] == 1);
    REQUIRE(mem.getDimensions()[1] == 3);
    REQUIRE(mem.getDimensions()[2] == 4);
    REQUIRE(mem.getDimensions()[3] == 1);

    SECTION("Host memory")
    {
        // Check host memory
        mem.allocateHostMemory();
        REQUIRE(mem.hostMemory != nullptr);
        REQUIRE(mem.hostMemory->getMemoryHandle() != nullptr);
        REQUIRE(mem.hostMemory->getSize() == 1 * 3 * 4);
        REQUIRE(mem.hostMemory->getSizeInBytes() == 1 * 3 * 4 * sizeof(real_t));
        mem.hostMemory->fill(10.f);
        for (int i = 0; i < mem.hostMemory->getSize(); ++i) {
            REQUIRE(mem.hostMemory->getMemoryHandle()[i] == 10.f);
        }
    }

    SECTION("Device memory")
    {
        // Check host memory
        mem.allocateDeviceMemory();
        REQUIRE(mem.deviceMemory != nullptr);
        REQUIRE(mem.deviceMemory->getMemoryHandle() != nullptr);
        REQUIRE(mem.deviceMemory->getSize() == 1 * 3 * 4);
        REQUIRE(mem.deviceMemory->getSizeInBytes() == 1 * 3 * 4 * sizeof(real_t));
        mem.deviceMemory->fill(20.f);
        mem.copyToHost();
        for (int i = 0; i < mem.hostMemory->getSize(); ++i) {
            REQUIRE(mem.hostMemory->getMemoryHandle()[i] == 20.f);
        }
    }
}