/// Elsa example program: appropriately using a cache resource

#include "elsa.h"

#include <iostream>

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "elsaDefines.h"

#include <thrust/generate.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>

using namespace elsa;

/* Note: you will probably not see any benefit when comparing against a normal user space
   heap allocator, so this example only makes sense with CUDA. */
void exampleCacheResource()
{
    using data_t = elsa::complex<float>;

    auto setup = [&](size_t dim, size_t coeff) {
        std::random_device r;

        std::default_random_engine e(r());
        std::uniform_real_distribution<float> uniform_dist;

        auto size = elsa::IndexVector_t(dim);
        size.setConstant(coeff);

        auto desc = elsa::VolumeDescriptor(size);

        auto dc = elsa::DataContainer<elsa::complex<float>>(desc);
        thrust::generate(thrust::host, dc.begin(), dc.end(), [&]() {
            elsa::complex<float> c;
            c.real(uniform_dist(e));
            c.imag(uniform_dist(e));
            return c;
        });
        return dc;
    };

    /**
     * Allocation pattern:
     *
     * allocated size ^
     *              s |
     *                |        ___              ___
     *                |       /   \__          /   \__
     *                |      /       \        /       \     -- REPEAT --
     *                |     /         \      /         \
     *                |    /           \    /           \
     *              0 | __/             \__/             \__
     *                +---------------------------------------------------------->
     *                                                                     time t
     * This kind of pattern works well for the region resource, because the allocated blocks
     * are completely freed by the end of the loop. Hence, even if this
     * runs for 70 iterations, in this example only 1 allocation would have to be made
     * from the upstream allocator.
     */
    auto test = [&](auto input) {
        static constexpr int ITERATIONS = 70;

        for (int i = 0; i < ITERATIONS; i++) {
            /* note that this is not a realistic example, and it does not compute
            anything meaningful */

            /* allocation of two temporary datacontainers, one of which is then moved
            => one is freed immediately */
            auto z = 1.2 * input;

            /* three temporaries, one is then moved into a */
            auto a = z + 0.9;

            /* another temporary, moved into b*/
            auto b = -1.3 * a;

            auto c = -1.3 * a + b;

            input = c;
        }
    };

    auto input = setup(32, 32);

    {
        /* first run is not timed, to make sure the first access cost/initialization
           of the CUDA API is out of the picture*/
        mr::hint::ScopedMR _scope{mr::UniversalResource::make()};
        test(input);
    }

    input = setup(32, 32);

    {
        mr::hint::ScopedMR _scope{mr::UniversalResource::make()};
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < 10; i++) {
            test(input);
        }
        auto stop = std::chrono::system_clock::now();
        Logger::get("Timing")->info(
            "Universal resource time: {} microseconds\n",
            static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()));
    }

    input = setup(32, 32);

    {
        mr::RegionResourceConfig config = mr::RegionResourceConfig::defaultConfig();
        /* factor 32 * 32 accounts for the size of the data container
         * factor 7 is used here, although only 5 allocations are made,
         * to account for any additional overheads
         */
        config.setRegionSize(sizeof(elsa::complex<data_t>) * 32 * 32 * 7);
        mr::hint::ScopedMR _scope{mr::RegionResource::make(mr::UniversalResource::make(), config)};
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < 10; i++) {
            test(input);
        }
        auto stop = std::chrono::system_clock::now();
        Logger::get("Timing")->info(
            "Region resource time: {} microseconds\n",
            static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()));
    }
}

int main()
{
    try {
        exampleCacheResource();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
