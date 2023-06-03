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
     * allocated size ^
     *              s |                 __                         __
     *                |                /  \__    __               /  \__    __
     *                |         __    /      \__/  \       __    /      \__/  \
     *                |        /  \__/              \     /  \__/              \  -- REPEAT --
     *                |     __/                      \___/                      \___
     *                |    /
     *                |   /
     *                +------------------------------------------------------------------------>
     *                                                                                   time t
     * This kind of pattern works well for the cache resource, because the allocated blocks are of
     * equal size and, more importantly, the pattern repeats. Hence, even if this runs for 70
     * iterations, in this example only 4 allocations would have to be made from the upstream
     * allocator.
     */
    auto test = [&]() {
        static constexpr int ITERATIONS = 70;

        /* generate a contigous-vector, which uses the default resource */
        DataContainer<data_t> x = setup(2, 32);

        /* generate a contiguous-vector, which uses a different resource */
        DataContainer<data_t> y = setup(2, 32);

        for (int i = 0; i < ITERATIONS; i++) {
            /* note that this is not a realistic example, and it does not compute
            anything meaningful */

            /* allocation of two temporary datacontainers, one of which is then moved
            => one is freed immediately */
            auto z = x + 2 * y;

            /* three temporaries, one is then moved into a */
            auto a = 0.3 * z - 0.9 * y;

            /* another temporary, moved into b*/
            auto b = z + a;

            y = b;
        }
    };

    {
        /* first run is not timed, to make sure the first access cost/initialization
           of the CUDA API is out of the picture*/
        mr::hint::ScopedMR _scope{mr::UniversalResource::make()};
        test();
    }

    {
        mr::hint::ScopedMR _scope{mr::UniversalResource::make()};
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < 10; i++) {
            test();
        }
        auto stop = std::chrono::system_clock::now();
        Logger::get("Timing")->info(
            "Universal resource time: {} microseconds\n",
            static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()));
    }

    {
        mr::CacheResourceConfig config = mr::CacheResourceConfig::defaultConfig();
        /* sufficient for this example, typically higher in practice, or just leave as default*/
        config.setMaxCachedCount(8);
        mr::hint::ScopedMR _scope{mr::CacheResource::make(mr::UniversalResource::make(), config)};
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < 10; i++) {
            test();
        }
        auto stop = std::chrono::system_clock::now();
        Logger::get("Timing")->info(
            "Cache resource time: {} microseconds\n",
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
