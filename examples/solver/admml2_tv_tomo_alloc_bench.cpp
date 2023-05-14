/*
 * This examples solves the following problem:
 * \[
 * \min_x 0.5 * || A x - b ||_2^2 + \lambda || \nabla x ||_1
 * \]
 * using a specific version of Alternating Direction Method of Multipliers (ADMM).
 * The specific version is sometimes referred to as L2ADMM or ADMML2, where \f$f\f$ is
 * the least squares term, and only \f$g\f$ is required to have a known proximal operator.
 *
 * This is a quite easy solution to the TV problem.
 */
#include "elsa.h"

#include "storage/memory_resource/UniversalResource.h"
#include "storage/memory_resource/LoggingResource.h"
#include "storage/memory_resource/CacheResource.h"
#include "storage/memory_resource/RegionResource.h"
#include "storage/memory_resource/AllocationHint.h"

#include <iostream>

using namespace elsa;

void reconstruction(int s)
{
    // generate 2d phantom
    IndexVector_t size({{s, s}});

    auto phantom = phantoms::modifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // generate circular trajectory
    index_t numAngles{512}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    // VolumeDescriptor
    JosephsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    // simulate the sinogram
    Logger::get("Info")->info("Calculate sinogram");
    auto sinogram = projector.apply(phantom);

    auto A = FiniteDifferences<real_t>(volumeDescriptor);
    auto proxg = ProximalL1<real_t>{};
    auto tau = real_t{0.1};

    // solve the reconstruction problem
    ADMML2<real_t> admm(projector, sinogram, A, proxg, tau);

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations", noIterations);
    auto reco = admm.solve(noIterations);

    // write the reconstruction out
    io::write(reco, "reco_admml2_tv_tomo.pgm");
}

void speed_test(const char* resourceName, int size)
{
    Logger::get("Info")->info(
        "\n************************************************************************************"
        "\n\nRunning with: {}\n\n"
        "************************************************************************************\n",
        resourceName);
    auto start = std::chrono::system_clock::now();
    reconstruction(size);
    auto stop = std::chrono::system_clock::now();
    auto milliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    Logger::get("Info")->info("*** Duration: {}ms ***", milliSeconds);
}

constexpr bool resourceLoggingEnable = false;

template <typename R, typename... Ts>
static inline mr::MemoryResource make(Ts... args)
{
    if constexpr (resourceLoggingEnable) {
        return mr::LoggingResource<R>::make(args...);
    } else {
        return R::make(args...);
    }
}

int main(int argc, char** argv)
{
    auto usage = [&] { std::cout << "usage: " << argv[0] << " [--size size]" << std::endl; };

    auto argExits = [&](auto arg) {
        auto begin = argv;
        auto end = begin + argc;
        return std::find(begin, end, arg) != end;
    };

    auto getPos = [&](auto arg) {
        auto begin = argv;
        auto end = begin + argc;
        return std::distance(begin, std::find(begin, end, arg));
    };

    if (argExits("-h") || argExits("--help")) {
        usage();
        return 0;
    }

    const auto size = [&] {
        if (argExits("-s")) {
            auto pos = getPos("-s");

            if (pos >= argc) {
                usage();
                return 1;
            }
            return std::stoi(argv[pos + 1]);
        } else if (argExits("--size")) {
            auto pos = getPos("--size");

            if (pos >= argc) {
                usage();
                return 1;
            }
            return std::stoi(argv[pos + 1]);
        } else {
            return 128;
        }
    }();

    // the first CUDA allocation seems to always encur a huge penalty
    mr::MemoryResource baseline = mr::baselineInstance();
    void* firstAlloc = nullptr;

    try {
        firstAlloc = baseline->allocate(1, 1);

        {
            mr::hint::ScopedMR univ{make<mr::UniversalResource>()};
            speed_test("Universal Resource", size);
        }
        {
            mr::hint::ScopedMR pool{make<mr::PoolResource>(mr::UniversalResource::make())};
            speed_test("PoolResource (default)", size);
        }
        {
            mr::PoolResourceConfig config = mr::PoolResourceConfig::defaultConfig();
            config.setMaxBlockSize(static_cast<size_t>(1) << 32);
            config.setChunkSize(static_cast<size_t>(1) << 33);
            mr::hint::ScopedMR pool_large{
                make<mr::PoolResource>(mr::UniversalResource::make(), config)};
            speed_test("PoolResource (large chunks)", size);
        }
        {
            mr::hint::ScopedMR cache{make<mr::CacheResource>(mr::UniversalResource::make())};
            speed_test("CacheResource (default)", size);
        }
        {
            mr::CacheResourceConfig cacheConfig = mr::CacheResourceConfig::defaultConfig();
            cacheConfig.setMaxCachedCount(64);
            mr::hint::ScopedMR cache_unlimited{
                make<mr::CacheResource>(mr::UniversalResource::make(), cacheConfig)};
            speed_test("CacheResource (64 entries)", size);
        }
        {
            mr::hint::ScopedMR region{make<mr::RegionResource>(mr::UniversalResource::make())};
            speed_test("RegionResource", size);
        }
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
    baseline->deallocate(firstAlloc, 1, 1);
}
