#include "ExpressionPredicates.h"
#include "elsa.h"

#include <iostream>
#include "IS_ADMML2.h"

using namespace elsa;

template <typename data_t>
auto generateRandomMatrix(index_t size)
{
    Vector_t<data_t> randVec(size);

    if constexpr (std::is_integral_v<data_t>) {
        // Define range depending on signed or unsigned type
        const auto [rangeBegin, rangeEnd] = []() -> std::tuple<data_t, data_t> {
            if constexpr (std::is_signed_v<data_t>) {
                return {-100, 100};
            } else {
                return {1, 100};
            }
        }();

        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<data_t> distr(rangeBegin, rangeEnd);

        for (index_t i = 0; i < size; ++i) {
            data_t num = distr(eng);

            // remove zeros as this leads to errors when dividing
            if (num == 0)
                num = 1;
            randVec[i] = num;
        }
    } else {
        randVec.setRandom();
    }

    return randVec;
}

void makeGif(int s)
{
    // generate 2d phantom
    IndexVector_t size({{s, s}});

    auto phantom = phantoms::modifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // generate circular trajectory
    index_t numAngles{100}, arc{360};
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
    IS_ADMML2<real_t> admm{projector, sinogram, A, proxg, tau};

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations", noIterations);

    auto afterStep = [](DataContainer<float> state, index_t i, index_t) {
        io::write(state, fmt::format("raw/{}.pgm", i));
    };

    auto containerSize = volumeDescriptor.getNumberOfCoefficients();

    auto randVec = generateRandomMatrix<float>(containerSize);

    auto x0 = DataContainer<float>(volumeDescriptor, randVec);

    admm.run(noIterations, x0, afterStep);

    // write the reconstruction out
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
            return 100;
        }
    }();

    try {
        makeGif(size);
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
