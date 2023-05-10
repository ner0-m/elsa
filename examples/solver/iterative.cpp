#include "DataContainer.h"
#include "DataDescriptor.h"
#include "ExpressionPredicates.h"
#include "VolumeDescriptor.h"
#include "elsa.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include "IS_ADMML2.h"
#include "elsaDefines.h"

using namespace elsa;

template <typename D>
DataContainer<double> importHTC(std::string filename, D descriptor)
{

    DataContainer<double> sinogram{descriptor};
    std::ifstream file{filename};

    file.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(sinogram.storage().data())),
              descriptor.getNumberOfCoefficients() * sizeof(double));
    return sinogram;
}

void makeGif()
{

    index_t numPixels{560}, numAngles{721};

    VolumeDescriptor phantomDescriptor{512, 512};

    // generate circular trajectory
    index_t arc{360};
    const auto dSO = 410.66;
    const auto dSD = 553.74;

    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantomDescriptor, arc, dSD, dSD - dSO, std::nullopt, std::nullopt,
        IndexVector_t{{numPixels}});

    auto sinogram = importHTC("htc2022_ta_full.mat.bin", *sinoDescriptor);
    io::write(sinogram, "ta.pgm");

    SiddonsMethod<double> projector(dynamic_cast<const VolumeDescriptor&>(phantomDescriptor),
                                    *sinoDescriptor);

    // auto cosine = makeCosine<double>(sinogram.getDataDescriptor());

    // auto reconstruction = FBP<double>{projector, cosine}.apply(sinogram);
    // io::write(reconstruction, "fbp_htc.pgm");

    auto A = FiniteDifferences<double>(phantomDescriptor);
    auto proxg = ProximalL1<double>{};
    auto tau = double{0.1};

    index_t noIterations{10};
    auto afterStep = [](DataContainer<double> state, index_t i, index_t) {
        io::write(state, fmt::format("raw/{}.pgm", i));
    };

    // solve the reconstruction problem
    IS_ADMML2<double> admm{projector, sinogram, A, proxg, tau};

    Logger::get("Info")->info("Solving reconstruction using {} iterations", noIterations);

    admm.solve(noIterations, std::nullopt, afterStep);
}

int main(int, char**)
{
    try {
        makeGif();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
