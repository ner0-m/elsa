#include "DataContainer.h"
#include "DataDescriptor.h"
#include "ExpressionPredicates.h"
#include "VolumeDescriptor.h"
#include "elsa.h"
#include "SiddonsMethodCUDA.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
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
    const auto distanceSourceOrigin = 410.66;
    const auto distanceSourceDetector = 553.74;

    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantomDescriptor, arc, distanceSourceOrigin,
        (distanceSourceDetector - distanceSourceOrigin), std::nullopt, std::nullopt,
        IndexVector_t{{numPixels}});

    auto sinogram = importHTC("htc2022_td_full.mat.bin", *sinoDescriptor);
    io::write(sinogram, "td.pgm");

    SiddonsMethodCUDA<double> projector(phantomDescriptor, *sinoDescriptor);

    auto A = FiniteDifferences<double>(phantomDescriptor);
    auto proxg = ProximalL1<double>{};
    auto tau = double{0.1};

    // solve the reconstruction problem
    ADMML2<double> admm{projector, sinogram, A, proxg, tau};

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations", noIterations);
    auto reco = admm.solve(noIterations);

    // write the reconstruction out
    io::write(reco, "htc_admml2.pgm");

    // auto cosine = makeCosine<double>(sinogram.getDataDescriptor());

    // auto reconstruction = FBP<double>{projector, cosine}.apply(sinogram);
    // io::write(reconstruction, "fbp_htc_td.pgm");
}

int main(int, char**)
{
    try {
        makeGif();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
