#include "DataContainer.h"
#include "DataDescriptor.h"
#include "ExpressionPredicates.h"
#include "VolumeDescriptor.h"
#include "elsa.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include "IS_ADMML2.h"
#include "elsaDefines.h"

using namespace elsa;

DataContainer<double> importHTC(std::string filename, index_t rows, index_t cols)
{

    DataContainer<double> sinogram{VolumeDescriptor{{cols, rows}}};
    std::ifstream file{filename};

    file.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(sinogram.storage().data())),
              rows * cols * sizeof(double));
    return sinogram;
}

void makeGif()
{

    index_t numAngles{560}, numRows{721};

    auto sinogram = importHTC("htc2022_ta_full.mat.bin", 721, 560);
    io::write(sinogram, "ta.pgm");

    VolumeDescriptor volumeDescriptor{560, 721};

    //     // generate circular trajectory
    //     index_t arc{360};
    //     const auto dSO = 410.66;
    //     const auto dSD = 553.74;

    //     auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(numAngles,
    //     volumeDescriptor,
    //                                                                       arc, dSD, dSD - dSO);

    //     // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    //     // VolumeDescriptor
    //     JosephsMethod<double> projector(volumeDescriptor, *sinoDescriptor);

    //     auto A = FiniteDifferences<double>(volumeDescriptor);
    //     auto proxg = ProximalL1<double>{};
    //     auto tau = double{0.1};

    //     index_t noIterations{10};
    //     auto afterStep = [](DataContainer<double> state, index_t i, index_t) {
    //         io::write(state, fmt::format("raw/{}.pgm", i));
    //     };

    //     // solve the reconstruction problem
    //     IS_ADMML2<double> admm{projector, sinogram, A, proxg, tau};

    //     Logger::get("Info")->info("Solving reconstruction using {} iterations", noIterations);

    //     admm.solve(noIterations, std::nullopt, afterStep);
    // }

    int main(int, char**)
    {
        try {
            makeGif();
        } catch (std::exception& e) {
            std::cerr << "An exception occurred: " << e.what() << "\n";
        }
    }
