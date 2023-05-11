#include "DataDescriptor.h"
#include "VolumeDescriptor.h"
#include "elsa.h"
#include "Phantoms.h"
#include "analytical/Analytical.h"
#include "projectors_cuda/JosephsMethodCUDA.h"

using namespace elsa;
using namespace elsa::phantoms;

int main(int, char*[])
{

    VolumeDescriptor image{{500, 500}};

    index_t numAngles{512}, arc{360};
    const auto distance = 100.0;
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(numAngles, image, arc,
                                                                      distance * 100.0f, distance);

    auto sinogram = analyticalSheppLogan<float>(image, *sinoDescriptor);

    io::write(sinogram, "sinogram.pgm");

    JosephsMethodCUDA projector{image, *sinoDescriptor};

    auto A = FiniteDifferences<real_t>{image};
    auto proxg = ProximalL1<real_t>{};
    auto tau = real_t{0.1};

    // solve the reconstruction problem
    ADMML2<real_t> admm(projector, sinogram, A, proxg, tau);

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations", noIterations);
    auto reco = admm.solve(noIterations);

    // write the reconstruction out
    io::write(reco, "reconstruction.pgm");

    return 0;
}
