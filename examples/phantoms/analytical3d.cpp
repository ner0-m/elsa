#include "DataDescriptor.h"
#include "VolumeDescriptor.h"
#include "elsa.h"
#include "Phantoms.h"
#include "analytical/Analytical.h"
#include "elsaDefines.h"
#include "JosephsMethodCUDA.h"

using namespace elsa;
using namespace elsa::phantoms;

int main(int, char*[])
{

    IndexVector_t size({{100, 100, 100}});
    VolumeDescriptor image{size};

    index_t numAngles{50}, arc{360};
    const auto distance = 100.0;
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(numAngles, image, arc,
                                                                      distance * 100.0f, distance);

    Logger::get("Info")->info("Making sinogram...");
    auto sinogram = analyticalSheppLogan<float>(image, *sinoDescriptor);

    Logger::get("Info")->info("Writing out sinogram...");
    io::write(sinogram, "3dsinogram.edf");

    // JosephsMethodCUDA projector{image, *sinoDescriptor};

    // auto A = FiniteDifferences<real_t>{image};
    // auto proxg = ProximalL1<real_t>{};
    // auto tau = real_t{0.1};

    // // solve the reconstruction problem
    // ADMML2<real_t> admm(projector, sinogram, A, proxg, tau);
    // CGLS solver(projector, sinogram);

    // index_t noIterations{50};
    // Logger::get("Info")->info("Solving reconstruction using {} iterations of conjugate gradient",
    //                           noIterations);
    // auto reconstruction = solver.solve(noIterations);

    // // write the reconstruction out
    // EDF::write(reconstruction, "3dreconstruction.edf");
    // for (index_t i = 0; i < size[2]; i++) {
    //     std::ostringstream o;
    //     o << "slices/slice" << i << ".pgm";
    //     io::write(reconstruction.slice(i), o.str());
    // }

    return 0;
}
