#include "DataDescriptor.h"
#include "VolumeDescriptor.h"
#include "elsa.h"
#include "Phantoms.h"
#include "analytical/Image.h"
#include "projectors_cuda/JosephsMethodCUDA.h"

using namespace elsa;
using namespace elsa::phantoms;

int main(int, char*[])
{

    VolumeDescriptor gollum{{500, 500}};

    index_t numAngles{512}, arc{360};
    const auto distance = 100.0;
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(numAngles, gollum, arc,
                                                                      distance * 100.0f, distance);

    auto sheppLogan = Ellipse<float>{{0, 0}, 50, 100};
    auto sinogram = sheppLogan.makeSinogram(*sinoDescriptor);

    io::write(sinogram, "ellipsen.pgm");

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    // VolumeDescriptor
    JosephsMethodCUDA projector{gollum, *sinoDescriptor};

    auto A = FiniteDifferences<real_t>(gollum);
    auto proxg = ProximalL1<real_t>{};
    auto tau = real_t{0.1};

    // solve the reconstruction problem
    ADMML2<real_t> admm(projector, sinogram, A, proxg, tau);

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations", noIterations);
    auto reco = admm.solve(noIterations);

    // write the reconstruction out
    io::write(reco, "analytical_reco.pgm");

    return 0;
}
