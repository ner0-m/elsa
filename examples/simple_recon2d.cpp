/// [simplerecon header begin]
#include "elsa.h"

#include <iostream>

using namespace elsa;
/// [simplerecon header end]

void example2d()
{
    /// [simplerecon phantom create]
    IndexVector_t size({{128, 128}});
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    /// [simplerecon phantom create]

    /// [simplerecon phantom write]
    io::write(phantom, "2dphantom.pgm");
    /// [simplerecon phantom write]

    /// [simplerecon trajectory]
    const index_t numAngles = 180;
    const index_t arc = 360;
    const auto distance = static_cast<real_t>(size[0]);
    const auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);
    /// [simplerecon trajectory]

    /// [simplerecon sinogram]
    const auto& phanDescriptor = dynamic_cast<const VolumeDescriptor&>(phantom.getDataDescriptor());
    SiddonsMethod projector(phanDescriptor, *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);
    /// [simplerecon sinogram]

    // write the sinogram out
    io::write(sinogram, "2dsinogram.pgm");

    // setup reconstruction problem
    /// [simplerecon solver]
    WLSProblem wlsProblem(projector, sinogram);
    CG solver(wlsProblem);

    index_t iterations{20};
    auto reconstruction = solver.solve(iterations);
    /// [simplerecon solver]

    // write the reconstruction out
    io::write(reconstruction, "2dreconstruction.pgm");

    /// [simplerecon analysis]
    DataContainer diff = phantom - reconstruction;
    io::write(diff, "2ddiff.pgm");
    Logger::get("example")->info("L2-Norm of the phantom: {:.4f}", phantom.l2Norm());
    Logger::get("example")->info("L2-Norm of the phantom: {:.4f}", reconstruction.l2Norm());
    /// [simplerecon analysis]
}

int main()
{
    try {
        example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
