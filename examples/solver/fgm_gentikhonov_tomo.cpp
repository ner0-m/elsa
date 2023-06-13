/*
 * This examples solves the following problem:
 * \[
 * \min_x 0.5 * || A x - b ||_2^2 + \lambda || \nabla x ||_2
 * \]
 * using a fast gradient method (FGM). The problem is often referred to as generalized Tikhonov.
 *
 * The generalized Tikhonov penalizes large gradients. Hence, it will result in
 * smoother reconstructions while still preserving edges.
 *
 * The application is attenuation X-ray computed tomography (CT). This
 * translates to the above equation that the system matrix \f$A\f$ is an
 * discrete approximation of the Radon Transform. In this example 512
 * projections are acquired over the complete circle. This is an easy setup, as
 * there are many projections from the complete circle.
 *
 * The phantom is a simple modified Shepp-Logan phantom, which is commonly
 * found for synthetic reconstructions.
 */
#include "Identity.h"
#include "elsa.h"

using namespace elsa;

void example2d()
{
    // generate 2d phantom
    IndexVector_t size({{128, 128}});
    auto phantom = phantoms::modifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // generate circular trajectory
    index_t numAngles{512}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    // VolumeDescriptor
    SiddonsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    // simulate the sinogram
    Logger::get("Info")->info("Calculate sinogram");
    auto sinogram = projector.apply(phantom);

    // Setup problem
    auto dataterm = LeastSquares(projector, sinogram);

    auto grad = FiniteDifferences(volumeDescriptor);
    auto lambda = 100.f;
    auto tvl2reg = lambda * L2Reg(grad);

    auto problem = dataterm + tvl2reg;

    // solve the reconstruction problem
    FGM solver(problem);

    index_t niters{50};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of FGM", niters);
    auto recon = solver.solve(niters);

    // write the reconstruction out
    io::write(recon, "reco_fgm_gentikhonov_tomo.pgm");
}

int main()
{
    try {
        example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
