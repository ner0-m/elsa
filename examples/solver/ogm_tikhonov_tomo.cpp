/*
 * This examples solves the following problem:
 * \[
 * \min_x 0.5 * || A x - b ||_2^2 + \lambda || x ||_2
 * \]
 * using a optimized gradient descent (OGM). The problem is often referred to as Tikhonov or
 * L2-regularization.
 *
 * The L2-regularization penalizes large values in the reconstruction, i.e. it
 * tries to minimize the L2-norm of the solution. This usually results in quite
 * smooth reconstructions. Edges can be blurred easily.
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
    auto dataterm = L2NormPow2(projector, sinogram);

    auto lambda = 0.1f;
    auto l2reg = lambda * L2NormPow2(volumeDescriptor);

    auto problem = dataterm + l2reg;

    // solve the reconstruction problem
    OGM solver(problem);

    index_t niters{50};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of OGM", niters);
    auto recon = solver.solve(niters);

    // write the reconstruction out
    io::write(recon, "reco_ogm_tikhonov_tomo.pgm");
}

int main()
{
    try {
        example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
