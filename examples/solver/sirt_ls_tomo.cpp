/*
 * This examples solves the following problem:
 * \[
 * \min_x 0.5 * || A x - b ||_2^2
 * \]
 * using a Simultaneous Iterative Reconstruction Techniques (SIRT).
 *
 * Landweber type solvers have the following update rule:
 * \[
 * x_{k+1} =  x_{k} + \lambda T A^T M (A(x_{k}) - b)
 * \]
 * For SIRT, \f$T\f$ and \f$M\f$ are the inverse of the row sum and column sum
 * of the matrix \f$A\f$ respectively.
 *
 * The least squares problem is the "easiest" problem to solve many different
 * inverse problems. Many different solvers can solve it, but especially in the
 * presence of noise it quickly breaks down.
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
#include "PowerIterations.h"

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
    JosephsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    // simulate the sinogram
    Logger::get("Info")->info("Calculate sinogram");
    auto sinogram = projector.apply(phantom);

    // A decent step length is calculated by default, you can always pass it
    // as a third parameter to avoid the expensive calculation of the largest
    // eigenvalue
    SIRT solver(projector, sinogram);

    index_t niters{50};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of SIRT", niters);
    auto recon = solver.solve(niters);

    // write the reconstruction out
    io::write(recon, "reco_sirt_ls_tomo.pgm");
}

int main()
{
    try {
        example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
