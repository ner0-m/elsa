/*
 * This examples solves the following problem:
 * \[
 * \min_x 0.5 * || A x - b ||_2^2
 * \]
 * using a conjugate gradient for least squares (CGLS).
 *
 * The least squares problem is the "easiest" problem to solve many different
 * inverse problems. Many different solvers can solve it, but especially in the
 * presence of noise it quickly breaks down.
 *
 * The application is attenuation X-ray computed tomography (CT). This
 * translates to the above equation that the system matrix \f$A\f$ is an
 * discrete approximation of the Radon Transform. In this specific example,
 * a limited angle tomography is reconstructed. I.e. a circular trajectory,
 * which does not cover the complete circle, and/or has certain missing pieces
 * of the circle. This can be quite a challenging, but interesting use case for
 * X-ray CT.
 *
 * The phantom is a simple modified Shepp-Logan phantom, which is commonly
 * found for synthetic reconstructions.
 */

#include "elsa.h"

#include <iostream>

using namespace elsa;

void limited_angle_example2d()
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = phantoms::modifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // generate circular trajectory
    index_t numAngles{360}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
        numAngles, std::pair(geometry::Degree(40), geometry::Degree(85)),
        phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // setup operator for 2d X-ray transform
    Logger::get("Info")->info("Simulating sinogram using Siddon's method");

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    // VolumeDescriptor
    SiddonsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // solve the reconstruction problem
    CGLS cgSolver(projector, sinogram);

    index_t niters{20};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of conjugate gradient",
                              niters);
    auto reco = cgSolver.solve(niters);

    // write the reconstruction out
    EDF::write(reco, "reco_cgls_ls_limitedtomo.edf");
}

int main()
{
    try {
        limited_angle_example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
