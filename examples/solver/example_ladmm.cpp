/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "BlockDescriptor.h"
#include "DataContainer.h"
#include "FiniteDifferences.h"
#include "IdenticalBlocksDescriptor.h"
#include "Identity.h"
#include "JosephsMethodCUDA.h"
#include "NoiseGenerators.h"
#include "RandomBlocksDescriptor.h"
#include "elsa.h"
#include "LinearizedADMM.h"
#include "CombinedProximal.h"
#include "ProximalL1.h"
#include "CombinedProximal.h"
#include "ProximalIdentity.h"
#include "ProximalL2Squared.h"

#include <iostream>
#include <limits>

using namespace elsa;

void example()
{
    // generate 2d phantom
    IndexVector_t size(2);
    /* size << 128, 128; */
    size << 512, 512;

    auto phantom = phantoms::modifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // generate circular trajectory
    index_t numAngles{512}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    JosephsMethodCUDA projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                                *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);
    /* sinogram += GaussianNoiseGenerator(0, 0.05)(sinogram); */

    auto proxf = ProximalIdentity<real_t>();

    auto prox1 = ProximalL2Squared(sinogram, 10);
    auto prox2 = ProximalL1<real_t>(0.1);
    auto proxg = CombinedProximal<real_t>(prox1, prox2);

    auto iden = Identity<real_t>(volumeDescriptor);
    auto grad = FiniteDifferences<real_t>(volumeDescriptor);

    std::vector<std::unique_ptr<DataDescriptor>> descs;
    descs.push_back(sinoDescriptor->clone());
    descs.push_back(grad.getRangeDescriptor().clone());

    auto blockDesc = RandomBlocksDescriptor(descs);

    std::vector<std::unique_ptr<LinearOperator<real_t>>> opList;
    opList.push_back(projector.clone());
    opList.push_back(grad.clone());

    BlockLinearOperator K(volumeDescriptor, blockDesc, opList,
                          BlockLinearOperator<real_t>::BlockType::ROW);

    // solve the reconstruction problem
    LinearizedADMM<real_t> solver(K, proxf, proxg, 10, 0.00005);

    index_t noIterations{20};
    auto reco = solver.solve(noIterations);

    std::cout << "reco l2 norm: " << reco.l2Norm() << "\n";
    std::cout << "phantom l2 norm: " << phantom.l2Norm() << "\n";

    // write the reconstruction out
    io::write(reco, "2dreconstruction_ladmm.pgm");
}

int main()
{
    try {
        example();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
