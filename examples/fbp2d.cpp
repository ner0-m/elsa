/// Elsa example program: basic 2d X-ray CT simulation and reconstruction using FBP

#include "CartesianIndices.h"
#include "DataContainer.h"
#include "Filter.h"
#include "VolumeDescriptor.h"
#include "elsa.h"

#include <iostream>
#include <ostream>

using namespace elsa;


void fbp2d()
{
    // generate 2d phantom
    IndexVector_t size({{128, 128}});
    auto phantom = phantoms::modifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // write the phantom out
    io::write(phantom, "fbp2d_phantom.pgm");

    // generate circular trajectory
    index_t numAngles{200}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 10000.0f, distance);

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    // VolumeDescriptor
    Logger::get("Info")->info("Create BlobProjector");
    SiddonsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    // simulate the sinogram
    Logger::get("Info")->info("Calculate sinogram");
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    Logger::get("Info")->info("Write sinogram");
    io::write(sinogram, "fbp2d_sinogram.pgm");

    Logger::get("Info")->info("Attempting reconstruction using filtered backprojection");

    auto ramlak = makeRamLak(sinogram.getDataDescriptor());
    auto shepplogan = makeSheppLogan(sinogram.getDataDescriptor());
    auto cosine = makeCosine(sinogram.getDataDescriptor());
    auto hann = makeHann(sinogram.getDataDescriptor());

    auto reconstruction = FBP{projector, ramlak}.apply(sinogram);
    io::write(reconstruction, "fbp2d_RamLak.pgm");

    reconstruction = FBP{projector, cosine}.apply(sinogram);
    io::write(reconstruction, "fbp2d_Cosine.pgm");

    reconstruction = FBP{projector, shepplogan}.apply(sinogram);
    io::write(reconstruction, "fbp2d_SheppLogan.pgm");

    reconstruction = FBP{projector, hann}.apply(sinogram);
    io::write(reconstruction, "fbp2d_Hann.pgm");

    DataContainer diff = reconstruction - phantom;
    std::cout << diff.lInfNorm() << std::endl
              << phantom.lInfNorm() << std::endl
              << reconstruction.lInfNorm() << std::endl
              << reconstruction.minElement() << "|" << reconstruction.maxElement();
}

int main()
{
    try {
        fbp2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
