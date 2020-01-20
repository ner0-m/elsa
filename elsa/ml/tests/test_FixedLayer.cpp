#include <catch2/catch.hpp>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "JosephsMethod.h"
#include "FixedLayer.h"
#include "PhantomGenerator.h"
#include "CircleTrajectoryGenerator.h"
#include "Geometry.h"

using namespace elsa;

TEST_CASE("FixedLayer", "elsa_ml")
{
    SECTION("Forward semantics")
    {
        // A fixed layer's forward pass is defined by its applyAdjoint method

        // generate 2d phantom
        IndexVector_t size(2);
        size << 32, 32;
        auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

        index_t noAngles{180}, arc{360};
        auto [geometry, sinoDescriptor] = CircleTrajectoryGenerator::createTrajectory(
            noAngles, phantom.getDataDescriptor(), arc, size(0) * 100, size(0));

        JosephsMethod projector(phantom.getDataDescriptor(), *sinoDescriptor, geometry);

        auto sinogram = projector.apply(phantom);

        auto phantom2 = projector.applyAdjoint(sinogram);

        FixedLayer layer(*sinoDescriptor ,projector);

        // The layer's input descriptor is the operator's range descriptor
        REQUIRE(layer.getInputDescriptor() == projector.getRangeDescriptor());

        // The layer's output descriptor is the operator's domain descriptor
        REQUIRE(layer.getOutputDescriptor() == projector.getDomainDescriptor());

        // Setup layer
        auto backend = layer.getBackend();
        backend->setInput(sinogram);
        backend->initialize();
        backend->compile();

        // Forward propagate
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);

        // Get output
        auto output = backend->getOutput();

        for (index_t i = 0; i < output.getDataDescriptor().getNumberOfCoefficients(); ++i)
          REQUIRE(output[i] == Approx(phantom2[i]));
    }

    SECTION("Backward semantics")
    {
        // A fixed layer's forward pass is defined by its apply method
    }
}
