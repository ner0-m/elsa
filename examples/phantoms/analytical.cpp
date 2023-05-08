#include "DataDescriptor.h"
#include "VolumeDescriptor.h"
#include "elsa.h"
#include "Phantoms.h"
#include "analytical/Image.h"

using namespace elsa;
using namespace elsa::phantoms;

int main(int, char*[])
{

    VolumeDescriptor gollum{{500, 500}};

    index_t numAngles{512}, arc{360};
    const auto distance = 100.0;
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(numAngles, gollum, arc,
                                                                      distance * 100.0f, distance);

    auto sheppLogan = Ellipse<float>{{1, 2}, 1, 1} + Ellipse<float>{{0, 0}, 1, 1};
    auto sinogram = sheppLogan.makeSinogram(*sinoDescriptor);
    return 0;
}
