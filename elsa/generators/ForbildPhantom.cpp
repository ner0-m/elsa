#include "ForbildPhantom.h"

using namespace std;

namespace elsa::phantoms
{

    template <typename data_t>
    ForbildPhantom<data_t>::ForbildPhantom(index_t maxDimension, data_t phantomMaxSize,
                                           int maxOrderIndex)
        : maxDimension{maxDimension},
          phantomMaxSize{phantomMaxSize},
          maxOrderIndex{maxOrderIndex} {};

    template <typename data_t>
    index_t scaleForbildShift(index_t maxDimension, data_t value, data_t maxValue)
    {
        return std::lround((value + (maxValue / 2)) * static_cast<data_t>(maxDimension)
                           / maxValue /* cm*/);
    }

    template <typename data_t>
    data_t scaleForbild(index_t maxDimension, data_t value, data_t maxValue)
    {
        return value * static_cast<data_t>(maxDimension) / maxValue /* cm*/;
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addEllipsoids(std::vector<std::array<data_t, 11>> datas)
    {
        for (auto data : datas) {
            data_t amplit = data[0];
            Vec3X<data_t> halfAxis(3);
            halfAxis << scaleForbild(maxDimension, data[1], phantomMaxSize),
                scaleForbild(maxDimension, data[2], phantomMaxSize),
                scaleForbild(maxDimension, data[3], phantomMaxSize);

            Vec3i center(3);
            center << scaleForbildShift(maxDimension, data[4], phantomMaxSize),
                scaleForbildShift(maxDimension, data[5], phantomMaxSize),
                scaleForbildShift(maxDimension, data[6], phantomMaxSize);

            Vec3X<data_t> euler(3);
            euler << data[7], data[8], data[9];

            int order = int(data[10]);

            if (std::abs(halfAxis[0]) < 0.5 || std::abs(halfAxis[1]) < 0.5
                || std::abs(halfAxis[2]) < 0.5 || amplit < data_t(0)) {
                Logger::get("ForbildPhantom::addEllipsoids")
                    ->warn("Ellipsoid will not be rendered, because of amplitude<0 or an invalid "
                           "half axis! amplitude {}, half axis ({},{},{}) ",
                           amplit, halfAxis[0], halfAxis[1], halfAxis[2]);
                continue;
            }

            ellipsoids.insert_or_assign(order, Ellipsoid<data_t>{amplit, center, halfAxis, euler});
        }
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addEllipsoidsClippedX(std::vector<std::array<data_t, 12>> datas)
    {
        for (auto data : datas) {
            data_t amplit = data[0];
            Vec3X<data_t> halfAxis(3);
            halfAxis << scaleForbild(maxDimension, data[1], phantomMaxSize),
                scaleForbild(maxDimension, data[2], phantomMaxSize),
                scaleForbild(maxDimension, data[3], phantomMaxSize);

            Vec3i center(3);
            center << scaleForbildShift(maxDimension, data[4], phantomMaxSize),
                scaleForbildShift(maxDimension, data[5], phantomMaxSize),
                scaleForbildShift(maxDimension, data[6], phantomMaxSize);

            Vec3X<data_t> euler(3);
            euler << data[7], data[8], data[9];

            index_t clippedX = scaleForbildShift(maxDimension, data[10], phantomMaxSize);

            int order = int(data[11]);

            if (std::abs(halfAxis[0]) < 0.5 || std::abs(halfAxis[1]) < 0.5
                || std::abs(halfAxis[2]) < 0.5 || amplit < data_t(0)) {
                Logger::get("ForbildPhantom::addEllipsoidsClippedX")
                    ->warn("Ellipsoid will not be rendered, because of amplitude<0 or an invalid "
                           "half axis! amplitude {}, half axis ({},{},{})",
                           amplit, halfAxis[0], halfAxis[1], halfAxis[2]);
                continue;
            }
            auto clippX = [clippedX, center](auto minMax) {
                // minMax in object space
                minMax[3] = data_t(clippedX) - data_t(center[INDEX_X]);
                return minMax;
            };

            ellipsoidsClippedX.insert_or_assign(
                order, std::pair{Ellipsoid<data_t>{amplit, center, halfAxis, euler}, clippX});
        }
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addSpheres(std::vector<std::array<data_t, 6>> datas)
    {
        for (auto data : datas) {
            data_t amplit = data[0];
            Vec3i center(3);
            center << scaleForbildShift(maxDimension, data[1], phantomMaxSize),
                scaleForbildShift(maxDimension, data[2], phantomMaxSize),
                scaleForbildShift(maxDimension, data[3], phantomMaxSize);

            data_t radius = scaleForbild(maxDimension, data[4], phantomMaxSize);

            int order = int(data[5]);

            if (radius < 0.5 || amplit < data_t(0)) {
                Logger::get("ForbildPhantom::addSpheres")
                    ->warn("Sphere will not be rendered, because of amplitude<0 or an invalid "
                           "radius!"
                           " amplitude {}, radius {} ",
                           amplit, radius);
                continue;
            }

            spheres.insert_or_assign(order, Sphere<data_t>{amplit, center, radius});
        }
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addEllipCylinders(std::vector<std::array<data_t, 9>> datas)
    {
        for (auto data : datas) {

            Orientation o = static_cast<Orientation>(std::lround(data[0]));
            data_t amplit = data[1];
            data_t length = scaleForbild(maxDimension, data[2], phantomMaxSize);
            Vec3i center;
            center << scaleForbildShift(maxDimension, data[3], phantomMaxSize),
                scaleForbildShift(maxDimension, data[4], phantomMaxSize),
                scaleForbildShift(maxDimension, data[5], phantomMaxSize);

            Vec2X<data_t> halfAxis;
            halfAxis << scaleForbild(maxDimension, data[6], phantomMaxSize),
                scaleForbild(maxDimension, data[7], phantomMaxSize);

            int order = int(data[8]);

            if (std::abs(halfAxis[0]) < 0.5 || std::abs(halfAxis[1]) < 0.5 || amplit < data_t(0)
                || length < 1.0) {
                Logger::get("ForbildPhantom::addEllipCylinders")
                    ->warn("EllipCylinder will not be rendered, because of amplitude<0 or an "
                           "invalid "
                           "half axis! amplitude {}, half axis ({},{}), length {} ",
                           amplit, halfAxis[0], halfAxis[1], length);
                continue;
            }

            ellipCylinder.insert_or_assign(
                order, EllipCylinder<data_t>{o, amplit, center, halfAxis, length});
        }
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addEllipCylindersFree(std::vector<std::array<data_t, 11>> datas)
    {
        for (auto data : datas) {

            data_t amplit = data[0];
            data_t length = scaleForbild(maxDimension, data[1], phantomMaxSize);
            Vec3i center;
            center << scaleForbildShift(maxDimension, data[2], phantomMaxSize),
                scaleForbildShift(maxDimension, data[3], phantomMaxSize),
                scaleForbildShift(maxDimension, data[4], phantomMaxSize);

            Vec2X<data_t> halfAxis;
            halfAxis << scaleForbild(maxDimension, data[5], phantomMaxSize),
                scaleForbild(maxDimension, data[6], phantomMaxSize);

            Vec3X<data_t> eulers;
            eulers << data[7], data[8], data[9];

            int order = int(data[10]);

            if (std::abs(halfAxis[0]) < 0.5 || std::abs(halfAxis[1]) < 0.5 || amplit < data_t(0)
                || length < 1.0) {
                Logger::get("ForbildPhantom::addEllipCylindersFree")
                    ->warn("EllipCylinderFree will not be rendered, because of amplitude<0 "
                           "or an "
                           "invalid "
                           "half axis! amplitude {}, half axis ({},{}), length {}, eulers "
                           "({},{},{}) ",
                           amplit, halfAxis[0], halfAxis[1], length, eulers[INDEX_A],
                           eulers[INDEX_B], eulers[INDEX_C]);
                continue;
            }

            ellipCylinderFree.insert_or_assign(
                order, EllipCylinderFree<data_t>{amplit, center, halfAxis, length, eulers});
        }
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addCylinders(std::vector<std::array<data_t, 8>> datas)
    {
        for (auto data : datas) {

            Orientation o = static_cast<Orientation>(std::lround(data[0]));
            data_t amplit = data[1];
            Vec3i center;
            center << scaleForbildShift(maxDimension, data[2], phantomMaxSize),
                scaleForbildShift(maxDimension, data[3], phantomMaxSize),
                scaleForbildShift(maxDimension, data[4], phantomMaxSize);

            data_t radius = scaleForbild(maxDimension, data[5], phantomMaxSize);
            data_t length = scaleForbild(maxDimension, data[6], phantomMaxSize);

            int order = int(data[7]);

            if (amplit < data_t(0) || radius < 0.5 || length < 1.0) {
                Logger::get("ForbildPhantom::addCylinders")
                    ->warn("Cylinder will not be rendered, because of amplitude<0 or an "
                           "invalid "
                           "radius or invalid length! amplitude {}, radius {}, length {} ",
                           amplit, radius, length);
                continue;
            }

            cylinders.insert_or_assign(order, Cylinder<data_t>{o, amplit, center, radius, length});
        }
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addCylindersFree(std::vector<std::array<data_t, 10>> datas)
    {
        for (auto data : datas) {
            data_t amplit = data[0];
            Vec3i center;
            center << scaleForbildShift(maxDimension, data[1], phantomMaxSize),
                scaleForbildShift(maxDimension, data[2], phantomMaxSize),
                scaleForbildShift(maxDimension, data[3], phantomMaxSize);

            data_t radius = scaleForbild(maxDimension, data[4], phantomMaxSize);
            data_t length = scaleForbild(maxDimension, data[5], phantomMaxSize);

            Vec3X<data_t> eulers;
            eulers << data[6], data[7], data[8];

            int order = int(data[9]);

            if (amplit < data_t(0) || radius < 0.5 || length < 1.0) {
                Logger::get("ForbildPhantom::addCylinders")
                    ->warn("Cylinder will not be rendered, because of amplitude<0 or an "
                           "invalid "
                           "radius or invalid length! amplitude {}, radius {}, length {} ",
                           amplit, radius, length);
                continue;
            }

            cylindersFree.insert_or_assign(
                order, CylinderFree<data_t>{amplit, center, radius, length, eulers});
        }
    }

    template <typename data_t>
    void ForbildPhantom<data_t>::addBoxes(std::vector<std::array<data_t, 8>> datas)
    {
        for (auto data : datas) {

            data_t amplit = data[0];
            Vec3i center;
            center << scaleForbildShift(maxDimension, data[1], phantomMaxSize),
                scaleForbildShift(maxDimension, data[2], phantomMaxSize),
                scaleForbildShift(maxDimension, data[3], phantomMaxSize);

            Vec3X<data_t> edgeLengths;
            edgeLengths << scaleForbild(maxDimension, data[4], phantomMaxSize),
                scaleForbild(maxDimension, data[5], phantomMaxSize),
                scaleForbild(maxDimension, data[6], phantomMaxSize);

            int order = int(data[7]);

            if (std::abs(edgeLengths[0]) < 1 || std::abs(edgeLengths[1]) < 1
                || std::abs(edgeLengths[2]) < 1 || amplit < data_t(0)) {
                Logger::get("ForbildPhantom::addBoxes")
                    ->warn("Box will not be rendered, because of amplitude<0 or an invalid "
                           "edge lengths! amplitude {}, edge lengths ({},{},{}) ",
                           amplit, edgeLengths[0], edgeLengths[1], edgeLengths[2]);
                continue;
            }

            boxes.insert_or_assign(order, Box<data_t>{amplit, center, edgeLengths});
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ForbildPhantom<float>;
    template class ForbildPhantom<double>;

} // namespace elsa::phantoms