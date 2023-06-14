#pragma once
#include "Ellipsoid.h"
#include "Sphere.h"
#include "EllipCylinder.h"
#include "EllipCylinderFree.h"
#include "Cylinder.h"
#include "CylinderFree.h"
#include "Box.h"

#include <vector>
#include <map>

namespace elsa::phantoms
{

    template <typename data_t>
    class ForbildPhantom
    {
        /**
         * @brief max number of voxel of the datacontainer per dimension
         */
        index_t maxDimension;

        /**
         * @brief max dimension for the given values e.g. 26 cm for FORBILD head phantom
         */
        data_t phantomMaxSize;

        /**
         * @brief max order index. 0 to n where n is the max order index.
         */
        int maxOrderIndex;

        std::map<int, Ellipsoid<data_t>> ellipsoids{};
        std::map<int, std::pair<Ellipsoid<data_t>, MinMaxFunction<data_t>>> ellipsoidsClippedX{};
        std::map<int, Sphere<data_t>> spheres{};
        std::map<int, EllipCylinder<data_t>> ellipCylinder{};
        std::map<int, EllipCylinderFree<data_t>> ellipCylinderFree{};
        std::map<int, Cylinder<data_t>> cylinders{};
        std::map<int, CylinderFree<data_t>> cylindersFree{};
        std::map<int, Box<data_t>> boxes{};

    public:
        ForbildPhantom(index_t maxDimension, data_t phantomMaxSize, int maxOrderIndex);
        void addEllipsoids(std::vector<std::array<data_t, 11>> data);
        void addEllipsoidsClippedX(std::vector<std::array<data_t, 12>> data);
        void addSpheres(std::vector<std::array<data_t, 6>> datas);
        void addEllipCylinders(std::vector<std::array<data_t, 9>> datas);
        void addEllipCylindersFree(std::vector<std::array<data_t, 11>> datas);
        void addCylinders(std::vector<std::array<data_t, 8>> datas);
        void addCylindersFree(std::vector<std::array<data_t, 10>> datas);
        void addBoxes(std::vector<std::array<data_t, 8>> datas);

        int getMaxOrderIndex() { return maxOrderIndex; };

        std::map<int, Ellipsoid<data_t>> getEllipsoids() const { return ellipsoids; };
        std::map<int, std::pair<Ellipsoid<data_t>, MinMaxFunction<data_t>>>
            getEllipsoidsClippedX() const
        {
            return ellipsoidsClippedX;
        };
        std::map<int, Sphere<data_t>> getSpheres() const { return spheres; };
        std::map<int, EllipCylinder<data_t>> getEllipCylinders() const { return ellipCylinder; };
        std::map<int, EllipCylinderFree<data_t>> getEllipCylindersFree() const
        {
            return ellipCylinderFree;
        };
        std::map<int, Cylinder<data_t>> getCylinders() const { return cylinders; };
        std::map<int, CylinderFree<data_t>> getCylindersFree() const { return cylindersFree; };
        std::map<int, Box<data_t>> getBoxes() const { return boxes; };
    };

} // namespace elsa::phantoms
