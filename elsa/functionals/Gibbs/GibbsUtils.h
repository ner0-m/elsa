#pragma once
#include "DataContainer.h"
#include "DataDescriptor.h"
#include <algorithm>
#include <numeric>

namespace elsa
{
    namespace Gibbs
    {
        /**
         * Helper function for neighbours shifting index
         * @param n
         * @return
         */
        std::vector<IndexVector_t> generateNeighbourShift(index_t n);

        /**
         * Helper function for neighbours shifting index
         * @param n
         * @return
         */
        std::vector<IndexVector_t> generateNeighbourShift(index_t n);

        const std::vector<IndexVector_t> shiftsOthers = generateNeighbourShift(0);
        const std::vector<IndexVector_t> shifts1d = generateNeighbourShift(1);
        const std::vector<IndexVector_t> shifts2d = generateNeighbourShift(2);
        const std::vector<IndexVector_t> shifts3d = generateNeighbourShift(3);

        inline std::vector<IndexVector_t> getNeighbourShift(index_t n)
        {
            switch (n) {
                case 1:
                    return shifts1d;
                case 2:
                    return shifts2d;
                case 3:
                    return shifts3d;
                default:
                    return shiftsOthers;
            }
        }

        /**
         *
         * @tparam data_t
         * @param container
         * @param point
         * @param psi - function over value at the point and value at the neighbour (default is 0.5
         * * difference^2)
         * @return
         */
        template <typename data_t>
        data_t allNeighboursSum(
            const DataContainer<data_t>& container, const IndexVector_t& point,
            std::function<data_t(data_t, data_t)> psi =
                [](data_t x, data_t y) { return 0.5 * (x - y) * (x - y); },
            std::function<data_t(index_t)> coefGibbs =
                [](index_t diff) { return diff <= 0 ? 0 : (diff == 1 ? 1 : 1 / sqrt(diff)); })
        {

            std::vector<IndexVector_t> shifts = elsa::Gibbs::getNeighbourShift(
                container.getDataDescriptor().getNumberOfDimensions());
            data_t pointData = container.at(point);
            IndexVector_t sizes =
                container.getDataDescriptor().getNumberOfCoefficientsPerDimension();
            std::vector<data_t> transformedData(shifts.size());

            std::transform(shifts.begin(), shifts.end(), transformedData.begin(),
                           [=](IndexVector_t& shift) {
                               IndexVector_t coordinate = point + shift;
                               if ((coordinate.array() < 0).any()
                                   || ((coordinate.array() - sizes.array()) >= 0).any())
                                   return static_cast<data_t>(0);
                               return coefGibbs(shift.array().abs().sum())
                                      * psi(container.at(coordinate), pointData);
                           });
            return std::accumulate(transformedData.begin(), transformedData.end(),
                                   static_cast<data_t>(0));
        }

    } // namespace Gibbs
} // namespace elsa