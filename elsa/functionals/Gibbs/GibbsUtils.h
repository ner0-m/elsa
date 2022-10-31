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
         * @brief Finds all neighbours
         *
         * @param[in] descriptor
         *
         * @return std::vector<index_t> list of entry indexes of all real neighbours according to
         * descriptor
         *
         * @throw InvalidArgumentError if the vector is empty or the descriptors in the vector
         * don't all have the same size
         *
         *
         */
        std::vector<index_t> allNeighbours(const DataDescriptor& descriptor,
                                           const IndexVector_t& point);

        /**
         * @brief Finds all neighbours that set based on descriptor's index
         *
         * @param[in] descriptor
         *
         * @return std::vector<index_t> list of entry index of all real neighbours according to
         * descriptor
         *
         * @throw InvalidArgumentError if the vector is empty or the descriptors in the vector
         * don't all have the same size
         *
         *
         */
        inline std::vector<index_t> allNeighbours(const DataDescriptor& descriptor, index_t point)
        {
            return elsa::Gibbs::allNeighbours(descriptor, descriptor.getCoordinateFromIndex(point));
        }

        template <typename data_t>
        std::vector<data_t> allNeighboursData(const DataContainer<data_t>& container,
                                              const IndexVector_t& point)
        {
            std::vector<index_t> indexes =
                elsa::Gibbs::allNeighbours(container.getDataDescriptor(), point);
            std::vector<data_t> res(indexes.size());
            std::transform(indexes.begin(), indexes.end(), res.begin(),
                           [container](index_t ind) { return container[ind]; });
            return res;
        }

        template <typename data_t>
        std::vector<data_t> allNeighboursData(const DataContainer<data_t>& container, index_t point)
        {
            return allNeighboursData(container,
                                     container.getDataDescriptor().getCoordinateFromIndex(point));
        };

        /**
         *
         * @tparam data_t
         * @param container
         * @param point
         * @param calcTransform - some transform over value at neighbour point and value at point
         * before summarizing
         * @param calcCoefficient some calculation of coefficient for transformed value in sum
         * according to level of closeness the neighbour point to original point
         * @return neighbour points sum with all specified transform and coefficient
         */
        template <typename data_t>
        data_t allNeighboursSum(const DataContainer<data_t>& container, const IndexVector_t& point,
                                std::function<data_t(data_t, data_t)>& calcTransform,
                                std::function<data_t(index_t)>& calcCoefficient)
        {
            std::vector<IndexVector_t> shifts = elsa::Gibbs::generateNeighbourShift(
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
                               return calcCoefficient(shift.array().abs().sum())
                                      * calcTransform(container.at(coordinate), pointData);
                           });
            return std::accumulate(transformedData.begin(), transformedData.end(),
                                   static_cast<data_t>(0));
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
        data_t allNeighboursGibbsSubSum(
            const DataContainer<data_t>& container, const IndexVector_t& point,
            std::function<data_t(data_t, data_t)> psi = [](data_t x, data_t y) {
                return 0.5 * (x - y) * (x - y);
            })
        {
            return allNeighboursSum(container, point, psi, [](index_t diff) {
                return diff <= 0 ? 0 : (diff == 1 ? 1 : 1 / sqrt(diff));
            });
        }

    } // namespace Gibbs
} // namespace elsa