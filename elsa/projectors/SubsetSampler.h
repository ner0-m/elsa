#pragma once

#include "elsaDefines.h"
#include "DetectorDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "SiddonsMethod.h"
#include "JosephsMethod.h"

namespace elsa
{
    /**
     * @brief Class representing a subset sampling method.
     *
     * @author Michael Loipführer - initial code
     *
     * @tparam detectorDescriptor_t
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     *
     */
    template <typename detectorDescriptor_t, typename data_t = real_t>
    class SubsetSampler : public Cloneable<SubsetSampler<detectorDescriptor_t, data_t>>
    {
    public:
        /// enum to differentiate between different subset sampling strategies
        enum class SamplingStrategy {
            ROUND_ROBIN,   /// (default) divides data points into subsets via simple round-robin
            EQUI_ROTATION, /// equally spaces the data points based on their rotation
        };

        /**
         * @brief Constructor for SubsetSampler
         *
         * @param[in] volumeDescriptor of the problem
         * @param[in] detectorDescriptor describes the geometry and trajectory of the measurements
         * @param[in] sinogram is the actual measurement data
         * @param[in] nSubsets is number of subsets that should be generated
         * @param[in] samplingStrategy the strategy with which to sample the subsets
         */
        SubsetSampler(const VolumeDescriptor& volumeDescriptor,
                      const detectorDescriptor_t& detectorDescriptor, index_t nSubsets,
                      SamplingStrategy samplingStrategy = SamplingStrategy::ROUND_ROBIN);

        /// default destructor
        ~SubsetSampler() = default;

        /**
         * @brief return a new data descriptor with a BlockDescriptor containing the reordered
         * sinogram data in each block corresponding to a subset
         *
         * @param[in] sinogram the original sinogram
         */
        DataContainer<data_t> getPartitionedData(const DataContainer<data_t>& sinogram);

        /**
         * @brief return the full projector
         */
        template <typename projector_t>
        std::unique_ptr<LinearOperator<data_t>> getProjector()
        {
            return std::make_unique<projector_t>(_volumeDescriptor, _fullDetectorDescriptor);
        }

        /**
         * @brief return a list of projectors that correspond to each subset
         */
        template <typename projector_t>
        std::vector<std::unique_ptr<LinearOperator<data_t>>> getSubsetProjectors()
        {
            std::vector<std::unique_ptr<LinearOperator<data_t>>> projectors;

            for (const auto& detectorDescriptor : _detectorDescriptors) {
                projectors.emplace_back(
                    std::make_unique<projector_t>(_volumeDescriptor, detectorDescriptor));
            }

            return projectors;
        }

    protected:
        /// default copy constructor for cloning
        SubsetSampler<detectorDescriptor_t, data_t>(
            const SubsetSampler<detectorDescriptor_t, data_t>& other);
        /// implement the polymorphic comparison operation
        bool isEqual(const SubsetSampler<detectorDescriptor_t, data_t>& other) const override;

        /// implement the polymorphic clone operation
        SubsetSampler<detectorDescriptor_t, data_t>* cloneImpl() const override;

    private:
        /// mapping of data point indices to respective subsets
        std::vector<std::vector<index_t>> _indexMapping;

        /// volume descriptor of the problem
        VolumeDescriptor _volumeDescriptor;

        /// the full detector descriptor of the problem
        detectorDescriptor_t _fullDetectorDescriptor;

        /// list of detector descriptors corresponding to each block
        std::vector<detectorDescriptor_t> _detectorDescriptors;

        /// number of subsets
        index_t _nSubsets;
    };
} // namespace elsa
