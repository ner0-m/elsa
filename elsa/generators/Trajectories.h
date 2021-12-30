#pragma once

#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "StrongTypes.h"

namespace elsa::trajectories
{
    struct Distance {
        explicit Distance(real_t distance) : distance_(distance) {}
        Distance(real_t distance, real_t scale) : distance_(distance), scale_{scale} {}

        std::pair<real_t, real_t> operator()() const { return {distance_, scale_ * distance_}; }

    private:
        real_t distance_;
        real_t scale_ = {100};
    };

    /// Strong type to define an circle segments or arc with a number of sampling points
    struct SampledArc {
        /// Default construct a full circle with 360 sampling points
        constexpr SampledArc() = default;

        /// Construct an arc of `segment` degree, with a 1 degree sample distance
        constexpr explicit SampledArc(geometry::Degree segment)
            : segment_(static_cast<index_t>(segment)), samples_(static_cast<index_t>(segment))
        {
        }

        /// Construct an arc of `segment` degree and a user defined number of samples
        constexpr SampledArc(geometry::Degree segment, index_t samples)
            : segment_(static_cast<index_t>(segment)), samples_(samples)
        {
        }

        /// Return a pair of the segments and samples
        constexpr std::pair<index_t, index_t> operator()() const { return {segment_, samples_}; }

        constexpr friend bool operator==(SampledArc lhs, SampledArc rhs)
        {
            return lhs.segment_ == rhs.segment_ && lhs.samples_ == rhs.samples_;
        }

        constexpr friend bool operator!=(SampledArc lhs, SampledArc rhs) { return !(lhs == rhs); }

    private:
        index_t segment_ = {360};
        index_t samples_ = {360};
    };

    /// @brief Construct the geometry for a circular 2 or 3 dimensional trajectory. The returned
    /// PlanarDetectorDescriptor should be used to construct a sinogram.
    ///
    /// @param volumeDescriptor descriptor for the volume
    /// @param arc size of arc and number of samples used
    /// @param distance distance between source and object center, and object center and detector
    /// center
    PlanarDetectorDescriptor circular(const DataDescriptor& volumeDescriptor, SampledArc arc,
                                      Distance distance);

    /// @param volumeDescriptor descriptor for the volume
    /// @param arc size of arc and number of samples used
    /// @overload
    PlanarDetectorDescriptor circular(const DataDescriptor& volumeDescriptor, SampledArc arc = {});

    /// @brief Overload to provide an explicit trajectory using a full circle
    ///
    /// @param volumeDescriptor descriptor for the volume
    ///
    /// @overload
    PlanarDetectorDescriptor fullCircle(const DataDescriptor& volDesc);

    /// @brief Overload to provide an explicit trajectory using a circle segment of 180 degree
    ///
    /// @param volumeDescriptor descriptor for the volume
    ///
    /// @overload
    PlanarDetectorDescriptor halfCircle(const DataDescriptor& volDesc);

    /// @brief Overload to provide an explicit trajectory using a full circle, but with fewer number
    /// of samples
    ///
    /// @param volumeDescriptor descriptor for the volume
    /// @param numSamples number of samples used for the full circle (default 90)
    ///
    /// @overload
    PlanarDetectorDescriptor sparseCircle(const DataDescriptor& volDesc, index_t numSamples = 90);
} // namespace elsa::trajectories
