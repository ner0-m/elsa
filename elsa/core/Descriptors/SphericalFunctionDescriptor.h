#pragma once

#include "DataDescriptor.h"

namespace elsa
{
    /**
     * @brief Class for metadata of spherical functions
     *
     * @author Max Endrass (endrass@cs.tum.edu), most boilerplate code courtesy of Matthias
     * Wieczorek
     * @author Matthias Wieczore (wieczore@cs.tum.edu), logic merge with former
     * XTTQuadratureDescriptor and minor adjustments switched to 3D coordinates instead of theta/phi
     * @author Nikola Dinev (nikola.dinev@tum.de), port to elsa
     *
     * @tparam data_t data type of weights and directions
     */
    template <typename data_t = real_t>
    class SphericalFunctionDescriptor : public DataDescriptor
    {
    public:
        using DirVec = Eigen::Matrix<data_t, 3, 1>;
        using DirVecList = std::vector<Eigen::Matrix<data_t, 3, 1>>;
        using WeightVec = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    protected:
        DirVecList _dirs;
        WeightVec _weights;

    public:
        /**
         * @brief Constructor for SphericalFunctionDescriptor
         *
         * @param[in] dirs Sampling pattern on the sphere (3D vector on unit sphere)
         * @param[in] weights Weights for each sampling direction (e.g. voronoi weigth)
         */
        SphericalFunctionDescriptor(const DirVecList& dirs, const WeightVec& weights);

        /**
         * @brief Constructor for SphericalFunctionDescriptor
         *
         * @param[in] dirs Sampling pattern on the sphere (3D vectors on unit sphere)
         */
        SphericalFunctionDescriptor(const DirVecList& dirs);

        SphericalFunctionDescriptor* cloneImpl() const override;

        ~SphericalFunctionDescriptor() {}

        /// get single direction
        const DirVec& getIthDir(index_t i) const { return _dirs[i]; }
        /// get directions
        const DirVecList& getDirs() const { return _dirs; }
        /// get single weight
        data_t getIthWeight(std::size_t i) const { return _weights[i]; }
        /// get weights vector
        WeightVec getWeights() const { return _weights; }
    };
} // namespace elsa
