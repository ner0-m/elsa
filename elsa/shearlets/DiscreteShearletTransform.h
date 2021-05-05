#pragma once

#include "LinearOperator.h"
#include "VolumeDescriptor.h"
//#include "FourierTransform.h"

namespace elsa
{
    /**
     * @brief Class representing a (regular) Discrete Shearlet Transform
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * References:
     * https://www.math.uh.edu/~dlabate/SHBookIntro.pdf
     */
    template <typename data_t = real_t>
    class DiscreteShearletTransform : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for a (regular) discrete shearlet transform.
         *
         * @param[in] descriptor DataDescriptor describing the domain and the range of the operator
         * @param[in] scaleFactor the scalar factor to scale with
         */
        // TODO what are the shearlet system inputs? infer here max of these values and span those
        //  index spaces (use different inputs here)
        DiscreteShearletTransform(index_t width, index_t height, index_t scales);
        // consider scale index j, the orientation index k, and the position index m.

        /// default destructor
        ~DiscreteShearletTransform() override = default;

    protected:
        // image to wavefront?
        // SH: R ^ n^2 -> R ^ J x n x n
        // dot product of the image and psi(j, k, m)?
        /// apply the (regular) discrete shearlet transform
        void applyImpl(const DataContainer<data_t>& f, DataContainer<data_t>& SHf) const override;

        // wavefront to image?
        // SH^-1: R ^ J x n x n -> R ^ n^2 // TODO is SH orthogonal i.e. SH^-1 = SH^T?
        /// apply the adjoint of the (regular) discrete shearlet transform // TODO rename y to g?
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& SHty) const override;

        /// implement the polymorphic clone operation
        DiscreteShearletTransform<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        // TODO add here variables that define the shearlet system
        index_t _width;
        index_t _height;
        index_t _numberOfScales;

        // scale index j, the orientation index k, and the position index m.
        /// shearlet generating function
        DataContainer<data_t> psi(int j, int k, std::vector<int> m);
        // m should be of size 2
        // here we could have e.g. 2^(3/4)j * psi(SkA2^j · −m)

        // scale index j, the orientation index k, and the position index m.
        /// shearlet scaling function
        DataContainer<data_t> phi(int j, int k, std::vector<int> m);

        // TODO how to represent the two matrices S shearing matrix and A parabolic scaling matrix?
        //  No LinearOperator offers selectively setting each matrix value?
        //  Created ShearingOperator for the first one, Scaling covers the parabolic scaling
        //  operator and its inverse

        // NN: R ^ J x n x n -> R ^ J x n x n
    };
} // namespace elsa

// additional todos
// rename x to f (maybe y to g?)
