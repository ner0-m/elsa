#include "DiscreteShearletTransform.h"

namespace elsa
{
    /// note that we are only dealing with square nxn images (for now at least)
    // TODO the inputs here should be enough to define the entire system
    template <typename data_t>
    DiscreteShearletTransform<data_t>::DiscreteShearletTransform(index_t width, index_t height,
                                                                 index_t numberOfScales)
        // dummy values for the LinearOperator constructor
        : LinearOperator<data_t>(VolumeDescriptor{{width, height}},
                                 VolumeDescriptor{{width, height}}),
          _width{width},
          _height{height},
          _numberOfScales{numberOfScales}
    {
        // sanity check the parameters here
        //        if (scales something) {
        //            throw InvalidArgumentError(
        //                "DiscreteShearletTransform: the allowed number of scales is ... ");
        //        }

        // TODO generate here the system?
        //  this goes against the docs in LinearOperator: "Hence any pre-computations/caching should
        //  only be done in a lazy manner (e.g. during the first call of apply), and not in the
        //  constructor."

        /// DataContainer<data_t> _SH; //declare in the header file?
        /// _SH = // fancy magic goes here
        /// _SH shape is J x n^2 x n^2
    }

    template <typename data_t>
    void DiscreteShearletTransform<data_t>::applyImpl(const DataContainer<data_t>& f,
                                                      DataContainer<data_t>& SHf) const
    {
        // TODO should image vectorization be done here or beforehand? f is input as an nxn image or
        //  n^2 vectorized

        // TODO current plan:
        //  fft on the image (FFT: nxn -> nxn ?)
        //  fftshift on the frequency domain components  (FFT_SHIFT: nxn -> nxn ?)
        //  apply the shearlet generator/scaling functions

        // FourierTransform<data_t> ft(VolumeDescriptor{{_width, _height}});
        // DataContainer<data_t> F = ft.fft2D(f); // if we're applying FFT 2D, make f to be nxn
        // F = ft.fftShift2D(F);
        // DataContainer<data_t> SHF = _SH.apply(F); // ?
        // SHF = _SH.ifftShift2D(F); // ?
        // DataContainer<data_t> SHf = _SH.ifft2D(SHF); // ?
        // return SHf; // (either as n^2xJ or nxnxJ)

        // TODO should image de-vectorization be done here or afterwards? SHf is output as an n^2xJ
        //  or nxnxJ de-vectorized
    }

    template <typename data_t>
    void DiscreteShearletTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                             DataContainer<data_t>& SHty) const
    {
    }

    template <typename data_t>
    DiscreteShearletTransform<data_t>* DiscreteShearletTransform<data_t>::cloneImpl() const
    {
    }

    template <typename data_t>
    bool DiscreteShearletTransform<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
    }

    template <typename data_t>
    DataContainer<data_t> DiscreteShearletTransform<data_t>::psi(int j, int k, std::vector<int> m)
    {
    }

    template <typename data_t>
    DataContainer<data_t> DiscreteShearletTransform<data_t>::phi(int j, int k, std::vector<int> m)
    {
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DiscreteShearletTransform<float>;
    template class DiscreteShearletTransform<double>;
    // TODO what about complex types
} // namespace elsa
