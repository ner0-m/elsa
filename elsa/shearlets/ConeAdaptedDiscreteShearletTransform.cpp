#include "ConeAdaptedDiscreteShearletTransform.h"
#include "VolumeDescriptor.h"

namespace elsa
{
    // TODO calculate eta here, 61 is a dummy value
    template <typename data_t>
    ConeAdaptedDiscreteShearletTransform<data_t>::ConeAdaptedDiscreteShearletTransform(
        index_t width, index_t height)
        : LinearOperator<data_t>(
            VolumeDescriptor{{width, height}},
            VolumeDescriptor{
                {static_cast<index_t>(std::pow(2, (static_cast<index_t>(std::floor(
                                                       1 / 2 * std::log2(std::max(width, height))))
                                                   + 2))
                                      - 3),
                 width, height}})
    {
        // TODO currently only solving for square images with odd sizes?
        if (width != height || width % 2 == 0 || height % 2 == 0) {
            throw InvalidArgumentError("ConeAdaptedDiscreteShearletTransform: currently only "
                                       "supporting square images with odd sizes");
        }
        // TODO precompute the spectra here?
    }

    template <typename data_t>
    void ConeAdaptedDiscreteShearletTransform<data_t>::applyImpl(const DataContainer<data_t>& f,
                                                                 DataContainer<data_t>& SHf) const
    {
        index_t width = this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t height = this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[1];

        // TODO SHf shape here should be zeros([eta, width, height], dtype=complex)

        // TODO is data_t a good choice here? what about index_t? what about std::complex<real_t>

        auto jZero = static_cast<index_t>(std::floor(1 / 2 * std::log2(std::max(width, height))));
        auto eta = static_cast<index_t>(std::pow(2, (jZero + 2)) - 3);

        DataContainer<std::complex<real_t>> fftImg = fft.fft2(f);

        index_t i = 0;

        // TODO do DataContainers support negative indexing?

        for (index_t j = 0; j < jZero; j++) {
            for (auto k = static_cast<index_t>(std::pow(-2, j));
                 k < static_cast<index_t>(std::pow(2, j)) + 1; k++) {
                DataContainer<std::complex<real_t>> tempSHSectionh(
                    VolumeDescriptor{{width, height}});
                tempSHSectionh = 0;
                DataContainer<std::complex<real_t>> tempSHSectionv(
                    VolumeDescriptor{{width, height}});
                tempSHSectionv = 0;
                DataContainer<std::complex<real_t>> tempSHSectionhxv(
                    VolumeDescriptor{{width, height}});
                tempSHSectionhxv = 0;
                for (auto w1 = static_cast<int>(-std::floor(width / 2));
                     w1 < static_cast<int>(std::ceil(width / 2)); w1++) {
                    for (auto w2 = static_cast<int>(-std::floor(height / 2));
                         w2 < static_cast<int>(std::ceil(height / 2)); w2++) {
                        if (std::abs(k) <= static_cast<index_t>(std::pow(2, j)) - 1) {
                            tempSHSectionh[w1, w2] =
                                psiHat(std::pow(4, -j) * w1,
                                       std::pow(4, -j) * k * w1 + std::pow(2, -j) * w2)
                                * fftImg[w1][w2];
                            tempSHSectionv[w1, w2] =
                                psiHat(std::pow(4, -j) * w2,
                                       std::pow(4, -j) * k * w2 + std::pow(2, -j) * w1)
                                * fftImg[w1][w2];
                        } else if (std::abs(k) == static_cast<index_t>(std::pow(2, j))) {
                            tempSHSectionhxv[w1, w2] =
                                3
                                * psiHat(std::pow(4, -j) * w1,
                                         std::pow(4, -j) * k * w1 + std::pow(2, -j) * w2)
                                * fftImg[w1][w2];
                        }
                    }
                }
                if (std::abs(k) <= static_cast<index_t>(std::pow(2, j)) - 1) {
                    DataContainer<std::complex<real_t>> SHSectionh = fft.ifft2(tempSHSectionh);
                    // TODO check here
                    SHf[i] = SHSectionh; // TODO actually SHSectionh really should only have reals
                    i += 1;

                    DataContainer<std::complex<real_t>> SHSectionv = fft.ifft2(tempSHSectionv);
                    // TODO check here
                    SHf[i] = SHSectionv;
                    i += 1;
                } else if (std::abs(k) == static_cast<index_t>(std::pow(2, j))) {
                    DataContainer<std::complex<real_t>> SHSectionhxv = fft.ifft2(tempSHSectionhxv);
                    // TODO check here
                    SHf[i] = SHSectionhxv;
                    i += 1;
                }
            }
        }

        DataContainer<std::complex<real_t>> tempSHSectionZero(VolumeDescriptor{{width, height}});
        tempSHSectionZero = 0;

        for (int w1 = static_cast<int>(-std::floor(width / 2)); w1 < std::ceil(width / 2); w1++) {
            for (int w2 = static_cast<int>(-std::floor(height / 2)); w2 < std::ceil(height / 2);
                 w2++) {
                tempSHSectionZero[w1, w2] = phiHat(w1, w2) * fftImg[w1, w2];
            }
        }

        DataContainer<std::complex<real_t>> SHSectionZero = fft.ifft2(tempSHSectionZero);
        SHf[i] = SHSectionZero;

        printf("Finished shearlet transform");
        //            return SHf;
    }

    template <typename data_t>
    void ConeAdaptedDiscreteShearletTransform<data_t>::applyAdjointImpl(
        const DataContainer<data_t>& y, DataContainer<data_t>& SHty) const
    {
        index_t width = this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t height = this->getDomainDescriptor().getNumberOfCoefficientsPerDimension()[1];

        auto jZero = static_cast<index_t>(std::floor(1 / 2 * std::log2(std::max(width, height))));

        DataContainer<std::complex<real_t>> fHat(VolumeDescriptor{{width, height}});
        fHat = 0;

        // NB sampling from is [-255, 255] x [-255, 255]

        index_t i = 0;
        // [0, ..., j0 - 1]
        for (index_t j = 0; j < jZero; j++) {
            // [- 2 ^ j, ..., 2 ^ j]
            for (auto k = static_cast<index_t>(std::pow(-2, j));
                 k < static_cast<index_t>(std::pow(2, j) + 1); k++) {
                DataContainer<std::complex<real_t>> tempSHSectionh(
                    VolumeDescriptor{{width, height}});
                tempSHSectionh = 0;
                DataContainer<std::complex<real_t>> tempSHSectionv(
                    VolumeDescriptor{{width, height}});
                tempSHSectionv = 0;
                DataContainer<std::complex<real_t>> tempSHSectionhxv(
                    VolumeDescriptor{{width, height}});
                tempSHSectionhxv = 0;
                // [- floor(M / 2), ..., ceil(M / 2) - 1]
                for (auto w1 = static_cast<int>(-std::floor(width / 2));
                     w1 < static_cast<int>(std::ceil(width / 2)); w1++) {
                    // [- floor(N / 2), ..., ceil(N / 2) - 1]
                    for (auto w2 = static_cast<int>(-std::floor(height / 2));
                         w2 < static_cast<int>(std::ceil(height / 2)); w2++) {
                        if (std::abs(k) <= static_cast<index_t>(std::pow(2, j) - 1)) {
                            // section of horizontal cone
                            tempSHSectionh[w1, w2] =
                                psiHat(std::pow(4, -j) * w1,
                                       std::pow(4, -j) * k * w1 + std::pow(2, -j) * w2);
                            // section of vertical cone
                            tempSHSectionv[w1, w2] =
                                psiHat(std::pow(4, -j) * w2,
                                       std::pow(4, -j) * k * w2 + std::pow(2, -j) * w1);
                        } else if (std::abs(k) == static_cast<index_t>(std::pow(2, j))) {
                            // section of the seam lines
                            tempSHSectionhxv[w1, w2] =
                                psiHat(std::pow(4, -j) * w1,
                                       std::pow(4, -j) * k * w1 + std::pow(2, -j) * w2);
                        }
                    }
                }
                if (std::abs(k) <= static_cast<index_t>(std::pow(2, j - 1))) {
                    DataContainer<std::complex<real_t>> SHtSectionh =
                        fft.ifft2(fft.fft2(SHf[i]) * tempSHSectionh);
                    i += 1;
                    DataContainer<std::complex<real_t>> SHtSectionv =
                        fft.ifft2(fft.fft2(SHf[i]) * tempSHSectionv);
                    i += 1;
                    fHat += SHtSectionh + SHtSectionv;
                } else if (std::abs(k) == static_cast<index_t>(std::pow(2, j))) {
                    DataContainer<std::complex<real_t>> SHtSecctionhxv =
                        fft.ifft2(fft.fft2(SHf[i]) * tempSHSectionhxv);
                    i += 1;
                    fHat += SHtSecctionhxv;
                }
            }
        }

        // section of the low frequency
        DataContainer<std::complex<real_t>> tempSHtcSectionZero(VolumeDescriptor{{width, height}});
        tempSHtcSectionZero = 0;

        for (auto w1 = static_cast<int>(-std::floor(width / 2));
             w1 < static_cast<int>(std::ceil(width / 2)); w1++) {
            for (auto w2 = static_cast<int>(-std::floor(height / 2));
                 w2 < static_cast<int>(std::ceil(height / 2)); w2++) {
                tempSHtcSectionZero[w1, w2] = phiHat(w1, w2);
            }
        }

        DataContainer<std::complex<real_t>> SHtSectionZero = SHf[i] * fft.fft2(tempSHtcSectionZero);
        fHat += SHtSectionZero;

        // try fft.fftshift(fHat) ?
        // TODO this is the reconstruction, is it?
        f = fft.ifft2(fHat);
        printf("Finished inverse shearlet transform in"); // SH^T == SH^-1?
        return fHat;
    }

    template <typename data_t>
    ConeAdaptedDiscreteShearletTransform<data_t>*
        ConeAdaptedDiscreteShearletTransform<data_t>::cloneImpl() const
    {
    }

    template <typename data_t>
    bool ConeAdaptedDiscreteShearletTransform<data_t>::isEqual(
        const LinearOperator<data_t>& other) const
    {
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ConeAdaptedDiscreteShearletTransform<float>;
    template class ConeAdaptedDiscreteShearletTransform<double>;
    // TODO maybe have only one of these two
    // template class
    // ConeAdaptedDiscreteShearletTransform<std::complex<float>>;
    // template class
    // ConeAdaptedDiscreteShearletTransform<std::complex<double>>;
} // namespace elsa

// TODO previous notes from the applyImpl method

// what exactly is R (Radon transform) and y (measurement noise)?
// also, f = an n x n image vectorized? f ∈ R^n^2?

// TODO are the images only grayscale or rgb or accept any number of channels?

// TODO insert all these indices to a hash table? memoization?
//  e.g. shearletRepresentation[(m′), (j,k,m), (˜j,˜k,m˜)] =
//  (⟨f,ϕ_m′⟩, ⟨f,ψ_j,k,m⟩, ⟨f,ψ˜_˜j,˜k,m˜⟩)

// TODO
//  memoization also would be important when doing the inverseTransform as we would
//  simply transpose it and apply it to the input, assuming that SH^-1 = SH^T

// TODO
//  if the shearletRepresentation hash tables will contain the same data thus be the
//  same for all ConeAdaptedDiscreteShearlet objects, make transform and
//  inverseTransform static?

// return (⟨f,ϕ_m′⟩,⟨f,ψ_j,k,m⟩,⟨f,ψ˜_˜j,˜k,m˜⟩),
