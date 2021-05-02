#include "ConeAdaptedDiscreteShearletTransform.h"

namespace elsa
{
    template <typename data_t>
    ConeAdaptedDiscreteShearletTransform<data_t>::ConeAdaptedDiscreteShearletTransform(
        std::vector<int> mPrime, int j, int k, std::vector<int> m, int jTilde, int kTilde,
        std::vector<int> mTilde)
        : DiscreteShearletTransform<data_t>(0, 0, 0)
    {
    }

    template <typename data_t>
    void ConeAdaptedDiscreteShearletTransform<data_t>::applyImpl(const DataContainer<data_t>& f,
                                                                 DataContainer<data_t>& SHf) const
    {
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
    }

    template <typename data_t>
    void ConeAdaptedDiscreteShearletTransform<data_t>::applyAdjointImpl(
        const DataContainer<data_t>& y, DataContainer<data_t>& SHty) const
    {
    }

    template <typename data_t>
    DiscreteShearletTransform<data_t>*
        ConeAdaptedDiscreteShearletTransform<data_t>::cloneImpl() const
    {
    }

    template <typename data_t>
    bool ConeAdaptedDiscreteShearletTransform<data_t>::isEqual(
        const LinearOperator<data_t>& other) const
    {
    }

    template <typename data_t>
    DataContainer<data_t> ConeAdaptedDiscreteShearletTransform<data_t>::phi(std::vector<int> mPrime)
    {
    }

    template <typename data_t>
    DataContainer<data_t> ConeAdaptedDiscreteShearletTransform<data_t>::psi(int j, int k,
                                                                            std::vector<int> m)
    {
    }

    template <typename data_t>
    DataContainer<data_t>
        ConeAdaptedDiscreteShearletTransform<data_t>::psiTilde(int jTilde, int kTilde,
                                                               std::vector<int> mTilde)
    {
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ConeAdaptedDiscreteShearletTransform<float>;
    template class ConeAdaptedDiscreteShearletTransform<double>;
    // TODO what about complex types
} // namespace elsa
