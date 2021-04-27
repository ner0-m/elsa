#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa
{
    // TODO inheritance makes more sense here?
    template <typename data_t = real_t>
    class ConeAdaptedDiscreteShearlet //: Shearlet<data_t>
    {
        // image to wavefront?
        // SH: R ^ n^2 -> R ^ J x n x n
        DataContainer<data_t> transform(DataContainer<data_t> image, std::vector<int> mPrime, int j,
                                        int k, std::vector<int> m, int jTilde, int kTilde,
                                        std::vector<int> mTilde)
        {
            // what exactly is R (Radon transform) and eta (measurement noise)?
            // also, f = an n x n image vectorized? f in R^n^2?

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

        // wavefront to image?
        // SH^-1: R ^ J x n x n -> R ^ n^2
        DataContainer<data_t> inverseTransform(DataContainer<data_t> wavefront); // wavefront?

        // 3 functions of cone-adapted shearlets

        DataContainer<data_t> phi(std::vector<int> mPrime); // m should have size 2

        DataContainer<data_t> psi(int j, int k, std::vector<int> m);

        DataContainer<data_t> psiTilde(int jTilde, int kTilde, std::vector<int> mTilde);

        // Scone = {(a,s,t): a ∈ (0,1], |s| ≤ 1+a^1/2, t ∈ R^2}.

        // NN: R ^ J x n x n -> R ^ J x n x n
    };
} // namespace elsa
