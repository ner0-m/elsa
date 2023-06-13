#include "SIRT.h"
#include "Scaling.h"
#include "elsaDefines.h"

namespace elsa
{
    template <typename data_t>
    SIRT<data_t>::SIRT(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                       SelfType_t<data_t> stepSize)
        : LandweberIteration<data_t>(A, b, stepSize)
    {
    }

    template <typename data_t>
    SIRT<data_t>::SIRT(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : LandweberIteration<data_t>(A, b)
    {
    }

    template <class data_t>
    std::unique_ptr<LinearOperator<data_t>>
        SIRT<data_t>::setupOperators(const LinearOperator<data_t>& A) const
    {
        const auto& domain = A.getDomainDescriptor();
        const auto& range = A.getRangeDescriptor();

        DataContainer<data_t> domOnes(domain);
        domOnes = 1;
        auto rowsum = A.apply(domOnes);

        // Preven division by zero (and hence NaNs), by sligthly lifting everything up a touch
        rowsum += data_t{1e-10f};

        Scaling<data_t> M(rowsum.getDataDescriptor(), data_t{1.} / rowsum);

        DataContainer<data_t> rangeOnes(range);
        rangeOnes = 1;
        auto colsum = A.applyAdjoint(rangeOnes);

        Scaling<data_t> T(colsum.getDataDescriptor(), data_t{1.} / colsum);

        return (T * adjoint(A) * M).clone();
    }

    template <typename data_t>
    bool SIRT<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!LandweberIteration<data_t>::isEqual(other))
            return false;

        auto otherSolver = downcast_safe<SIRT<data_t>>(&other);
        return static_cast<bool>(otherSolver);
    }

    template <typename data_t>
    SIRT<data_t>* SIRT<data_t>::cloneImpl() const
    {
        if (this->stepSize_.isInitialized()) {
            return new SIRT(*this->A_, this->b_, *this->stepSize_);
        } else {
            return new SIRT(*this->A_, this->b_);
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SIRT<float>;
    template class SIRT<double>;
} // namespace elsa
