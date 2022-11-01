#include "Landweber.h"
#include "Scaling.h"
#include "elsaDefines.h"

namespace elsa
{
    template <typename data_t>
    Landweber<data_t>::Landweber(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                                 SelfType_t<data_t> stepSize)
        : LandweberIteration<data_t>(A, b, stepSize)
    {
    }

    template <typename data_t>
    Landweber<data_t>::Landweber(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : LandweberIteration<data_t>(A, b)
    {
    }

    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem, data_t stepSize)
        : LandweberIteration<data_t>(problem, stepSize)
    {
    }

    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem)
        : LandweberIteration<data_t>(problem)
    {
    }

    template <class data_t>
    std::unique_ptr<LinearOperator<data_t>>
        Landweber<data_t>::setupOperators(const LinearOperator<data_t>& A) const
    {
        return adjoint(A).clone();
    }

    template <typename data_t>
    bool Landweber<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!LandweberIteration<data_t>::isEqual(other))
            return false;

        auto otherSolver = downcast_safe<Landweber<data_t>>(&other);
        return static_cast<bool>(otherSolver);
    }

    template <typename data_t>
    Landweber<data_t>* Landweber<data_t>::cloneImpl() const
    {
        if (this->stepSize_.isInitialized()) {
            return new Landweber(*this->A_, this->b_, *this->stepSize_);
        } else {
            return new Landweber(*this->A_, this->b_);
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Landweber<float>;
    template class Landweber<double>;
} // namespace elsa
