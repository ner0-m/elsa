#include "Quadric.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Quadric<data_t>::Quadric(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Functional<data_t>(A.getDomainDescriptor()),
          _linearResidual{A, b}
    {
        // TODO: enable this spd check
//        if (!A.isSpd())
//            throw std::invalid_argument("Quadric: operator A needs to be spd");
    }


    template <typename data_t>
    data_t Quadric<data_t>::_evaluate(const DataContainer<data_t>& Rx)
    {
        auto temp = _linearResidual.getOperator().apply(Rx);

        return static_cast<data_t>(0.5) * Rx.dot(temp) - Rx.dot(_linearResidual.getDataVector());
    }

    template <typename data_t>
    void Quadric<data_t>::_getGradientInPlace(DataContainer<data_t>& Rx)
    {
        auto temp = _linearResidual.getOperator().apply(Rx);
        temp -= _linearResidual.getDataVector();
        Rx = temp;
    }

    template <typename data_t>
    LinearOperator<data_t> Quadric<data_t>::_getHessian(const DataContainer<data_t>& Rx)
    {
        return leaf(_linearResidual.getOperator());
    }


    template <typename data_t>
    Quadric<data_t>* Quadric<data_t>::cloneImpl() const
    {
        return new Quadric<data_t>(_linearResidual.getOperator(), _linearResidual.getDataVector());
    }

    template <typename data_t>
    bool Quadric<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherQuadric = dynamic_cast<const Quadric*>(&other);
        if (!otherQuadric)
            return false;

        if (_linearResidual != otherQuadric->_linearResidual)
            return false;

        return true;
    }


    // ------------------------------------------
    // explicit template instantiation
    template class Quadric<float>;
    template class Quadric<double>;
    template class Quadric<std::complex<float>>;
    template class Quadric<std::complex<double>>;

} // namespace elsa
