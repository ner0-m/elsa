#include "RepresentationProblem.h"

namespace elsa
{

    template <typename data_t>
    RepresentationProblem<data_t>::RepresentationProblem(const Dictionary<data_t>& D,
                                                         const DataContainer<data_t>& y)
        : Problem<data_t>{L2NormPow2<data_t>{LinearResidual<data_t>{D, y}}}, _signal(y)
    {
        _dict = std::unique_ptr<Dictionary<data_t>>{
            static_cast<Dictionary<data_t>*>(D.clone().release())};
    }

    template <typename data_t>
    const Dictionary<data_t>& RepresentationProblem<data_t>::getDictionary() const
    {
        return *_dict;
    }

    template <typename data_t>
    const DataContainer<data_t>& RepresentationProblem<data_t>::getSignal() const
    {
        return _signal;
    }

    template <typename data_t>
    void RepresentationProblem<data_t>::getGradientImpl(DataContainer<data_t>&)
    {
        throw LogicError(
            "RepresentationProblem::getGradient: Objective function is not differentiable");
    }

    template <typename data_t>
    RepresentationProblem<data_t>* RepresentationProblem<data_t>::cloneImpl() const
    {
        return new RepresentationProblem(*_dict, _signal);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class RepresentationProblem<float>;
    template class RepresentationProblem<double>;

} // namespace elsa
