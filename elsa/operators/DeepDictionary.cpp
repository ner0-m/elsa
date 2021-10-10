#include "DeepDictionary.h"
#include "TypeCasts.hpp"
#include <algorithm>

namespace elsa
{
    template <typename data_t>
    DeepDictionary<data_t>::DeepDictionary(
        const DataDescriptor& signalDescriptor, const std::vector<index_t>& nAtoms,
        const std::vector<std::function<data_t(data_t)>>& activationFunctions)
        : LinearOperator<data_t>(VolumeDescriptor({nAtoms.back()}), signalDescriptor),
          _nAtoms{nAtoms},
          _activationFunctions{activationFunctions},
          _dictionaries{
              std::move(generateInitialData(signalDescriptor, nAtoms, activationFunctions))}
    {
    }

    template <typename data_t>
    DeepDictionary<data_t>::DeepDictionary(
        const std::vector<Dictionary<data_t>>& dictionaries,
        const std::vector<std::function<data_t(data_t)>>& activationFunctions)
        : LinearOperator<data_t>(dictionaries.back().getDomainDescriptor(),
                                 dictionaries.begin()->getRangeDescriptor()),
          _dictionaries{}, // TODO !! copy ctor
          _activationFunctions{activationFunctions}
    {
        if (_dictionaries.size() - 1 != _activationFunctions.size()) {
            throw InvalidArgumentError("foo");
        }
    }

    template <typename data_t>
    std::vector<Dictionary<data_t>> DeepDictionary<data_t>::generateInitialData(
        const DataDescriptor& signalDescriptor, const std::vector<index_t>& nAtoms,
        const std::vector<std::function<data_t(data_t)>>& activationFunctions)
    {
        if (nAtoms.size() - 1 != activationFunctions.size()) {
            throw InvalidArgumentError("foo");
        }

        std::vector<Dictionary<data_t>> dicts;

        const DataDescriptor* nextSignalDescriptor = &signalDescriptor;

        for (index_t n : nAtoms) {
            Dictionary<data_t> dict(*nextSignalDescriptor, n);
            nextSignalDescriptor = &dict.getDomainDescriptor();
            dicts.push_back(std::move(dict));
        }

        return std::move(dicts);
    }

    template <typename data_t>
    void DeepDictionary<data_t>::applyImpl(const DataContainer<data_t>& x,
                                           DataContainer<data_t>& Ax) const
    {
        Timer timeguard("DeepDictionary", "apply");

        if (x.getDataDescriptor() != *_domainDescriptor
            || Ax.getDataDescriptor() != *_rangeDescriptor)
            throw InvalidArgumentError("DeepDictionary::apply: incorrect input/output sizes");

        DataContainer<data_t> lastResult = x;
        index_t i = _activationFunctions.size() - 1;

        for (auto dict = _dictionaries.end() - 1; dict > _dictionaries.begin(); --dict) {
            lastResult = dict->apply(lastResult);
            std::for_each(lastResult.begin(), lastResult.end(), _activationFunctions.at(i));
            --i;
        }

        Ax = _dictionaries.begin()->apply(lastResult);
    }

    template <typename data_t>
    void DeepDictionary<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                  DataContainer<data_t>& Aty) const
    {

        throw LogicError("idontthinkthereisanyreasonabledefinitionforthis,sryyy");
    }

    template <typename data_t>
    DeepDictionary<data_t>* DeepDictionary<data_t>::cloneImpl() const
    {
        return new DeepDictionary(_dictionaries, _activationFunctions);
    }

    template <typename data_t>
    bool DeepDictionary<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherDeepDictionary = downcast_safe<DeepDictionary>(&other);
        if (!otherDeepDictionary)
            return false;

        if (_dictionaries != otherDeepDictionary->_dictionaries)
            /* we would also like to do this, but std::function can not be compared...
            || _activationFunctions != otherDeepDictionary->_activationFunctions)*/
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DeepDictionary<float>;
    template class DeepDictionary<double>;

} // namespace elsa
