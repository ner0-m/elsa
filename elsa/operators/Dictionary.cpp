#include "Dictionary.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    Dictionary<data_t>::Dictionary(const DataDescriptor& signalDescriptor, index_t nAtoms)
        : LinearOperator<data_t>(VolumeDescriptor({nAtoms}), signalDescriptor),
          _dictionary{DataContainer<data_t>(generateInitialData(signalDescriptor, nAtoms))},
          _nAtoms{nAtoms}
    {
    }

    template <typename data_t>
    Dictionary<data_t>::Dictionary(const DataContainer<data_t>& dictionary)
        : LinearOperator<data_t>(
            VolumeDescriptor({getIdenticalBlocksDescriptor(dictionary).getNumberOfBlocks()}),
            getIdenticalBlocksDescriptor(dictionary).getDescriptorOfBlock(0)),
          _dictionary{dictionary},
          _nAtoms{getIdenticalBlocksDescriptor(dictionary).getNumberOfBlocks()}
    {
        for (int i = 0; i < _nAtoms; ++i) {
            auto block = _dictionary.getBlock(i);
            data_t l2Norm = block.l2Norm();
            if (l2Norm == 0) {
                throw InvalidArgumentError("Dictionary: initializing with 0-atom not possible");
            }
            // don't normalize if the norm is very close to 1 already
            if (std::abs(l2Norm - 1)
                > std::numeric_limits<data_t>::epsilon() * std::abs(l2Norm + 1)) {
                block /= l2Norm;
            }
        }
    }

    template <typename data_t>
    const IdenticalBlocksDescriptor&
        Dictionary<data_t>::getIdenticalBlocksDescriptor(const DataContainer<data_t>& data)
    {
        try {
            return downcast_safe<IdenticalBlocksDescriptor>(data.getDataDescriptor());
        } catch (const BadCastError&) {
            throw InvalidArgumentError(
                "Dictionary: cannot initialize from data without IdenticalBlocksDescriptor");
        }
    }

    template <typename data_t>
    DataContainer<data_t>
        Dictionary<data_t>::generateInitialData(const DataDescriptor& signalDescriptor,
                                                index_t nAtoms)
    {
        Vector_t<data_t> randomData(signalDescriptor.getNumberOfCoefficients() * nAtoms);
        randomData.setRandom();
        IdenticalBlocksDescriptor desc(nAtoms, signalDescriptor);
        DataContainer<data_t> initData{desc, randomData};

        for (int i = 0; i < desc.getNumberOfBlocks(); ++i) {
            auto block = initData.getBlock(i);
            block /= block.l2Norm();
        }

        return initData;
    }

    template <typename data_t>
    void Dictionary<data_t>::updateAtom(index_t j, const DataContainer<data_t>& atom)
    {
        if (j < 0 || j >= _nAtoms)
            throw InvalidArgumentError("Dictionary::updateAtom: atom index out of bounds");
        if (*_rangeDescriptor != atom.getDataDescriptor())
            throw InvalidArgumentError("Dictionary::updateAtom: atom has invalid size");
        data_t l2Norm = atom.l2Norm();
        if (l2Norm == 0) {
            throw InvalidArgumentError("Dictionary::updateAtom: updating to 0-atom not possible");
        }
        // don't normalize if the norm is very close to 1 already
        _dictionary.getBlock(j) =
            (std::abs(l2Norm - 1) < std::numeric_limits<data_t>::epsilon() * std::abs(l2Norm + 1))
                ? atom
                : (atom / l2Norm);
    }

    template <typename data_t>
    void Dictionary<data_t>::updateAtoms(const DataContainer<data_t>& atoms)
    {
        if (atoms.getDataDescriptor() != _dictionary.getDataDescriptor()) {
            throw InvalidArgumentError(
                "Atoms have to have same DataDescriptor as on initialization");
        }
        DataContainer<data_t> tmp = atoms;

        for (int i = 0; i < _nAtoms; ++i) {
            auto block = tmp.getBlock(i);
            data_t l2Norm = block.l2Norm();
            if (l2Norm == 0) {
                throw InvalidArgumentError("Dictionary: initializing with 0-atom not possible");
            }
            // don't normalize if the norm is very close to 1 already
            if (std::abs(l2Norm - 1)
                > std::numeric_limits<data_t>::epsilon() * std::abs(l2Norm + 1)) {
                block /= l2Norm;
            }
        }

        _dictionary = tmp;
    }

    template <typename data_t>
    const DataContainer<data_t> Dictionary<data_t>::getAtom(index_t j) const
    {
        if (j < 0 || j >= _nAtoms)
            throw InvalidArgumentError("Dictionary: atom index out of bounds");
        return _dictionary.getBlock(j);
    }

    template <typename data_t>
    const DataContainer<data_t>& Dictionary<data_t>::getAtoms() const
    {
        return _dictionary;
    }

    template <typename data_t>
    index_t Dictionary<data_t>::getNumberOfAtoms() const
    {
        return _nAtoms;
    }

    template <typename data_t>
    Dictionary<data_t> Dictionary<data_t>::getSupportedDictionary(IndexVector_t support) const
    {
        Dictionary supportDict(*_rangeDescriptor, support.rows());
        index_t j = 0;

        for (const auto& i : support) {
            if (i < 0 || i >= _nAtoms)
                throw InvalidArgumentError(
                    "Dictionary::getSupportedDictionary: support contains out-of-bounds index");

            supportDict.updateAtom(j, getAtom(i));
            ++j;
        }

        return supportDict;
    }

    template <typename data_t>
    void Dictionary<data_t>::applyImpl(const DataContainer<data_t>& x,
                                       DataContainer<data_t>& Ax) const
    {
        Timer timeguard("Dictionary", "apply");

        if (x.getSize() != _nAtoms || Ax.getDataDescriptor() != *_rangeDescriptor)
            throw InvalidArgumentError("Dictionary::apply: incorrect input/output sizes");

        index_t j = 0;
        Ax = 0;

        for (const auto& x_j : x) {
            const auto& atom = getAtom(j);
            Ax += atom * x_j; // vector*scalar

            ++j;
        }
    }

    template <typename data_t>
    void Dictionary<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                              DataContainer<data_t>& Aty) const
    {
        Timer timeguard("Dictionary", "applyAdjoint");

        if (Aty.getSize() != _nAtoms || y.getDataDescriptor() != *_rangeDescriptor) {
            throw InvalidArgumentError("Dictionary::applyAdjoint: incorrect input/output sizes");
        }

        index_t i = 0;
        Aty = 0;
        for (auto& Aty_i : Aty) {
            const auto& atom_i = getAtom(i);
            for (int j = 0; j < atom_i.getSize(); ++j) {
                Aty_i += atom_i[j] * y[j];
            }
            ++i;
        }
    }

    template <typename data_t>
    Dictionary<data_t>* Dictionary<data_t>::cloneImpl() const
    {
        return new Dictionary(_dictionary);
    }

    template <typename data_t>
    bool Dictionary<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherDictionary = downcast_safe<Dictionary>(&other);
        if (!otherDictionary)
            return false;

        if (_dictionary != otherDictionary->_dictionary)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Dictionary<float>;
    template class Dictionary<double>;

} // namespace elsa
