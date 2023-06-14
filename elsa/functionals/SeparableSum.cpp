#include "SeparableSum.h"
#include "Functional.h"
#include "IdenticalBlocksDescriptor.h"
#include "TypeCasts.hpp"
#include <memory>

namespace elsa
{
    template <class data_t>
    SeparableSum<data_t>::SeparableSum(std::vector<std::unique_ptr<Functional<data_t>>> fns)
        : Functional<data_t>(*detail::determineDescriptor(fns)), fns_(std::move(fns))
    {
    }

    template <class data_t>
    SeparableSum<data_t>::SeparableSum(const Functional<data_t>& fn)
        : Functional<data_t>(IdenticalBlocksDescriptor(1, fn.getDomainDescriptor()))
    {
        fns_.push_back(fn.clone());
    }

    template <class data_t>
    SeparableSum<data_t>::SeparableSum(const Functional<data_t>& fn1, const Functional<data_t>& fn2)
        : SeparableSum<data_t>(detail::make_vector<data_t>(fn1, fn2))
    {
    }

    template <class data_t>
    SeparableSum<data_t>::SeparableSum(const Functional<data_t>& fn1, const Functional<data_t>& fn2,
                                       const Functional<data_t>& fn3)
        : SeparableSum<data_t>(detail::make_vector<data_t>(fn1, fn2, fn3))
    {
    }

    template <class data_t>
    data_t SeparableSum<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        if (!is<BlockDescriptor>(Rx.getDataDescriptor())) {
            throw Error("SeparableSum: Blocked DataContainer expected");
        }

        if (Rx.getDataDescriptor() != this->getDomainDescriptor()) {
            throw Error("SeparableSum: Descriptor of argument is unexpected");
        }

        auto& blockdesc = downcast_safe<BlockDescriptor>(Rx.getDataDescriptor());

        data_t sum{0};
        for (int i = 0; i < blockdesc.getNumberOfBlocks(); ++i) {
            sum += fns_[asUnsigned(i)]->evaluate(Rx.getBlock(i));
        }
        return sum;
    }

    template <class data_t>
    void SeparableSum<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                               DataContainer<data_t>& out)
    {
        if (!is<BlockDescriptor>(Rx.getDataDescriptor())) {
            throw Error("SeparableSum: Blocked DataContainer expected for gradient");
        }

        if (Rx.getDataDescriptor() != this->getDomainDescriptor()) {
            throw Error("SeparableSum: Descriptor of argument is unexpected");
        }

        auto& blockdesc = downcast_safe<BlockDescriptor>(Rx.getDataDescriptor());

        for (int i = 0; i < blockdesc.getNumberOfBlocks(); ++i) {
            auto outview = out.getBlock(i);
            fns_[asUnsigned(i)]->getGradient(Rx.getBlock(i), outview);
        }
    }

    template <class data_t>
    LinearOperator<data_t> SeparableSum<data_t>::getHessianImpl(const DataContainer<data_t>&)
    {
        throw NotImplementedError("SeparableSum: Hessian not implemented");
    }

    template <class data_t>
    SeparableSum<data_t>* SeparableSum<data_t>::cloneImpl() const
    {
        std::vector<std::unique_ptr<Functional<data_t>>> copyfns;
        for (std::size_t i = 0; i < fns_.size(); ++i) {
            copyfns.push_back(fns_[i]->clone());
        }

        return new SeparableSum<data_t>(std::move(copyfns));
    }

    template <class data_t>
    bool SeparableSum<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other)) {
            return false;
        }

        auto* fn = downcast<SeparableSum<data_t>>(&other);
        return std::equal(fns_.begin(), fns_.end(), fn->fns_.begin(),
                          [](const auto& l, const auto& r) { return (*l) == (*r); });
    }

    namespace detail
    {
        template <class data_t>
        std::unique_ptr<BlockDescriptor>
            determineDescriptor(const std::vector<std::unique_ptr<Functional<data_t>>>& fns)
        {
            // For now assume non empty
            auto& firstDesc = fns.front()->getDomainDescriptor();

            // Determine if all descriptors are equal
            bool allEqual = std::all_of(fns.begin(), fns.end(), [&](const auto& x) {
                return x->getDomainDescriptor() == firstDesc;
            });

            // Then we can return an identical block descriptor
            if (allEqual) {
                return std::make_unique<IdenticalBlocksDescriptor>(fns.size(), firstDesc);
            }

            // There are different descriptors, so extract them from the vector of functionals
            std::vector<std::unique_ptr<DataDescriptor>> descriptors;
            descriptors.reserve(fns.size());
            for (const auto& f : fns) {
                descriptors.push_back(f->getDomainDescriptor().clone());
            }

            return std::make_unique<RandomBlocksDescriptor>(std::move(descriptors));
        }
    } // namespace detail

    // ------------------------------------------
    // explicit template instantiation
    template class SeparableSum<float>;
    template class SeparableSum<double>;

    template std::unique_ptr<BlockDescriptor> detail::determineDescriptor<float>(
        const std::vector<std::unique_ptr<Functional<float>>>& fns);
    template std::unique_ptr<BlockDescriptor> detail::determineDescriptor<double>(
        const std::vector<std::unique_ptr<Functional<double>>>& fns);
} // namespace elsa
