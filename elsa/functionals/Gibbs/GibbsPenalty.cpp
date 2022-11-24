#include "GibbsPenalty.h"
//#include "GibbsUtils.h"
#include "TypeCasts.hpp"

#include <stdexcept>

namespace elsa
{
    namespace Gibbs
    {
        template <typename data_t>
        GibbsPenalty<data_t>::GibbsPenalty(const DataDescriptor& domainDescriptor)
            : Functional<data_t>(domainDescriptor)
        {
        }

        // template <typename data_t>
        // GibbsPenalty<data_t>::GibbsPenalty(const Residual<data_t>& residual)
        //     : Functional<data_t>(residual)
        //{
        // }

        template <typename data_t>
        data_t GibbsPenalty<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
        {
            auto result = static_cast<data_t>(0.0);

            for (index_t i = 0; i < Rx.getSize(); ++i)
                result += allNeighboursSum(Rx, Rx.getDataDescriptor().getCoordinateFromIndex(i));

            return static_cast<data_t>(0.5) * result;
        }

        // TODO: test gradient
        template <typename data_t>
        void
            GibbsPenalty<data_t>::getGradientInPlaceImpl([[maybe_unused]] DataContainer<data_t>& Rx)
        {
            std::function<data_t(data_t, data_t)> psi = [](data_t x, data_t y) { return (x - y); };

            for (index_t i = 0; i < Rx.getSize(); ++i) {
                Rx[i] = allNeighboursSum(Rx, Rx.getDataDescriptor().getCoordinateFromIndex(i), psi);
            }
        }

        // TODO: check Hessian from gradient
        template <typename data_t>
        LinearOperator<data_t>
            GibbsPenalty<data_t>::getHessianImpl([[maybe_unused]] const DataContainer<data_t>& Rx)
        {
            DataContainer<data_t> scaleFactors(Rx.getDataDescriptor());

            std::function<data_t(data_t, data_t)> psi = [](data_t x, data_t y) {
                return static_cast<data_t>(1);
            };

            for (index_t i = 0; i < Rx.getSize(); ++i) {
                scaleFactors[i] =
                    allNeighboursSum(Rx, Rx.getDataDescriptor().getCoordinateFromIndex(i), psi);
            }

            return leaf(Scaling<data_t>(Rx.getDataDescriptor(), scaleFactors));
        }

        template <typename data_t>
        GibbsPenalty<data_t>* GibbsPenalty<data_t>::cloneImpl() const
        {
            return new GibbsPenalty(this->getDomainDescriptor());
        }

        template <typename data_t>
        bool GibbsPenalty<data_t>::isEqual(const Functional<data_t>& other) const
        {
            if (!Functional<data_t>::isEqual(other))
                return false;

            auto otherGP = downcast_safe<GibbsPenalty>(&other);
            if (!otherGP)
                return false;

            return true;
        }

        // ------------------------------------------
        // explicit template instantiation
        template class GibbsPenalty<float>;
        template class elsa::Gibbs::GibbsPenalty<double>;
        template class elsa::Gibbs::GibbsPenalty<complex<float>>;
        template class GibbsPenalty<complex<double>>; //???
    };                                                // namespace Gibbs
} // namespace elsa
