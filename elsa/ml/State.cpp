#include "State.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        typename State<data_t>::GraphType& State<data_t>::getGraph()
        {
            return State<data_t>::graph_;
        }

        template <typename data_t>
        void State<data_t>::reset()
        {
            graph_.clear();
        }

        template <typename data_t>
        typename State<data_t>::GraphType State<data_t>::graph_{State<data_t>::GraphType()};

        template class State<float>;
        template class State<double>;
    } // namespace detail
} // namespace elsa::ml