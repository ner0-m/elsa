#pragma once

#include <map>

#include "elsaDefines.h"
#include "Graph.h"

namespace elsa::ml
{
    template <typename T>
    class Layer;

    namespace detail
    {
        template <typename data_t>
        class State
        {
        public:
            using GraphType = Graph<Layer<data_t>>;

            static GraphType& getGraph();

            static void reset();

        private:
            static GraphType graph_;
        };

    } // namespace detail
} // namespace elsa::ml