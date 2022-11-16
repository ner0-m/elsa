#include "Blobs.h"
#include "Bessel.h"

namespace elsa
{
    // ------------------------------------------
    // explicit template instantiation
    namespace blobs
    {
        template float blob_evaluate<float>(float, SelfType_t<float>, SelfType_t<float>, index_t);
        template double blob_evaluate<double>(double, SelfType_t<double>, SelfType_t<double>,
                                              index_t);

        template float blob_projected<float>(float, SelfType_t<float>, SelfType_t<float>, index_t);
        template double blob_projected<double>(double, SelfType_t<double>, SelfType_t<double>,
                                               index_t);

        template float blob_projected<float>(float);
        template double blob_projected<double>(double);

        template float blob_derivative_projected<float>(float, SelfType_t<float>, SelfType_t<float>,
                                                        int);
        template double blob_derivative_projected<double>(double, SelfType_t<double>,
                                                          SelfType_t<double>, int);

        template float blob_normalized_derivative_projected<float>(float, SelfType_t<float>,
                                                                   SelfType_t<float>, int);
        template double blob_normalized_derivative_projected<double>(double, SelfType_t<double>,
                                                                     SelfType_t<double>, int);

    } // namespace blobs

    template class Blob<float>;
    template class Blob<double>;
} // namespace elsa
