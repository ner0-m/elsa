#include "ImageDenoisingTask.h"

namespace elsa
{
    template <typename data_t>
    ImageDenoisingTask::ImageDenoisingTask(const DataContainer<data_t>& image, index_t blockSize,
                                           index_t stride, index_t sparsityLevel,
                                           index_t nIterations,
                                           data_t epsilon = std::numeric_limits<data_t>::epsilon())
    {
    }

    template <typename data_t>
    ImageDenoisingTask<data_t>* ImageDenoisingTask<data_t>::cloneImpl() const
    {
        return new ImageDenoisingTask(*this);
    }

    template <typename data_t>
    bool ImageDenoisingTask<data_t>::isEqual(const ImageDenoisingTask<data_t>& other) const
    {
        if (typeid(*this) != typeid(other))
            return false;

        if (_image != other._image)
            return false;

        // what else?

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ImageDenoisingTask<float>;

    template class ImageDenoisingTask<double>;

} // namespace elsa
