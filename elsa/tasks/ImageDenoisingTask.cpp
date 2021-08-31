#include "ImageDenoisingTask.h"

namespace elsa
{
    template <typename data_t>
    ImageDenoisingTask<data_t>::ImageDenoisingTask(index_t patchSize, index_t stride,
                                                   index_t sparsityLevel, index_t nAtoms,
                                                   index_t nIterations, data_t epsilon)
        : _patchSize(patchSize),
          _stride(stride),
          _sparsityLevel(sparsityLevel),
          _nAtoms(nAtoms),
          _nIterations(nIterations),
          _epsilon(epsilon)
    {
    }

    template <typename data_t>
    DataContainer<data_t> ImageDenoisingTask<data_t>::train(const DataContainer<data_t>& image)
    {
        const auto& imageDescriptor =
            dynamic_cast<const VolumeDescriptor&>(image.getDataDescriptor());
        Patchifier<data_t> patchifier(imageDescriptor, _patchSize, _stride);
        auto patches = patchifier.im2patches(image);

        DictionaryLearningProblem<data_t> dlProblem(patches, _nAtoms);
        KSVD<data_t> solver(dlProblem, _sparsityLevel, _epsilon);
        auto representations = solver.solve(_nIterations);
        const auto& dictionary = solver.getLearnedDictionary();

        DataContainer<data_t> denoised_patches(patches.getDataDescriptor());

        index_t nSamples =
            dynamic_cast<const IdenticalBlocksDescriptor&>(patches.getDataDescriptor())
                .getNumberOfBlocks();
        for (index_t i = 0; i < nSamples; ++i) {
            denoised_patches.getBlock(i) = dictionary.apply(representations.getBlock(i));
        }

        return patchifier.patches2im(denoised_patches);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ImageDenoisingTask<float>;
    template class ImageDenoisingTask<double>;

} // namespace elsa
