#include "ImageDenoisingTask.h"
#include <vector>
#include <memory>
#include <numeric>
#include <random>

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
    DataContainer<data_t> ImageDenoisingTask<data_t>::train(const DataContainer<data_t>& image,
                                                            float downSampleFactor)
    {
        if (downSampleFactor <= 0 || downSampleFactor > 1) {
            throw InvalidArgumentError(
                "ImageDenoisingTask::train: downSampleFactor must be between 0 and 1");
        }

        const auto& imageDescriptor =
            dynamic_cast<const VolumeDescriptor&>(image.getDataDescriptor());
        Patchifier<data_t> patchifier(imageDescriptor, _patchSize, _stride);
        auto patches = patchifier.im2patches(image);
        auto sampledPatches = downSample(patches, downSampleFactor);
        DataContainer<data_t> denoisedPatches(sampledPatches.getDataDescriptor());

        _problem = std::make_unique<DictionaryLearningProblem<data_t>>(sampledPatches, _nAtoms);
        KSVD<data_t> solver(*_problem, _sparsityLevel, _epsilon);
        auto representations = solver.solve(_nIterations);
        const auto& dictionary = solver.getLearnedDictionary();

        index_t nSamples =
            dynamic_cast<const IdenticalBlocksDescriptor&>(denoisedPatches.getDataDescriptor())
                .getNumberOfBlocks();
        for (index_t i = 0; i < nSamples; ++i) {
            denoisedPatches.getBlock(i) = dictionary.apply(representations.getBlock(i));
        }

        return patchifier.patches2im(denoisedPatches);
    }

    template <typename data_t>
    DataContainer<data_t> ImageDenoisingTask<data_t>::denoise(const DataContainer<data_t>& image)
    {
        const auto& imageDescriptor =
            dynamic_cast<const VolumeDescriptor&>(image.getDataDescriptor());
        Patchifier<data_t> patchifier(imageDescriptor, _patchSize, _stride);
        auto patches = patchifier.im2patches(image);
        DataContainer<data_t> denoisedPatches(patches.getDataDescriptor());

        const auto& dict = _problem->getDictionary();
        index_t nSamples =
            dynamic_cast<const IdenticalBlocksDescriptor&>(patches.getDataDescriptor())
                .getNumberOfBlocks();

        for (index_t i = 0; i < nSamples; ++i) {
            RepresentationProblem reprProblem(dict, patches.getBlock(i));
            OrthogonalMatchingPursuit<data_t> omp(reprProblem);
            denoisedPatches.getBlock(i) = dict.apply(omp.solve(_sparsityLevel));
        }

        return patchifier.patches2im(denoisedPatches);
    }

    template <typename data_t>
    DataContainer<data_t>
        ImageDenoisingTask<data_t>::downSample(const DataContainer<data_t>& patches,
                                               float downSampleFactor)
    {
        if (downSampleFactor == 1)
            return patches;

        const auto& patchesDescriptor =
            dynamic_cast<const IdenticalBlocksDescriptor&>(patches.getDataDescriptor());

        index_t nSamplesOld = patchesDescriptor.getNumberOfBlocks();
        index_t nSamplesNew = static_cast<index_t>(downSampleFactor * nSamplesOld);
        /*
            1. Make a list of numbers 0,nSamplesOld
            2. Shuffle said list
            3. Define nSamplesNew = downSampleFactor * nSamplesNew
            4. Create a new DataContainer with nSamplesNewPatches
            5. Take the first nSamplesNew entries from the shuffled list
            6. Use them as indices to copy from old patches to new ones
        */
        std::vector<index_t> indices(nSamplesOld);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        IdenticalBlocksDescriptor desc(nSamplesNew, patchesDescriptor.getDescriptorOfBlock(0));
        DataContainer<data_t> downSampledPatches(desc);
        for (index_t i = 0; i < nSamplesNew; ++i) {
            downSampledPatches.getBlock(i) = patches.getBlock(indices[i]);
        }

        return downSampledPatches;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ImageDenoisingTask<float>;
    template class ImageDenoisingTask<double>;

} // namespace elsa
