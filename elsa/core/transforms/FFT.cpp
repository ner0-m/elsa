#include "FFT.h"

#ifdef ELSA_CUDA_ENABLED

namespace elsa::detail
{
    cufftResult createPlan(cufftHandle* plan, cufftType type, const IndexVector_t& shape)
    {
        int rank = shape.size();
        if (rank > 3) {
            return CUFFT_INVALID_SIZE;
        }
        long long int dimensions[3];

        for (int i = 0; i < rank; i++) {
            dimensions[i] = shape(i);
        }

        cufftResult result;
        if (unlikely((result = cufftCreate(plan)) != CUFFT_SUCCESS)) {
            /* first ever cufftCreate call incurs overhead, performs GPU allocations.
                (may fail, failure is very unlikely though) */
            return result;
        }

        size_t workArea = 0;
        /* rationale for using the cufftMakePlanMany64:
            this is the only function in the cufft API that allows for very large FFTs,
            i.e. with >= 2^31 (or 2^32, the treatment of the sign is not clearly documented)
            elements. When the sizes allow it, cufft will use 32 bit indexes anyway, so we
            do not pay for the 64bit indices when it is not required.

            We do not perform batched FFTs, so the batch number is set to 1, effectively
            making the cufftMakePlanMany64 into cufftMakePlan64 (which does not exist on its own).
         */
        return cufftMakePlanMany64(*plan, rank, dimensions, NULL, 0, 0, NULL, 0, 0, type, 1,
                                   &workArea);
    }

    thread_local CuFFTPlanCache cufftCache;

    void CuFFTPlanCache::flush()
    {
        while (!_cache.empty()) {
            evict();
        }
    }

    void CuFFTPlanCache::evict()
    {
        if (_cache.empty()) {
            return;
        }
        auto& [plan, _, __] = _cache.front();
        cufftDestroy(plan);
        _cache.pop_front();
    }

    CuFFTPlanCache::CuFFTPlanCache() : _limit{ELSA_CUFFT_CACHE_SIZE} {}

    std::optional<cufftHandle> CuFFTPlanCache::get(cufftType type, const IndexVector_t& shape)
    {
        CacheList::iterator it;
        for (it = _cache.begin(); it != _cache.end(); it++) {
            auto& [cachedPlan, cachedShape, cachedType] = *it;
            if (shape.size() == cachedShape.size() && shape == cachedShape && type == cachedType) {
                /* move touched element to end */
                _cache.splice(_cache.end(), _cache, it);
                return cachedPlan;
            }
        }

        /* not cached, must create new one => potentially evict */
        if (_cache.size() >= _limit) {
            evict();
        }

        /* rationale for using the cufftMakePlanMany64:
            this is the only function in the cufft API that allows for very large FFTs,
            i.e. with >= 2^31 (or 2^32, the treatment of the sign is not clearly documented)
            elements. When the sizes allow it, cufft will use 32 bit indexes anyway, so we
            do not pay for the 64bit indices when it is not required.

            We do not perform batched FFTs, so the batch number is set to 1, effectively
            making the cufftMakePlanMany64 into cufftMakePlan64 (which does not exist on its own).
         */
        cufftHandle plan;
        cufftResult planResult;
        if ((planResult = createPlan(&plan, type, shape)) != CUFFT_SUCCESS) {
            /* may fail, e.g. out of GPU memory or dimensions do not
                match the requirements specified in the documentation */
            if (planResult == CUFFT_ALLOC_FAILED) {
                flush();
                /* try again after flushing cache */
                if (createPlan(&plan, type, shape) != CUFFT_SUCCESS) {
                    return std::nullopt;
                }
            } else {
                return std::nullopt;
            }
        }
        _cache.push_back(std::make_tuple(plan, shape, type));
        return plan;
    }
} // namespace elsa::detail

#endif
