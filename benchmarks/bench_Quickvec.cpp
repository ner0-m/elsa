#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch2/catch.hpp"
#include "Vector.cuh"
#include "Eigen/Dense"
#include <cublas_v2.h>

using namespace quickvec;

// for general problems, should be maximum 256 to avoid memory issues
static size_t SIZE = 128;
// for memory critical problems, using only two vectors
static size_t SIZE_2 = 512;

__global__ void computeDirect(size_t n, float* dc1, float a, float b, float* result)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride) {
        result[i] = dc1[i] * a - dc1[i] / dc1[i] + b * dc1[i];
    }
}

__global__ void computeDirect2(size_t n, float* dc1, float* dc2, float* result)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride) {
        result[i] = dc1[i] * dc2[i];
    }
}

__global__ void computeDirect3(size_t n, float* dc1, float a, float* result)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride) {
        result[i] = dc1[i] * a - dc1[i] / (1 + dc1[i]);
    }
}

__global__ void computeSaxpy(size_t n, float* dc1, float a, float* result)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride) {
        result[i] = dc1[i] * a + result[i];
    }
}

TEST_CASE("CUDA expression benchmark using Eigen with n=" + std::to_string(SIZE) + "^3")
{
    size_t size = SIZE * SIZE * SIZE;
    unsigned int blockSize = 256;
    unsigned int numBlocks = static_cast<unsigned int>((size + blockSize - 1) / blockSize);

    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec(size);
    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec2(size);
    Eigen::Matrix<float, Eigen::Dynamic, 1> resultVec(size);

    for (size_t i = 0; i < size; ++i) {
        randVec[static_cast<long>(i)] =
            static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        randVec2[static_cast<long>(i)] =
            static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    }

    Vector dc(randVec);
    Vector dc2(randVec2);
    Vector result(resultVec);

    auto expr = dc * dc2;

    BENCHMARK("Eigen") { resultVec = (randVec.array() * randVec2.array()).matrix(); };

    BENCHMARK("CUDA direct")
    {
        computeDirect2<<<numBlocks, blockSize>>>(size, dc._data.get(), dc2._data.get(),
                                                 result._data.get());
        cudaDeviceSynchronize();
    };

    BENCHMARK("CUDA over ET") { result.eval(expr); };
}

TEST_CASE("CUDA expression benchmark for memory critical with n=" + std::to_string(SIZE_2) + "^3")
{
    size_t size = SIZE_2 * SIZE_2 * SIZE_2;
    unsigned int blockSize = 256;
    unsigned int numBlocks = static_cast<unsigned int>((size + blockSize - 1) / blockSize);

    float a = 1.22f;
    float b = 2.222f;

    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec(size);
    Eigen::Matrix<float, Eigen::Dynamic, 1> resultVec(size);

    for (size_t i = 0; i < size; ++i) {
        randVec[static_cast<long>(i)] =
            static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    }

    Vector dc(randVec);
    Vector result(resultVec);

    auto expr = dc * a - dc / dc + dc * b;
    auto expr2 = dc * dc;
    auto expr3 = dc * a - dc / (1 + dc);

    BENCHMARK("Eigen: dc * dc") { resultVec = (randVec.array() * randVec.array()).matrix(); };

    BENCHMARK("CUDA direct: dc * dc")
    {
        computeDirect2<<<numBlocks, blockSize>>>(size, dc._data.get(), dc._data.get(),
                                                 result._data.get());
        cudaDeviceSynchronize();
    };

    BENCHMARK("CUDA over ET: dc * dc") { result.eval(expr2); };

    BENCHMARK("Eigen: dc * a - dc / dc + dc * b")
    {
        resultVec = (randVec.array() * a).matrix() - (randVec.array() / randVec.array()).matrix()
                    + (b * randVec.array()).matrix();
    };

    BENCHMARK("CUDA direct: dc * a - dc / dc + dc * b")
    {
        computeDirect<<<numBlocks, blockSize>>>(size, dc._data.get(), a, b, result._data.get());
        cudaDeviceSynchronize();
    };

    BENCHMARK("CUDA over ET: dc * a - dc / dc + dc * b") { result.eval(expr); };

    BENCHMARK("Eigen: dc * a - dc / (1 + dc)")
    {
        resultVec =
            (randVec.array() * a).matrix() - (randVec.array() / (1 + randVec.array())).matrix();
    };

    BENCHMARK("CUDA direct: dc * a - dc / (dc + 1)")
    {
        computeDirect3<<<numBlocks, blockSize>>>(size, dc._data.get(), a, result._data.get());
        cudaDeviceSynchronize();
    };

    BENCHMARK("CUDA over ET: dc * a - dc / (dc + 1)") { result.eval(expr3); };
}

TEST_CASE("CUDA SAXPY benchmark for memory critical saxpy with n=" + std::to_string(SIZE_2) + "^3")
{
    size_t size = SIZE_2 * SIZE_2 * SIZE_2;
    unsigned int blockSize = 256;
    unsigned int numBlocks = static_cast<unsigned int>((size + blockSize - 1) / blockSize);

    float a = 1.22f;

    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec(size);
    Eigen::Matrix<float, Eigen::Dynamic, 1> resultVec(size);

    for (size_t i = 0; i < size; ++i) {
        randVec[static_cast<long>(i)] =
            static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        resultVec[static_cast<long>(i)] =
            static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    }

    Vector dc(randVec);
    Vector result(resultVec);

    auto expr = a * dc + result;

    BENCHMARK("Eigen saxpy") { resultVec = (randVec.array() * a + resultVec.array()).matrix(); };

    BENCHMARK("CUDA direct saxpy")
    {
        computeSaxpy<<<numBlocks, blockSize>>>(size, dc._data.get(), a, result._data.get());
        cudaDeviceSynchronize();
    };

    BENCHMARK("CUDA over ET saxpy") { result.eval(expr); };

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("saxpy failed");
        cublasDestroy(handle);
        return;
    }

    BENCHMARK("cuBLAS saxpy")
    {
        stat = cublasSaxpy(handle, size, &a, dc._data.get(), 1, result._data.get(), 1);
        cudaDeviceSynchronize();
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("saxpy failed");
            cublasDestroy(handle);
            return;
        }
    };

    cublasDestroy(handle);
}

TEST_CASE("CUDA reductions benchmark with n=" + std::to_string(SIZE_2) + "^3")
{
    size_t size = SIZE_2 * SIZE_2 * SIZE_2;
    unsigned int blockSize = 256;
    unsigned int numBlocks = static_cast<unsigned int>((size + blockSize - 1) / blockSize);

    float a = 1.22f;

    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec(size);
    randVec.setRandom();

    Vector dc(randVec);

    BENCHMARK("Eigen Reduction") { return randVec.sum(); };

    BENCHMARK("CUDA Reduction ") { return dc.sum(); };
}
