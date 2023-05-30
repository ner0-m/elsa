#include "Ellipsoid.cuh"

template <int n, typename data_t>
__global__ void ellipsoidKernel(data_t* const __restrict__ sinogram,
                                data_t* const __restrict__ rays, data_t w,
                                Eigen::Vector<data_t, n> c, Eigen::DiagonalMatrix<data_t, n> A,
                                Eigen::Matrix<data_t, n, n> R)
{
    auto index = 0;

    const real_t* const rayOrigin = (real_t*) (rayOrigins + blockIdx.x * originPitch);

    const uint32_t xCoord = sinogramOffsetX + blockDim.x * blockIdx.z + threadIdx.x;

    data_t* sinogramPtr =
        ((data_t*) (sinogram + (blockIdx.x * gridDim.y + blockIdx.y) * sinogramPitch) + xCoord);

    *sinogramPtr = 0;

    // auto o = c - ray.origin();
    // auto d = ray.direction();

    // auto Ro = R * o;
    // auto Rd = R * d;

    // auto alpha = Ro.dot(A * Ro) - 1;
    // auto beta = Rd.dot(A * Ro);
    // auto gamma = Rd.dot(A * Rd);
    // auto discriminant = beta * beta / (alpha * alpha) - gamma / alpha;

    // if (discriminant < 0) {
    //     sinogram[index] = 0;
    // } else {
    //     sinogram[coord] = 2 * sqrt(discriminant) * w;
    // }
}

template <typename data_t, int n>
void renderEllipsoid(dim3 sinogramDims, int threads, data_t* const __restrict__ sinogram,
                     uint64_t sinogramPitch, data_t* const __restrict__ rays, uint64_t rayPitch,
                     Eoid<data_t, n> e)
{
    uint32_t numImageBlocks = sinogramDims.z / threads;
    uint32_t remaining = sinogramDims.z % threads;
    uint32_t offset = numImageBlocks * threads;

    if (numImageBlocks > 0) {
        dim3 grid(sinogramDims.x, sinogramDims.y, numImageBlocks);
        ellipsoidKernel<data_t, n>
            <<<grid, threads>>>(sinogram, sinoGramPitch, rays, rayPitch, e.w, e.c, e.A, e.R);
    }

    if (remaining > 0) {
        cudaStream_t remStream;

        if (cudaStreamCreate(&remStream) != cudaSuccess)
            throw std::logic_error("EllipsoidCUDA: Couldn't create stream for remaining images");

        dim3 grid(sinogramDims.x, sinogramDims.y, 1);
        ellipsoidKernel<data_t, n>
            <<<grid, remaining, 0, remStream>>>(sinogram, rays, e.w, e.c, e.A, e.R);

        if (cudaStreamDestroy(remStream) != cudaSuccess)
            throw std::logic_error("EllipsoidCUDA: Couldn't destroy cudaStream object");
    }
}
