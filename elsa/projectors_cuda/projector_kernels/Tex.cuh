#pragma once

#include "SharedArray.cuh"

/// convenience function for texture fetching
template <typename real_t, uint32_t dim>
__device__ __forceinline__ real_t tex(cudaTextureObject_t texObj,
                                      const elsa::detail::EasyAccessSharedArray<float, dim, 32> p)
{
    if (dim == 3)
        return tex3D<real_t>(texObj, p[0], p[1], p[2]);
    else
        return tex2D<real_t>(texObj, p[0], p[1]);
}

/// fetches double at position (x,y) from 2D texture
__device__ __forceinline__ double tex2Dd(cudaTextureObject_t texObj, const float x, const float y)
{
    uint2 rt = tex2D<uint2>(texObj, x, y);
    return __hiloint2double(rt.y, rt.x);
}

/// template specialization for double texture fetches
template <>
__device__ __forceinline__ double
    tex<double, 2>(cudaTextureObject_t texObj,
                   const elsa::detail::EasyAccessSharedArray<float, 2, 32> p)
{
    float x = p[0] - 0.5f;
    float y = p[1] - 0.5f;

    float i = floor(x);
    float j = floor(y);

    float a = x - i;
    float b = y - j;

    double T[2][2];
    T[0][0] = tex2Dd(texObj, i, j);
    T[1][0] = tex2Dd(texObj, i + 1, j);
    T[0][1] = tex2Dd(texObj, i, j + 1);
    T[1][1] = tex2Dd(texObj, i + 1, j + 1);

    return (1 - a) * (1 - b) * T[0][0] + a * (1 - b) * T[1][0] + (1 - a) * b * T[0][1]
           + a * b * T[1][1];
}

/// fetches double at position (x,y,z) from 3D texture
__device__ __forceinline__ double tex3Dd(cudaTextureObject_t texObj, const float x, const float y,
                                         const float z)
{
    uint2 rt = tex3D<uint2>(texObj, x, y, z);
    return __hiloint2double(rt.y, rt.x);
}

/// template specialization for double texture fetches
template <>
__device__ __forceinline__ double
    tex<double, 3>(cudaTextureObject_t texObj,
                   const elsa::detail::EasyAccessSharedArray<float, 3, 32> p)
{
    float x = p[0] - 0.5f;
    float y = p[1] - 0.5f;
    float z = p[2] - 0.5f;

    float i = floor(x);
    float j = floor(y);
    float k = floor(z);

    float a = x - i;
    float b = y - j;
    float c = z - k;

    double T[2][2][2];
    T[0][0][0] = tex3Dd(texObj, i, j, k);
    T[1][0][0] = tex3Dd(texObj, i + 1, j, k);
    T[0][1][0] = tex3Dd(texObj, i, j + 1, k);
    T[0][0][1] = tex3Dd(texObj, i, j, k + 1);
    T[1][1][0] = tex3Dd(texObj, i + 1, j + 1, k);
    T[1][0][1] = tex3Dd(texObj, i + 1, j, k + 1);
    T[0][1][1] = tex3Dd(texObj, i, j + 1, k + 1);
    T[1][1][1] = tex3Dd(texObj, i + 1, j + 1, k + 1);

    return (1 - a) * (1 - b) * (1 - c) * T[0][0][0] + a * (1 - b) * (1 - c) * T[1][0][0]
           + (1 - a) * b * (1 - c) * T[0][1][0] + a * b * (1 - c) * T[1][1][0]
           + (1 - a) * (1 - b) * c * T[0][0][1] + a * (1 - b) * c * T[1][0][1]
           + (1 - a) * b * c * T[0][1][1] + a * b * c * T[1][1][1];
}
