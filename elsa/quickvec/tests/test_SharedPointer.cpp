#include "doctest/doctest.h"

#include <thrust/complex.h>

#include "Helpers.cuh"
#include "SharedPointer.cuh"
#include "Defines.cuh"

using namespace quickvec;

template <typename T>
__global__ void testKernel(SharedPointer<T> ptr)
{
    printf("The pointer address is %p\n", (void*) ptr.get());
}

TEST_CASE_TEMPLATE("CUDA unified memory shared pointers", TestType, float, double,
                   thrust::complex<float>, thrust::complex<double>, index_t)
{
    GIVEN("CUDA shared pointer of two elements")
    {
        SharedPointer<TestType> ptr(2 * sizeof(TestType));

        THEN("the counter should be only one")
        {
            REQUIRE(ptr.useCount() == 1);
        }

        WHEN("copy constructing the pointer")
        {
            SharedPointer ptr2(ptr);

            THEN("the counter should be two of both")
            {
                REQUIRE(ptr.useCount() == 2);
                REQUIRE(ptr2.useCount() == 2);
            }
        }

        THEN("copy of pointer should be destroyed")
        {
            REQUIRE(ptr.useCount() == 1);
        }

        WHEN("move constructing new pointer")
        {
            SharedPointer ptr2(std::move(ptr));

            THEN("counter should be still only one")
            {
                REQUIRE(ptr2.useCount() == 1);
            }
        }

        WHEN("copy assignment of other pointer")
        {
            SharedPointer<TestType> ptr2(3 * sizeof(TestType));
            ptr = ptr2;

            THEN("the counter should be two")
            {
                REQUIRE(ptr.useCount() == 2);
            }
        }

        WHEN("move assignment of other pointer")
        {
            SharedPointer<TestType> ptr2(3 * sizeof(TestType));
            ptr = std::move(ptr2);

            THEN("the counter should be two")
            {
                REQUIRE(ptr.useCount() == 2);
            }
        }
    }

    cudaDeviceReset();
}
