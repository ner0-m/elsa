
#include "doctest/doctest.h"
#include "DataContainer.h"
#include "elsaDefines.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"
#include "transforms/FFT.h"

#ifdef ELSA_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <thrust/complex.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>

#include <random>

TEST_SUITE_BEGIN("core");

#ifdef ELSA_CUDA_ENABLED

TEST_CASE_TEMPLATE("fft", data_t, float, double)
{

    GIVEN("Some container")
    {
        auto setup = [&](size_t dim, size_t size) {
            std::random_device r;

            std::default_random_engine e(r());
            std::uniform_real_distribution<data_t> uniform_dist;

            auto shape = elsa::IndexVector_t(dim);
            shape.setConstant(size);

            auto desc = elsa::VolumeDescriptor(shape);

            auto dc = elsa::DataContainer<elsa::complex<data_t>>(desc);
            thrust::generate(thrust::host, dc.begin(), dc.end(), [&]() {
                elsa::complex<data_t> c;
                c.real(uniform_dist(e));
                c.imag(uniform_dist(e));
                return c;
            });
            return dc;
        };

        size_t size[] = {4096, 512, 64};

        for (size_t dims = 1; dims <= 3; dims++) {
            auto dc1 = setup(dims, size[dims - 1]);
            auto dc2 = dc1;

            const auto& desc = dc1.getDataDescriptor();
            const auto& src_shape = desc.getNumberOfCoefficientsPerDimension();
            const auto& src_dims = desc.getNumberOfDimensions();

            WHEN("Using fft (ORTHO)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, false>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::ORTHO));
                elsa::detail::fftHost<elsa::complex<data_t>, false>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::ORTHO);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }

            dc1 = setup(dims, size[dims - 1]);
            dc2 = dc1;

            WHEN("Using ifft (ORTHO)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, true>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::ORTHO));
                elsa::detail::fftHost<elsa::complex<data_t>, true>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::ORTHO);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }

            dc1 = setup(dims, size[dims - 1]);
            dc2 = dc1;

            WHEN("Using fft (FORWARD)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, false>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::FORWARD));
                elsa::detail::fftHost<elsa::complex<data_t>, false>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::FORWARD);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }

            dc1 = setup(dims, size[dims - 1]);
            dc2 = dc1;

            WHEN("Using ifft (BACKWARD)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, true>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::BACKWARD));
                elsa::detail::fftHost<elsa::complex<data_t>, true>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::BACKWARD);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }
        }
    }
}
#endif

TEST_SUITE_END();