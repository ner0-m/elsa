#include <cuda_runtime.h>

#include <Eigen/Dense>
#include "DetectorDescriptor.h"
#include "DataContainer.h"
#include "elsaDefines.h"
template <typename data_t, int n>
struct Eoid {
    data_t w;
    Eigen::Vector<data_t, n> c;
    Eigen::DiagonalMatrix<data_t, n> A;
    Eigen::Matrix<data_t, n, n> R;
};
namespace elsa::phantoms
{

    template <typename data_t, int n>
    void renderEllipsoid(dim3 sinogramDims, int threads, data_t* const sinogram,
                         uint64_t sinogramPitch, data_t* const rays, uint64_t rayPitch,
                         Eoid<data_t, n> ellipsoid);

}; // namespace elsa::phantoms