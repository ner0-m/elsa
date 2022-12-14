#include "PhantomDefines.h"

namespace elsa::phantoms
{

    template <typename data_t>
    void fillRotationMatrix(Vec3X<data_t> eulers, Eigen::Matrix<data_t, 3, 3>& rot)
    {
        // convert to radians
        auto phiRad = eulers[INDEX_PHI] * pi<double> / 180.0;
        auto thetaRad = eulers[INDEX_THETA] * pi<double> / 180.0;
        auto psiRad = eulers[INDEX_PSI] * pi<double> / 180.0;

        auto cosPhi = std::cos(phiRad);
        auto sinPhi = std::sin(phiRad);
        auto cosTheta = std::cos(thetaRad);
        auto sinTheta = std::sin(thetaRad);
        auto cosPsi = std::cos(psiRad);
        auto sinPsi = std::sin(psiRad);

        // setup ZXZ Euler rotation matrix
        rot(0, 0) = static_cast<real_t>(cosPhi * cosPsi - cosTheta * sinPhi * sinPsi);
        rot(0, 1) = static_cast<real_t>(cosPsi * sinPhi + cosPhi * cosTheta * sinPsi);
        rot(0, 2) = static_cast<real_t>(sinTheta * sinPsi);

        rot(1, 0) = static_cast<real_t>(-cosPhi * sinPsi - cosTheta * cosPsi * sinPhi);
        rot(1, 1) = static_cast<real_t>(cosPhi * cosTheta * cosPsi - sinPhi * sinPsi);
        rot(1, 2) = static_cast<real_t>(cosPsi * sinTheta);

        rot(2, 0) = static_cast<real_t>(sinPhi * sinTheta);
        rot(2, 1) = static_cast<real_t>(-cosPhi * sinTheta);
        rot(2, 2) = static_cast<real_t>(cosTheta);
    }

    std::string getString(Orientation o)
    {
        switch (o) {
            case Orientation::X_AXIS:
                return "X_AXIS";
            case Orientation::Y_AXIS:
                return "Y_AXIS";
            case Orientation::Z_AXIS:
                return "Z_AXIS";
            default:
                return "xxxx";
        }
    }

    // explicit template instantiation
    template void fillRotationMatrix<double>(Vec3X<double>, Eigen::Matrix<double, 3, 3>&);
    template void fillRotationMatrix<float>(Vec3X<float>, Eigen::Matrix<float, 3, 3>&);

} // namespace elsa::phantoms
