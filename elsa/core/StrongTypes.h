#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "Error.h"

#include <utility>
#include <cassert>
#include <array>

namespace elsa
{
    namespace geometry
    {
        class Degree;

        /**
         * @brief Class describing angles in Radians. Mostly used for parameter passing.
         * Has an implicit constructor from Degree, and converts to real_t implicitly as well
         */
        class Radian
        {
        public:
            /// Default constructor (sets 0 radians)
            constexpr Radian() : _radian(0) {}

            /// Constructor from real_t
            constexpr explicit Radian(real_t radian) : _radian(radian) {}

            /// Constructor (implicit) from Degree
            constexpr Radian(Degree d);

            /// Conversion to degree (as real_t)
            constexpr real_t to_degree() const { return _radian * 180 / pi_t; }

            /// Conversion (implicit) to real_t
            constexpr operator real_t() const { return _radian; }

        private:
            real_t _radian;
        };

        /**
         * @brief Class describing angles in Degree. Mostly used for parameter passing.
         * Has an implicit constructor from Radians, and converts to real_t implicitly as well
         */
        class Degree
        {
        public:
            /// Default constructor (sets 0 degrees)
            constexpr Degree() : _degree(0) {}

            /// Constructor from real_t
            constexpr explicit Degree(real_t degree) : _degree(degree) {}

            /// Constructor (implicit) from Radian
            constexpr Degree(Radian r);

            /// Conversion to radian (as real_t)
            constexpr real_t to_radian() const { return _degree * pi_t / 180; }

            /// Conversion (implicit) to real_t
            constexpr operator real_t() const { return _degree; }

        private:
            real_t _degree;
        };

        // Defined out of class, to access the to_... functions
        constexpr Radian::Radian(Degree d) : _radian(d.to_radian()) {}
        constexpr Degree::Degree(Radian r) : _degree(r.to_degree()) {}

        /// Strong type class to a specific angle alpha
        class Alpha : Radian
        {
        public:
            using Base = Radian;

            using Base::Base;
            using Base::to_degree;
            using Base::operator real_t;
        };

        /// Strong type class to a specific angle beta
        class Beta : Radian
        {
        public:
            using Base = Radian;

            using Base::Base;
            using Base::to_degree;
            using Base::operator real_t;
        };

        /// Strong type class to a specific angle gamma
        class Gamma : Radian
        {
        public:
            using Base = Radian;

            using Base::Base;
            using Base::to_degree;
            using Base::operator real_t;
        };

        namespace detail
        {
            /**
             * @brief Class to store a fixed length array of Radians. Ensures, that all entries are
             * convertible to radians
             */
            template <index_t Size>
            class RotationAngles
            {
            public:
                // Alias for Radian
                using Type = Radian;

            private:
                /// Alias for, enable if, all (the conjunction of) Ts... are convertible to Radian
                template <typename... Ts>
                using AllRadian = typename std::enable_if_t<
                    std::conjunction<std::is_convertible<Ts, Type>...>::value>;

            public:
                RotationAngles(const RotationAngles&) = default;
                RotationAngles& operator=(const RotationAngles&) = default;
                RotationAngles(RotationAngles&&) = default;
                RotationAngles& operator=(RotationAngles&&) = default;

                /// Construct array (expect it to be all convertible to Radians and number of
                /// arguments, equal to size)
                template <typename... Ts, typename = AllRadian<Ts...>,
                          typename = std::enable_if_t<(Size > 0) && (Size == (sizeof...(Ts)))>>
                constexpr RotationAngles(Ts&&... ts)
                    : RotationAngles(std::index_sequence_for<Ts...>{}, std::forward<Ts>(ts)...)
                {
                }

                /// Access operator
                constexpr Type operator[](index_t i) const
                {
                    return _angles[static_cast<size_t>(i)];
                }

                /// get function (non-const reference) to enable structured bindings
                template <size_t I>
                Type& get() &
                {
                    return (std::get<I>(_angles));
                }

                /// get function (const reference) to enable structured bindings
                template <size_t I>
                const Type& get() const&
                {
                    return (std::get<I>(_angles));
                }

                /// get function (r-value reference) to enable structured bindings
                template <size_t I>
                Type&& get() &&
                {
                    return std::move(std::get<I>(_angles));
                }

            private:
                /// Helper constructor to init array
                template <std::size_t... Is, typename... Ts, typename = AllRadian<Ts...>>
                constexpr RotationAngles(std::index_sequence<Is...>, Ts&&... vals)
                {
                    // Fold expression expands to _angles[0] = vals0, _angles[1] = vals1, ...,
                    // _angles[k] = valsk;
                    (static_cast<void>(_angles[Is] = vals), ...);
                }

                std::array<Type, Size> _angles;
            };
        } // namespace detail

        /**
         * @brief Strong type class for 3D geometry, which expects 3 angles (alpha, beta, gamma).
         * Class is constructible from the previously defined Alpha, Beta, Gamma and only subsets of
         * it (i.e. only with Beta and Alpha)
         *
         */
        class RotationAngles3D : public detail::RotationAngles<3>
        {
        public:
            using Base = detail::RotationAngles<3>;

        private:
            using Base::Base;

        public:
            /// Construction from Gamma
            constexpr RotationAngles3D(Gamma gamma)
                : Base(Radian{static_cast<real_t>(gamma)}, Radian{0}, Radian{0})
            {
            }

            /// Construction from Beta
            constexpr RotationAngles3D(Beta beta)
                : Base(Radian{0}, Radian{static_cast<real_t>(beta)}, Radian{0})
            {
            }

            /// Construction from Alpha
            constexpr RotationAngles3D(Alpha alpha)
                : Base(Radian{0}, Radian{0}, Radian{static_cast<real_t>(alpha)})
            {
            }

            /// Construction from Gamma and Beta
            constexpr RotationAngles3D(Gamma gamma, Beta beta)
                : Base(Radian{static_cast<real_t>(gamma)}, Radian{static_cast<real_t>(beta)},
                       Radian{0})
            {
            }

            /// Construction from Gamma and Alpha
            constexpr RotationAngles3D(Gamma gamma, Alpha alpha)
                : Base(Radian{static_cast<real_t>(gamma)}, Radian{0},
                       Radian{static_cast<real_t>(alpha)})
            {
            }

            /// Construction from Beta and Gamma
            constexpr RotationAngles3D(Beta beta, Gamma gamma) : RotationAngles3D(gamma, beta) {}

            /// Construction from Beta and Alpha
            constexpr RotationAngles3D(Beta beta, Alpha alpha)
                : Base(Radian{0}, Radian{static_cast<real_t>(beta)},
                       Radian{static_cast<real_t>(alpha)})
            {
            }

            /// Construction from Alpha and Beta
            constexpr RotationAngles3D(Alpha alpha, Gamma gamma) : RotationAngles3D(gamma, alpha) {}

            /// Construction from Alpha and Gamma
            constexpr RotationAngles3D(Alpha alpha, Beta beta) : RotationAngles3D(beta, alpha) {}

            /// Construction from Gamma, Beta and Alpha
            constexpr RotationAngles3D(Gamma gamma, Beta beta, Alpha alpha)
                : Base(Radian{static_cast<real_t>(gamma)}, Radian{static_cast<real_t>(beta)},
                       Radian{static_cast<real_t>(alpha)})
            {
            }

            /// Access to gamma
            constexpr Radian gamma() const { return operator[](0u); }
            /// Access to beta
            constexpr Radian beta() const { return operator[](1u); }
            /// Access to alpha
            constexpr Radian alpha() const { return operator[](2); }
        };

        namespace detail
        {
            /**
             * @brief Wrapper for real_t, used as base class for strong typing
             */
            class RealWrapper
            {
            public:
                constexpr RealWrapper() : _x(0) {}

                constexpr explicit RealWrapper(real_t x) : _x(x) {}

                constexpr operator real_t() { return _x; }

            private:
                real_t _x;
            };

            /**
             * @brief Class wrapping RealVector_t for strong typing, with a fixed size
             */
            template <index_t Size, typename Vector>
            class StaticVectorTemplate
            {
            private:
                using Scalar = typename Vector::Scalar;

                /// Alias for, enable if, all (the conjunction of) Ts... are convertible to Scalar
                template <typename... Ts>
                using AllScalar = typename std::enable_if_t<
                    std::conjunction<std::is_convertible<Ts, Scalar>...>::value>;

            public:
                /// Default constructor
                StaticVectorTemplate() : _vec(Size) {}

                StaticVectorTemplate(const StaticVectorTemplate&) = default;
                StaticVectorTemplate& operator=(const StaticVectorTemplate&) = default;
                StaticVectorTemplate(StaticVectorTemplate&&) = default;
                StaticVectorTemplate& operator=(StaticVectorTemplate&&) = default;

                /// Construct array (expect it to be all convertible to Scalar and number of
                /// arguments, equal to size)
                template <typename... Ts, typename = AllScalar<Ts...>,
                          typename = std::enable_if_t<(Size > 0) && (Size == (sizeof...(Ts) + 1))>>
                StaticVectorTemplate(Scalar x, Ts&&... ts) : _vec(Size)
                {
                    // Fold expression, which expands to _vec << x, t1, t2, t3, ..., tn;
                    ((_vec << x), ..., static_cast<Scalar>(ts));
                }

                /// Constructor from Vector
                StaticVectorTemplate(Vector vec) : _vec(vec)
                {
                    if (_vec.size() != Size)
                        throw InvalidArgumentError(
                            "StaticVectorTemplate::Given argument vector is of the wrong size");
                }

                /// Access operator
                Scalar operator[](index_t i) const { return _vec[i]; }

                /// Access operator
                Scalar& operator[](index_t i) { return _vec[i]; }

                /// Conversion operator to Vector
                operator Vector() const { return _vec; }

                /// Conversion operator to Vector&& (rvalue reference)
                operator Vector&&() { return std::move(_vec); }

                /// Access to vector (const reference)
                const Vector& get() & { return _vec; }

                /// Access to vector (r-value reference)
                Vector&& get() && { return std::move(_vec); }

            private:
                Vector _vec;
            };

            template <index_t Size>
            using StaticRealVector = StaticVectorTemplate<Size, RealVector_t>;

            template <index_t Size>
            using StaticIndexVector = StaticVectorTemplate<Size, IndexVector_t>;
        } // namespace detail

        /// Strong Type for Geometry construction (Distance Source to Center of Rotation)
        class SourceToCenterOfRotation : detail::RealWrapper
        {
        public:
            using Base = detail::RealWrapper;

            using Base::Base;
            using Base::operator real_t;
        };

        /// Strong Type for Geometry construction (Distance Center of Rotation to Principal point)
        class CenterOfRotationToDetector : detail::RealWrapper
        {
        public:
            using Base = detail::RealWrapper;

            using Base::Base;
            using Base::operator real_t;
        };

        /// Strong Type for Geometry construction (1D Principal point offset)
        class PrincipalPointOffset : detail::RealWrapper
        {
        public:
            using Base = detail::RealWrapper;

            using Base::Base;
            using Base::operator real_t;
        };

        /// Strong Type for Geometry construction (2D Principal point offset)
        class PrincipalPointOffset2D : detail::StaticRealVector<2>
        {
        public:
            using Base = detail::StaticRealVector<2>;

            using Base::Base;
            using Base::operator[];
            using Base::operator=;
            // using Base::operator RealVector_t;
            using Base::get;
        };

        /// Strong Type for Geometry construction (nD Offset of Rotation axis)
        template <index_t Size>
        class RotationOffset : detail::StaticRealVector<Size>
        {
        public:
            using Base = detail::StaticRealVector<Size>;

            using Base::Base;
            using Base::operator[];
            using Base::operator=;
            // using Base::operator RealVector_t;
            using Base::get;
        };

        using RotationOffset1D = RotationOffset<1>; ///< Alias for 1D
        using RotationOffset2D = RotationOffset<2>; ///< Alias for 2D
        using RotationOffset3D = RotationOffset<3>; ///< Alias for 3D

        /// Strong Type for Geometry construction (nD Spacing)
        template <index_t Size>
        class Spacing : detail::StaticRealVector<Size>
        {
        public:
            using Base = detail::StaticRealVector<Size>;

            using Base::Base;
            using Base::operator[];
            using Base::operator=;
            // using Base::operator RealVector_t;
            using Base::get;
        };

        using Spacing1D = Spacing<1>; ///< Alias for 1D
        using Spacing2D = Spacing<2>; ///< Alias for 2D
        using Spacing3D = Spacing<3>; ///< Alias for 3D

        /// Strong type for Geometry construction (nD shift of origin)
        template <index_t Size>
        class OriginShift : detail::StaticRealVector<Size>
        {
        public:
            using Base = detail::StaticRealVector<Size>;

            using Base::Base;
            using Base::operator[];
            using Base::operator=;
            // using Base::operator RealVector_t;
            using Base::get;
        };

        using OriginShift1D = OriginShift<1>; ///< Alias for 1D
        using OriginShift2D = OriginShift<2>; ///< Alias for 2D
        using OriginShift3D = OriginShift<3>; ///< Alias for 3D

        /// Strong type for Geometry construction (nD coefficients)
        template <index_t Size>
        class Coefficients : detail::StaticIndexVector<Size>
        {
        public:
            using Base = detail::StaticIndexVector<Size>;

            using Base::Base;
            using Base::operator[];
            using Base::operator=;
            // using Base::operator IndexVector_t;
            using Base::get;
        };

        using Size1D = Coefficients<1>; ///< Alias for 1D
        using Size2D = Coefficients<2>; ///< Alias for 2D
        using Size3D = Coefficients<3>; ///< Alias for 3D

        namespace detail
        {
            /**
             * @brief Base type for strong typing volume and sinogram data
             *
             * @tparam Size Dimension of problem
             */
            template <index_t Size>
            class GeometryData
            {
                template <typename... Ts>
                using AllReal = typename std::enable_if_t<
                    std::conjunction<std::is_convertible<Ts, real_t>...>::value>;

            public:
                /// Alias for RealVector_t
                using Vector = RealVector_t;

                /// Default Constructor
                GeometryData() : _spacing(Size), _locationOfOrigin(Size) {}

                GeometryData(const GeometryData&) = default;
                GeometryData& operator=(const GeometryData&) = default;
                GeometryData(GeometryData&&) = default;
                GeometryData& operator=(GeometryData&&) = default;

                /// Constructor from Strong type Spacing and OriginShift
                GeometryData(Coefficients<Size> size)
                    : _spacing(Vector::Ones(Size)), _locationOfOrigin(Size)
                {
                    IndexVector_t coeffs = size.get();
                    _locationOfOrigin = static_cast<real_t>(0.5)
                                        * (coeffs.cast<real_t>().array() * _spacing.array());
                }

                /// Constructor from Strong type Spacing and OriginShift
                GeometryData(Coefficients<Size> size, Spacing<Size> spacing)
                    : _spacing(std::move(spacing.get())), _locationOfOrigin(Size)
                {
                    IndexVector_t coeffs = size.get();
                    _locationOfOrigin = static_cast<real_t>(0.5)
                                        * (coeffs.cast<real_t>().array() * _spacing.array());
                }

                /// Constructor from Strong type Spacing and OriginShift
                GeometryData(Spacing<Size> spacing, OriginShift<Size> origin)
                    : _spacing(std::move(spacing.get())), _locationOfOrigin(std::move(origin.get()))
                {
                }

                /// Constructor from RealVector_t for Spacing and Strong type OriginShift
                GeometryData(RealVector_t spacing, OriginShift<Size> origin)
                    : _spacing(std::move(spacing)), _locationOfOrigin(std::move(origin.get()))
                {
                    if (_spacing.size() != Size || _locationOfOrigin.size() != Size
                        || _spacing.size() != _locationOfOrigin.size())
                        throw InvalidArgumentError(
                            "Spacing and Origin must have the same dimension");
                }

                /// Constructor from RealVector_t for origin shift and Strong type Spacing
                GeometryData(Spacing<Size> spacing, RealVector_t origin)
                    : _spacing(std::move(spacing.get())), _locationOfOrigin(std::move(origin))
                {
                    if (_spacing.size() != Size || _locationOfOrigin.size() != Size
                        || _spacing.size() != _locationOfOrigin.size())
                        throw InvalidArgumentError(
                            "Spacing and Origin must have the same dimension");
                }

                /// Constructor from RealVector_t for spacing and origin shift
                GeometryData(RealVector_t spacing, RealVector_t origin)
                    : _spacing(std::move(spacing)), _locationOfOrigin(std::move(origin))
                {
                    if (_spacing.size() != Size || _locationOfOrigin.size() != Size
                        || _spacing.size() != _locationOfOrigin.size())
                        throw InvalidArgumentError(
                            "Spacing and Origin must have the same dimension");
                }

                /// Getter for spacing (const reference)
                Vector getSpacing() const& { return _spacing; }

                /// Getter for spacing (r-value reference)
                Vector&& getSpacing() && { return std::move(_spacing); }

                /// Getter for origin shift/location of origin (const reference)
                Vector getLocationOfOrigin() const& { return _locationOfOrigin; }

                /// Getter for origin shift/location of origin (r-value reference)
                Vector&& getLocationOfOrigin() && { return std::move(_locationOfOrigin); }

                /// Get function (const reference overload) for structured bindings
                template <std::size_t N>
                decltype(auto) get() const&
                {
                    if constexpr (N == 0)
                        return (_spacing);
                    else if constexpr (N == 1)
                        return (_locationOfOrigin);
                }

                /// Get function (r-value reference overload) for structured bindings
                template <std::size_t N>
                decltype(auto) get() &&
                {
                    if constexpr (N == 0)
                        return std::move(_spacing);
                    else if constexpr (N == 1)
                        return std::move(_locationOfOrigin);
                }

            private:
                Vector _spacing = Vector::Zero(Size);
                Vector _locationOfOrigin = Vector::Zero(Size);
            };
        } // namespace detail

        /**
         * @brief Strong type for geometry data, describing volume/domain spacing and location of
         * origin.
         *
         * @tparam Size Dimension of problem
         *
         * This used to be private inheritance, changed to public because of NVCC bug involving
         * structured bindings and overloaded get() functions lifted with a using statement
         */
        template <index_t Size>
        class VolumeData : public detail::GeometryData<Size>
        {
        public:
            using Base = detail::GeometryData<Size>;

            using Base::Base;
        };

        using VolumeData2D = VolumeData<2>; ///< 2D volume data alias for 2D geometry
        using VolumeData3D = VolumeData<3>; ///< 3D volume data alias for 3D geometry

        /**
         * @brief Strong type for geometry data, describing sinogram/range spacing and location of
         * origin. Note sinogram data is expected to be 1 dimension less, than the actual problem!
         *
         * @tparam Size Dimension of problem
         *
         * This used to be private inheritance, changed to public because of NVCC bug involving
         * structured bindings and overloaded get() functions lifted with a using statement
         */
        template <index_t Size>
        class SinogramData : public detail::GeometryData<Size>
        {
        public:
            using Base = detail::GeometryData<Size>;

            using Base::Base;
        };

        using SinogramData2D = SinogramData<2>; ///< 2D sinogram data alias for 2D geometry
        using SinogramData3D = SinogramData<3>; ///< 3D sinogram data alias for 3D geometry

        /**
         * @brief Strong type for a single value of type data_t used in proximity operators.
         * Comparison, addition, subtraction are overridden by utilizing the private member
         * _threshold.
         * N.B. The threshold value is expected to be strictly greater than 0, otherwise an
         * exception is thrown
         *
         * @tparam data_t data type of the threshold
         */
        template <typename data_t = real_t>
        class Threshold
        {
        public:
            explicit Threshold(data_t threshold) : _threshold(threshold)
            {
                if (threshold <= 0) {
                    throw InvalidArgumentError("threshold must be strictly greater than 0");
                }
            }

            /// explicit casting operator
            explicit operator data_t() const { return _threshold; }

            /// return -Threshold
            auto operator-() -> const data_t { return (data_t) (-_threshold); }

            /// return computed subtraction
            auto operator-(const data_t t) const -> data_t { return (data_t) (_threshold - t); }

            /// return computed addition
            auto operator+(const data_t t) const -> data_t { return (data_t) (_threshold + t); }

            /// return computed less-than comparison
            auto operator<(const data_t t) const -> bool { return _threshold < t; }

            /// return computed less-than-equals comparison
            auto operator<=(const data_t t) const -> bool { return !(*this > t); }

            /// return computed greater-than comparison
            auto operator>(const data_t t) const -> bool { return _threshold > t; }

            /// return computed greater-than-equals comparison
            auto operator>=(const data_t t) const -> bool { return !(*this < t); }

            /// return computed equality comparison
            auto operator==(const data_t t) const -> bool { return this->_threshold == t; }

            /// return computed equality-negation comparison
            auto operator!=(const data_t t) const -> bool { return !(*this == t); }

        private:
            data_t _threshold;
        };

        /// return computed subtraction of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator-(const data_t a, const Threshold<data_t>& b) -> data_t
        {
            return (data_t) (-(b - a));
        }

        /// return computed addition of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator+(const data_t a, const Threshold<data_t>& b) -> data_t
        {
            return (data_t) (b + a);
        }

        /// return computed greater-than comparison of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator>(const data_t& a, const Threshold<data_t>& b) -> bool
        {
            return b < a;
        }

        /// return computed greater-than-equals comparison of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator>=(const data_t& a, const Threshold<data_t>& b) -> bool
        {
            return b <= a;
        }

        /// return computed less-than comparison of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator<(const data_t& a, const Threshold<data_t>& b) -> bool
        {
            return b > a;
        }

        /// return computed less-than-equals comparison of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator<=(const data_t& a, const Threshold<data_t>& b) -> bool
        {
            return b >= a;
        }

        /// return computed equality comparison of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator==(const data_t& a, const Threshold<data_t>& b) -> bool
        {
            return b == a;
        }

        /// return computed equality-negation comparison of data_t with Threshold<data_t>
        template <typename data_t = real_t>
        auto operator!=(const data_t& a, const Threshold<data_t>& b) -> bool
        {
            return b != a;
        }
    } // namespace geometry
} // namespace elsa

/*
#davidfrank 2020-05-09
All of these functions are needed to enable structured bindings for SinogramData, VolumeData and
RotationAngles I didn't find a way to do it generally for all sizes (at least not with explicit
instantiation). This should be fine for now
*/

/// template specialization for class SinogramData (2D and 3D)
namespace std
{
    template <>
    struct tuple_size<elsa::geometry::SinogramData2D> : std::integral_constant<std::size_t, 2> {
    };

    template <std::size_t N>
    struct tuple_element<N, elsa::geometry::SinogramData2D> {
        using type = decltype(std::declval<elsa::geometry::SinogramData2D>().get<N>());
    };

    template <>
    struct tuple_size<elsa::geometry::SinogramData3D> : std::integral_constant<std::size_t, 2> {
    };

    template <std::size_t N>
    struct tuple_element<N, elsa::geometry::SinogramData3D> {
        using type = decltype(std::declval<elsa::geometry::SinogramData3D>().get<N>());
    };
} // namespace std

/// template specialization for class VolumeData (2D and 3D)
namespace std
{
    template <>
    struct tuple_size<elsa::geometry::VolumeData2D> : std::integral_constant<std::size_t, 2> {
    };

    template <std::size_t N>
    struct tuple_element<N, elsa::geometry::VolumeData2D> {
        using type = decltype(std::declval<elsa::geometry::VolumeData2D>().get<N>());
    };

    template <>
    struct tuple_size<elsa::geometry::VolumeData3D> : std::integral_constant<std::size_t, 2> {
    };

    template <std::size_t N>
    struct tuple_element<N, elsa::geometry::VolumeData3D> {
        using type = decltype(std::declval<elsa::geometry::VolumeData3D>().get<N>());
    };
} // namespace std

/// template specialization for class RotationAngles3D (2D and 3D)
namespace std
{
    template <>
    struct tuple_size<elsa::geometry::RotationAngles3D> : std::integral_constant<std::size_t, 3> {
    };

    template <std::size_t N>
    struct tuple_element<N, elsa::geometry::RotationAngles3D> {
        using type = decltype(std::declval<elsa::geometry::RotationAngles3D>().get<N>());
    };
} // namespace std
