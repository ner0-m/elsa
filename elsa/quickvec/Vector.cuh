#pragma once

#include "Helpers.cuh"
#include "Defines.cuh"
#include <type_traits>
#include "Expression.cuh"
#include <memory>
#include "Eigen/Dense"
#include "SharedPointer.cuh"

namespace quickvec
{

    inline __host__ __device__ float abs(float number) { return std::abs(number); }

    inline __host__ __device__ double abs(double number) { return std::abs(number); }

    inline __host__ __device__ index_t abs(index_t number) { return std::abs(number); }

    inline __host__ __device__ double abs(thrust::complex<double> number)
    {
        return thrust::abs(number);
    }

    inline __host__ __device__ float abs(thrust::complex<float> number)
    {
        return thrust::abs(number);
    }

    /**
     * \brief The expression evaluation kernel
     *
     * \tparam Source the expression to evaluate
     * \tparam data_t datatype of the individual elements
     *
     * \param[in] n the number elements to evaluate
     * \param[in] source device pointer to the expression
     * \param[in] result device pointer to the beginning of the result array
     *
     * The kernel will descend in each individual thread into the tree which is provided by the
     * "source" expression.
     */
    template <typename Source, typename data_t>
    __global__ void compute(size_t n, Source* source, data_t* result)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        for (unsigned int i = index; i < n; i += stride) {
            result[i] = source->operator[](i);
        }
    }

    /**
     * \brief Sets all values of a device array to the same value.
     *
     * \param[in] n number of elements
     * \param[in] value resulting value of each element
     * \param[in] result pointer to the start of the array which will be filled with value
     */
    template <typename data_t>
    __global__ void set(size_t n, data_t value, data_t* result)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        for (unsigned int i = index; i < n; i += stride) {
            result[i] = value;
        }
    }

    /**
     * \brief Main class in Quickvec representing an ordered collection of n elements stored on the
     * GPU.
     *
     * \author Jens Petit
     *
     * \tparam data_t arithmetic type stored in
     *
     * Implements scalar with vector as well as vector vector arithmetic.
     */
    template <typename data_t>
    class Vector
    {
    public:
        /// Empty create not possible
        Vector() = delete;

        /// Constructor transfers data from Eigen::Matrix to the GPU
        Vector(Eigen::Matrix<data_t, Eigen::Dynamic, 1> const& data);

        /// Constructor creates empty object with size number of elements
        explicit Vector(size_t size);

        /**
         * \brief Constructor to create vector with existing allocated CUDA memory.
         *
         * \param[in] data pointer to the exisitng memory
         * \param[in] size number of elements at the data location
         * \param[in] owning indicates if the newly created object own the memory
         *
         * If owning is set to true, the destructor might free the memory.
         */
        Vector(data_t* data, size_t size, bool owning = false);

        /**
         * \brief Copy constructor will do a shallow copy
         *
         * \param[in] other
         *
         * This will perform a shallow copy which is intended as Vector is stored by value in the
         * expression. This is necessary to transfer the expression object later to the GPU and have
         * valid objects in it. Shallow refers to the fact that it still points to the same memory
         * as other.
         */
        Vector(Vector<data_t> const& other) = default;

        /**
         * \brief Move construct will do a shallow copy
         *
         * \param[in] other
         *
         * Same as the copy constructor, this will do a shallow copy, referencing the same data as
         * other.
         */
        Vector(Vector<data_t>&&) = default;

        /// deep copy assignment
        Vector<data_t>& operator=(const Vector& other);

        /// move assignment
        Vector<data_t>& operator=(Vector&& other);

        /// deep copy
        Vector<data_t> clone() const;

        /// destructor
        ~Vector() = default;

        /**
         * \brief Evaluates the expression and stores results into the object.
         *
         * \param[in] source the expression from which to evaluate
         *
         * \tparam Source the type of the expression which is evaluated
         *
         * The function copies the expression to the device and calls the compute Kernel with the
         * device expression.
         */
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        void eval(Source source)
        {
            // TODO: Automatically choose params depending on device
            unsigned int blockSize = 256;
            auto numBlocks = static_cast<unsigned int>((_size + blockSize - 1) / blockSize);

            Source* devExpression;
            gpuErrchk(cudaMalloc(&devExpression, sizeof(Source)));
            gpuErrchk(cudaMemcpy(devExpression, &source, sizeof(Source), cudaMemcpyHostToDevice));

            compute<<<numBlocks, blockSize>>>(_size, devExpression, _data.get());

            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaFree(devExpression));
        }

        /// device element wise access function
        __device__ __host__ const data_t& operator[](size_t index) const
        {
            return _data.get()[index];
        }

        __device__ __host__ data_t& operator[](size_t index) { return _data.get()[index]; }

        /// print out quickvec::Vector object
        friend std::ostream& operator<<(std::ostream& output, Vector const& dc)
        {
            for (size_t i = 0; i < dc._size; i++) {
                output << dc[i] << ", ";
            }
            output << "\n";
            return output;
        }

        /// return number of elements in the vector
        size_t size() const { return _size; }

        /// return the squared l2 norm (dot product with itself)
        GetFloatingPointType_t<data_t> squaredl2Norm() const;

        /// return the l2 norm (dot product with itself)
        GetFloatingPointType_t<data_t> l2Norm() const;

        /// return the l1 norm (sum of absolute values)
        GetFloatingPointType_t<data_t> l1Norm() const;

        /// return the linf norm
        GetFloatingPointType_t<data_t> lInfNorm() const;

        /// return the l0 "norm" (i.e the number of non zero elements)
        index_t l0PseudoNorm() const;

        /// return the sum of all elements
        data_t sum() const;

        /// return the dot product of the data vector with vector v
        data_t dot(const Vector<data_t>& v) const;

        /// compute in-place element-wise addition of another vector v
        Vector<data_t>& operator+=(const Vector<data_t>& v);

        /// compute in-place element-wise subtraction of another vector v
        Vector<data_t>& operator-=(const Vector<data_t>& v);

        /// compute in-place element-wise multiplication by another vector v
        Vector<data_t>& operator*=(const Vector<data_t>& v);

        /// compute in-place element-wise division by another vector v
        Vector<data_t>& operator/=(const Vector<data_t>& v);

        /// compute in-place addition of a scalar
        Vector<data_t>& operator+=(data_t scalar);

        /// compute in-place subtraction of a scalar
        Vector<data_t>& operator-=(data_t scalar);

        /// compute in-place multiplication by a scalar
        Vector<data_t>& operator*=(data_t scalar);

        /// compute in-place division by a scalar
        Vector<data_t>& operator/=(data_t scalar);

        /// assign a scalar to all elements of the data vector
        Vector<data_t>& operator=(data_t scalar);

        /// comparison operator
        bool operator==(Vector<data_t> const& other) const;

        /// unequal operator
        bool operator!=(Vector<data_t> const& other) const { return !this->operator==(other); };

    private:
        /// number of elements
        size_t _size;

    public:
        /// points to data on the GPU via a custom CUDA shared pointer
        SharedPointer<data_t> _data;
    };

    struct Multiplying {
        template <typename U, typename V>
        __device__ auto operator()(U u, V v) const
        {
            return u * v;
        }
    };

    struct Adding {
        template <typename U, typename V>
        __device__ auto operator()(U u, V v) const
        {
            return u + v;
        }
    };

    struct Subtracting {
        template <typename U, typename V>
        __device__ auto operator()(U u, V v) const
        {
            return u - v;
        }
    };

    struct Dividing {
        template <typename U, typename V>
        __device__ auto operator()(U u, V v) const
        {
            return u / v;
        }
    };

    /// overloaded multiplication operator returns the corresponding expression
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator*(LHS const& lhs, RHS const& rhs)
    {
        Multiplying multiply;
        auto expr = Expression<Multiplying, LHS, RHS>(multiply, lhs, rhs);
        return expr;
    }

    // overloaded addition operator returns the corresponding expression
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator+(LHS const& lhs, RHS const& rhs)
    {
        Adding add;
        auto expr = Expression<Adding, LHS, RHS>(add, lhs, rhs);
        return expr;
    }

    // overloaded subtraction operator returns the corresponding expression
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator-(LHS const& lhs, RHS const& rhs)
    {
        Subtracting subtract;
        auto expr = Expression<Subtracting, LHS, RHS>(subtract, lhs, rhs);
        return expr;
    }

    ///// overloaded division operator returns the corresonding expression
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator/(LHS const& lhs, RHS const& rhs)
    {
        Dividing divide;
        auto expr = Expression<decltype(divide), LHS, RHS>(divide, lhs, rhs);
        return expr;
    }

    struct Squaring {
        template <typename U>
        __device__ U operator()(U u) const
        {
            return u * u;
        }
    };

    /// square operation returns the corresponding expression
    template <typename Operand, typename = std::enable_if_t<isVectorOrExpression<Operand>>>
    auto square(Operand const& operand)
    {
        Squaring square;
        auto expr = Expression<Squaring, Operand>(square, operand);
        return expr;
    }

    struct Squareroot {
        template <typename U>
        __device__ U operator()(U u) const
        {
            if constexpr (isComplex<U>) {
                return static_cast<U>(thrust::sqrt(u));
            } else {
                return static_cast<U>(std::sqrt(u));
            }
        }
    };

    /// squareroot operation returns the corresponding expression
    template <typename Operand, typename = std::enable_if_t<isVectorOrExpression<Operand>>>
    auto sqrt(Operand const& operand)
    {
        Squareroot sqrt;
        auto expr = Expression<decltype(sqrt), Operand>(sqrt, operand);
        return expr;
    }

    struct Exponent {
        template <typename U>
        __device__ U operator()(U u) const
        {
            if constexpr (isComplex<U>) {
                return static_cast<U>(thrust::exp(u));
            } else {
                return static_cast<U>(std::exp(u));
            }
        }
    };

    /// exponent operation returns the corresponding expression
    template <typename Operand, typename = std::enable_if_t<isVectorOrExpression<Operand>>>
    auto exp(Operand const& operand)
    {
        Exponent exp;
        auto expr = Expression<Exponent, Operand>(exp, operand);
        return expr;
    }

    struct Logarithm {
        template <typename U>
        __device__ U operator()(U u) const
        {
            if constexpr (isComplex<U>) {
                return static_cast<U>(thrust::log(u));
            } else {
                return static_cast<U>(std::log(u));
            }
        }
    };

    /// logarithm operation returns the corresponding expression
    template <typename Operand, typename = std::enable_if_t<isVectorOrExpression<Operand>>>
    auto log(Operand const& operand)
    {
        Logarithm log;
        auto expr = Expression<decltype(log), Operand>(log, operand);
        return expr;
    }

} // namespace quickvec
