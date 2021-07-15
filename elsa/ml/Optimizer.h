#pragma once

#include <utility>
#include "elsaDefines.h"
#include "Cloneable.h"
#include "Common.h"

namespace elsa
{
    namespace ml
    {

        enum class OptimizerType { SGD, Adam };

        template <typename data_t>
        class Optimizer
        {
        public:
            virtual OptimizerType getOptimizerType() const;

            data_t getLearningRate() { return learningRate_; }

        protected:
            /// default constructor
            explicit Optimizer(OptimizerType optimizerType, data_t learningRate);

            /// The type of this optimizer
            OptimizerType optimizerType_;

            /// learning-rate
            data_t learningRate_;
        };

        /// Gradient descent (with momentum) optimizer.
        ///
        /// Update rule for parameter \f$ w \f$ with gradient \f$ g \f$ when momentum is \f$ 0 \f$:
        ///
        /// \f[
        /// w = w - \text{learning_rate} \cdot g
        /// \f]
        ///
        /// Update rule when momentum is larger than \f$ 0 \f$:
        ///
        /// \f[
        /// \begin{eqnarray*}
        /// \text{velocity} &=& \text{momentum} \cdot \text{velocity} - \text{learning_rate} \cdot g
        /// \\ w &=& w  \cdot \text{velocity} \end{eqnarray*} \f]
        ///
        /// When \p nesterov=True, this rule becomes:
        ///
        /// \f[
        /// \begin{eqnarray*}
        ///   \text{velocity} & = & \text{momentum} \cdot \text{velocity} - \text{learning_rate}
        ///   \cdot g \\ w & = & w + \text{momentum} \cdot \text{velocity} - \text{learning_rate}
        ///   \cdot g
        /// \end{eqnarray*}
        /// \f]
        template <typename data_t = real_t>
        class SGD : public Optimizer<data_t>
        {
        public:
            /// Construct an SGD optimizer
            ///
            /// @param learningRate The learning-rate. This parameter is
            /// optional and defaults to 0.01.
            /// @param momentum hyperparameter >= 0 that accelerates gradient
            /// descent in the relevant direction and dampens oscillations. This
            /// parameter is optional and defaults to 0, i.e., vanilla gradient
            /// descent.
            /// @param nesterov Whether to apply Nesterov momentum. This
            /// parameter is optional an defaults to false.
            SGD(data_t learningRate = data_t(0.01), data_t momentum = data_t(0.0),
                bool nesterov = false);

            /// Get momentum.
            data_t getMomentum() const;

            /// True if this optimizer applies Nesterov momentum, false otherwise.
            bool useNesterov() const;

        private:
            /// \copydoc Optimizer::learningRate_
            using Optimizer<data_t>::learningRate_;

            /// momentum parameter
            data_t momentum_;

            /// True if the Nesterov momentum should be used, false otherwise
            bool nesterov_;
        };

        /// Optimizer that implements the Adam algorithm.
        ///
        /// Adam optimization is a stochastic gradient descent method that is
        /// based on adaptive estimation of first-order and second-order
        /// moments.
        ///
        /// According to Kingma et al., 2014, the method is "computationally
        /// efficient, has little memory requirement, invariant to diagonal
        /// rescaling of gradients, and is well suited for problems that are
        /// large in terms of data/parameters".
        template <typename data_t = real_t>
        class Adam : public Optimizer<data_t>
        {
        public:
            /// Construct an Adam optimizer.
            ///
            /// @param learningRate The learning-rate. This parameter is
            /// optional and defaults to 0.001.
            /// @param beta1 The exponential decay rate for the 1st moment
            /// estimates. This parameter is optional and defaults to 0.9.
            /// @param beta2 The exponential decay rate for the 2nd moment
            /// estimates. This parameter is optional and defaults to 0.999.
            /// @param epsilon A small constant for numerical stability. This
            /// epsilon is "epsilon hat" in the Kingma and Ba paper (in the
            /// formula just before Section 2.1), not the epsilon in Algorithm 1
            /// of the paper. This parameter is optional and defaults to 1e-7.
            Adam(data_t learningRate = data_t(0.001), data_t beta1 = data_t(0.9),
                 data_t beta2 = data_t(0.999), data_t epsilon = data_t(1e-7));

            /// Get beta1.
            data_t getBeta1() const;

            /// Get beta2.
            data_t getBeta2() const;

            /// Get epsilon.
            data_t getEpsilon() const;

        private:
            /// \copydoc Optimizer::learningRate_
            using Optimizer<data_t>::learningRate_;

            /// exponential decay for 1st order momenta
            data_t beta1_;

            /// exponential decay for 2nd order momenta
            data_t beta2_;

            /// epsilon-value for numeric stability
            data_t epsilon_;
        };

        namespace detail
        {
            template <typename data_t>
            class OptimizerImplBase
            {
            public:
                OptimizerImplBase(index_t size, data_t learningRate)
                    : size_(size), learningRate_(learningRate)
                {
                }

                virtual ~OptimizerImplBase() = default;

                virtual void updateParameter(const data_t* gradient, index_t batchSize,
                                             data_t* param) = 0;

            protected:
                /// size of weights and gradients
                index_t size_;

                /// learning-rate
                data_t learningRate_;

                /// current execution step
                index_t step_ = 0;
            };

            template <typename data_t, MlBackend Backend>
            class OptimizerAdamImpl
            {
            };

            template <typename data_t, MlBackend Backend>
            class OptimizerSGDImpl
            {
            };

            template <typename data_t, MlBackend Backend>
            struct OptimizerFactory {
                static std::shared_ptr<OptimizerImplBase<data_t>> run([
                    [maybe_unused]] Optimizer<data_t>* opt)
                {
                    throw std::logic_error("No Ml backend available");
                }
            };

        } // namespace detail
    }     // namespace ml
} // namespace elsa
