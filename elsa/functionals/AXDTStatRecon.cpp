#include "AXDTStatRecon.h"
#include "TypeCasts.hpp"
#include "IdenticalBlocksDescriptor.h"
#include "Identity.h"
#include "LinearOperator.h"
#include "Scaling.h"
#include "BlockLinearOperator.h"
#include "ZeroOperator.h"
#include "Timer.h"

//#include "Logger.h"
//#include "cmath"

namespace elsa
{
    template <typename data_t>
    std::unique_ptr<DataDescriptor> AXDTStatRecon<data_t>::generate_placeholder_descriptor()
    {
        index_t numOfDims{1};
        IndexVector_t dims(numOfDims);
        dims << 1;
        return std::make_unique<VolumeDescriptor>(dims);
    }

    template <typename data_t>
    RandomBlocksDescriptor AXDTStatRecon<data_t>::generate_descriptors(const DataDescriptor& desc1,
                                                                       const DataDescriptor& desc2)
    {
        std::vector<std::unique_ptr<DataDescriptor>> descs;

        descs.emplace_back(desc1.clone());
        descs.emplace_back(desc2.clone());
        return RandomBlocksDescriptor(descs);
    }

    template <typename data_t>
    AXDTStatRecon<data_t>::AXDTStatRecon(const DataContainer<data_t>& ffa,
                                         const DataContainer<data_t>& ffb,
                                         const DataContainer<data_t>& a,
                                         const DataContainer<data_t>& b,
                                         const LinearOperator<data_t>& absorp_op,
                                         const LinearOperator<data_t>& axdt_op, index_t N,
                                         const StatReconType& recon_type)
        : Functional<data_t>(
            generate_descriptors((recon_type == Gaussian_approximate_racian ||
                                  recon_type == Racian_direct) ?
                                     absorp_op.getDomainDescriptor() :
                                     *generate_placeholder_descriptor(),
                                 axdt_op.getDomainDescriptor())),
          ffa_(ffa),
          ffb_(ffb),
          a_tilde_(a),
          b_tilde_(b),
          absorp_op_(absorp_op.clone()),
          axdt_op_(axdt_op.clone()),
          N_(static_cast<data_t>(N)),
          recon_type_(recon_type)
    {
        alpha_ = *ffb_ / *ffa_;
        d_tilde_ = *b_tilde_ / *a_tilde_ / *alpha_;
    }

    template <typename data_t>
    AXDTStatRecon<data_t>::AXDTStatRecon(const DataContainer<data_t>& axdt_proj,
                                         const LinearOperator<data_t>& axdt_op,
                                         const StatReconType& recon_type)
        : Functional<data_t>(generate_descriptors(*generate_placeholder_descriptor(),
                                                  axdt_op.getDomainDescriptor())),
          absorp_op_(nullptr),
          axdt_op_(axdt_op.clone()),
          N_(static_cast<data_t>(0)),
          recon_type_(recon_type)
    {
        d_tilde_ = exp(-axdt_proj);
        if (recon_type == Gaussian_approximate_racian || recon_type == Racian_direct) {
            throw std::invalid_argument(
                "flat-field data required for requested reconstruction type");
        }
    }

    template <typename data_t>
    data_t AXDTStatRecon<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        Timer timeguard("AXDTStatRecon", "evaluate");
        spdlog::stopwatch timer;

        const auto mu = materialize(Rx.getBlock(0));
        const auto eta = materialize(Rx.getBlock(1));

        auto log_d = -axdt_op_->apply(eta);
        auto d = exp(log_d);

        data_t ll;
        switch (recon_type_) {
            case Gaussian_log_d:
                ll = -square(-log(*d_tilde_) + log_d).sum();
                break;

            case Gaussian_d:
                ll = -square(d - *d_tilde_).sum();
                break;

            case Gaussian_approximate_racian: {
                auto a = exp(-absorp_op_->apply(mu)) * *ffa_;
                auto log_a = log(a);

                auto numerator_1 = static_cast<data_t>(2) * N_ * square(*a_tilde_ - a);
                auto numerator_2 = N_ * square(*b_tilde_ - (a * *alpha_ * d));
                ll = (-log_a - ((numerator_1 + numerator_2) / (static_cast<data_t>(4) * a))).sum();
            } break;

            case Racian_direct: {
                auto a = exp(-absorp_op_->apply(mu)) * *ffa_;
                auto log_a = log(a);

                auto term_1 = -static_cast<data_t>(1.5) * log_a;

                auto term_2 = -N_ / a / static_cast<data_t>(4);
                term_2 *= (static_cast<data_t>(2) * square(a))
                          + (static_cast<data_t>(2) * square(*a_tilde_)) + square(*b_tilde_)
                          + (square(a) * square(*alpha_) * square(d));

                auto term_3 = *b_tilde_ * *alpha_ * d * N_ / static_cast<data_t>(2);
                term_3 = axdt::log_bessel_0(term_3);

                ll = (term_1 + term_2 + term_3).sum();
            } break;
        }
//        Logger::get("AXDTStatRecon")->info("eval(), took {}s", timer);
        return -ll; // to minimize --> NEGATIVE log likelihood
    }

    template <typename data_t>
    void AXDTStatRecon<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                                DataContainer<data_t>& out)
    {
        Timer timeguard("AXDTStatRecon", "getGradient");
        spdlog::stopwatch timer;

        const auto mu = materialize(Rx.getBlock(0));
        const auto eta = materialize(Rx.getBlock(1));

        auto log_d = -axdt_op_->apply(eta);
        auto d = exp(log_d);

        DataContainer<data_t> grad_mu {out.getBlock(0).getDataDescriptor()};
        DataContainer<data_t> grad_eta {out.getBlock(1).getDataDescriptor()};

        switch (recon_type_) {
            case Gaussian_log_d:
                grad_mu = 0;
                grad_eta =
                    axdt_op_->applyAdjoint(static_cast<data_t>(2.0) * (log_d - log(*d_tilde_)));
                break;

            case Gaussian_d:
                grad_mu = 0;
                grad_eta = axdt_op_->applyAdjoint(static_cast<data_t>(2.0) * (d - *d_tilde_) * d);
                break;

            case Gaussian_approximate_racian: {
                auto a = exp(-absorp_op_->apply(mu)) * *ffa_;
                auto log_a = log(a);

                auto grad_mu_tmp = static_cast<data_t>(2.0) * ((a * a) - (*a_tilde_ * *a_tilde_));
                grad_mu_tmp += a * a * *alpha_ * *alpha_ * d * d;
                grad_mu_tmp -= *b_tilde_ * *b_tilde_;
                grad_mu_tmp *= N_ / a / static_cast<data_t>(4.0);
                grad_mu_tmp += static_cast<data_t>(1.0);

                grad_mu = absorp_op_->applyAdjoint(grad_mu_tmp);

                auto grad_eta_tmp = a * *alpha_ * d - *b_tilde_;
                grad_eta_tmp *= *alpha_ * d * N_ * static_cast<data_t>(0.5);

                grad_eta = axdt_op_->applyAdjoint(grad_eta_tmp);
            } break;

            case Racian_direct: {
                auto a = exp(-absorp_op_->apply(mu)) * *ffa_;
                auto log_a = log(a);

                auto grad_mu_tmp = static_cast<data_t>(2.0) * ((a * a) - (*a_tilde_ * *a_tilde_));
                grad_mu_tmp += a * a * *alpha_ * *alpha_ * d * d;
                grad_mu_tmp -= *b_tilde_ * *b_tilde_;
                grad_mu_tmp *= N_ / a / static_cast<data_t>(4.0);
                grad_mu_tmp += static_cast<data_t>(1.5);

                grad_mu = absorp_op_->applyAdjoint(grad_mu_tmp);

                auto grad_eta_tmp = static_cast<data_t>(0.5) * N_ * a * *alpha_ * *alpha_ * d * d;
                auto grad_eta_tmp_bessel = *b_tilde_ * *alpha_ * d * N_ * static_cast<data_t>(0.5);
                grad_eta_tmp -= grad_eta_tmp_bessel * axdt::quot_bessel_1_0(grad_eta_tmp_bessel);

                grad_eta = axdt_op_->applyAdjoint(grad_eta_tmp);
            } break;
        }

        grad_mu = -grad_mu;
        grad_eta = -grad_eta;
        out.getBlock(0) = grad_mu;
        out.getBlock(1) = grad_eta;
//        Logger::get("AXDTStatRecon")->info("getGradient(), took {}s", timer);

        //        {
        //            data_t max_mu = 0;
        //            data_t min_mu = 1000;
        //            data_t max_eta = 0;
        //            data_t min_eta = 1000;
        //            for (int i = 0; i < grad_mu.getDataDescriptor().getNumberOfCoefficients();
        //            ++i) {
        //                max_mu = std::max<data_t>(max_mu, grad_mu[i]);
        //                min_mu = std::min<data_t>(min_mu, grad_mu[i]);
        //            }
        //            for (int i = 0; i < grad_eta.getDataDescriptor().getNumberOfCoefficients();
        //            ++i) {
        //                max_eta = std::max<data_t>(max_eta, grad_eta[i]);
        //                min_eta = std::min<data_t>(min_eta, grad_eta[i]);
        //            }
        //            Logger::get("AXDTStatR")->info("Grad: mu_min {}", min_mu);
        //            Logger::get("AXDTStatR")->info("Grad: mu_max {}", max_mu);
        //            Logger::get("AXDTStatR")->info("Grad: eta_min {}", min_eta);
        //            Logger::get("AXDTStatR")->info("Grad: eta_max {}", max_eta);
        //        }
    }

    template <typename data_t>
    LinearOperator<data_t> AXDTStatRecon<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        Timer timeguard("AXDTStatRecon", "getHessian");
        spdlog::stopwatch timer;

        const auto mu = materialize(Rx.getBlock(0));
        const auto eta = materialize(Rx.getBlock(1));

        auto d = exp(-axdt_op_->apply(eta));

        std::unique_ptr<LinearOperator<data_t>> hessian_absorp, hessian_axdt;
        typename BlockLinearOperator<data_t>::OperatorList ops;
        using BlockType = typename BlockLinearOperator<data_t>::BlockType;
        switch (recon_type_) {
            case Gaussian_log_d: {
                auto hessian_absorp_0 = std::make_unique<ZeroOperator<data_t>>(
                    *generate_placeholder_descriptor(), *generate_placeholder_descriptor());
                auto hessian_absorp_1 = std::make_unique<ZeroOperator<data_t>>(
                    axdt_op_->getDomainDescriptor(), *generate_placeholder_descriptor());
                ops.clear();
                ops.emplace_back(hessian_absorp_0->clone());
                ops.emplace_back(hessian_absorp_1->clone());
                hessian_absorp = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);

                auto hessian_axdt_0 = std::make_unique<ZeroOperator<data_t>>(
                    *generate_placeholder_descriptor(), axdt_op_->getDomainDescriptor());
                auto hessian_axdt_1 = std::make_unique<LinearOperator<data_t>>(
                    static_cast<data_t>(2.0) * adjoint(*axdt_op_) * *axdt_op_);
                ops.clear();
                ops.emplace_back(hessian_axdt_0->clone());
                ops.emplace_back(hessian_axdt_1->clone());
                hessian_axdt = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);
            } break;

            case Gaussian_d: {
                auto hessian_absorp_0 = std::make_unique<ZeroOperator<data_t>>(
                    *generate_placeholder_descriptor(), *generate_placeholder_descriptor());
                auto hessian_absorp_1 = std::make_unique<ZeroOperator<data_t>>(
                    axdt_op_->getDomainDescriptor(), *generate_placeholder_descriptor());
                ops.clear();
                ops.emplace_back(hessian_absorp_0->clone());
                ops.emplace_back(hessian_absorp_1->clone());
                hessian_absorp = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);

                auto hessian_axdt_0 = std::make_unique<ZeroOperator<data_t>>(
                    *generate_placeholder_descriptor(), axdt_op_->getDomainDescriptor());
                auto hessian_axdt_1 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*axdt_op_)
                    * Scaling<data_t>(axdt_op_->getRangeDescriptor(),
                                      static_cast<data_t>(2.0) * d
                                          * (static_cast<data_t>(2.0) * d - *d_tilde_))
                    * *axdt_op_);
                ops.clear();
                ops.emplace_back(hessian_axdt_0->clone());
                ops.emplace_back(hessian_axdt_1->clone());
                hessian_axdt = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);
            } break;

            case Gaussian_approximate_racian: {
                auto a = exp(-absorp_op_->apply(mu)) * *ffa_;

                auto H_1_1 = N_
                             * (static_cast<data_t>(2.0) * a * a
                                + (static_cast<data_t>(2.0) * *a_tilde_ * *a_tilde_)
                                + (a * a * *alpha_ * *alpha_ * d * d) + (*b_tilde_ * *b_tilde_))
                             / static_cast<data_t>(4.0) / a;
                auto H_1_2 = N_ * static_cast<data_t>(0.5) * a * *alpha_ * *alpha_ * d * d;
                auto H_2_2 = N_ * static_cast<data_t>(0.5)
                             * (d
                                * (static_cast<data_t>(2.0) * *alpha_ * *alpha_ * a * d
                                   - (*alpha_ * *b_tilde_)));

                auto hessian_absorp_0 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*absorp_op_) * Scaling<data_t>(absorp_op_->getRangeDescriptor(), H_1_1)
                    * *absorp_op_);
                auto hessian_absorp_1 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*absorp_op_) * Scaling<data_t>(axdt_op_->getRangeDescriptor(), H_1_2)
                    * *axdt_op_);
                ops.clear();
                ops.emplace_back(hessian_absorp_0->clone());
                ops.emplace_back(hessian_absorp_1->clone());
                hessian_absorp = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);

                auto hessian_axdt_0 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*axdt_op_) * Scaling<data_t>(absorp_op_->getRangeDescriptor(), H_1_2)
                    * *absorp_op_);
                auto hessian_axdt_1 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*axdt_op_) * Scaling<data_t>(axdt_op_->getRangeDescriptor(), H_2_2)
                    * *axdt_op_);
                ops.clear();
                ops.emplace_back(hessian_axdt_0->clone());
                ops.emplace_back(hessian_axdt_1->clone());
                hessian_axdt = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);
            } break;

            case Racian_direct: {
                auto a = exp(-absorp_op_->apply(mu)) * *ffa_;

                auto H_1_1 = N_
                             * (static_cast<data_t>(2.0) * a * a
                                + (static_cast<data_t>(2.0) * *a_tilde_ * *a_tilde_)
                                + (a * a * *alpha_ * *alpha_ * d * d) + (*b_tilde_ * *b_tilde_))
                             / static_cast<data_t>(4.0) / a;
                auto H_1_2 = N_ * static_cast<data_t>(0.5) * a * *alpha_ * *alpha_ * d * d;

                auto H_2_2 = N_ * a * *alpha_ * *alpha_ * d * d;
                auto z = *b_tilde_ * *alpha_ * d * N_ * static_cast<data_t>(0.5);
                auto quot_z = axdt::quot_bessel_1_0(z);
                H_2_2 -= z * z * (static_cast<data_t>(1.0) - (quot_z * quot_z));

                auto hessian_absorp_0 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*absorp_op_) * Scaling<data_t>(absorp_op_->getRangeDescriptor(), H_1_1)
                    * *absorp_op_);
                auto hessian_absorp_1 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*absorp_op_) * Scaling<data_t>(axdt_op_->getRangeDescriptor(), H_1_2)
                    * *axdt_op_);
                ops.clear();
                ops.emplace_back(hessian_absorp_0->clone());
                ops.emplace_back(hessian_absorp_1->clone());
                hessian_absorp = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);

                auto hessian_axdt_0 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*axdt_op_) * Scaling<data_t>(absorp_op_->getRangeDescriptor(), H_1_2)
                    * *absorp_op_);
                auto hessian_axdt_1 = std::make_unique<LinearOperator<data_t>>(
                    adjoint(*axdt_op_) * Scaling<data_t>(axdt_op_->getRangeDescriptor(), H_2_2)
                    * *axdt_op_);
                ops.clear();
                ops.emplace_back(hessian_axdt_0->clone());
                ops.emplace_back(hessian_axdt_1->clone());
                hessian_axdt = std::make_unique<BlockLinearOperator<data_t>>(ops, BlockType::COL);
            } break;
        }

        ops.clear();
        ops.emplace_back(hessian_absorp->clone());
        ops.emplace_back(hessian_axdt->clone());
//        Logger::get("AXDTStatRecon")->info("getHessian(), took {}s", timer);
        return leaf(BlockLinearOperator<data_t>(ops, BlockType::ROW));
    }

    template <typename data_t>
    AXDTStatRecon<data_t>* AXDTStatRecon<data_t>::cloneImpl() const
    {
        switch (recon_type_) {
            case Gaussian_log_d:
            case Gaussian_d:
                return new AXDTStatRecon(-log(*d_tilde_), *axdt_op_, recon_type_);
            case Gaussian_approximate_racian:
            case Racian_direct:
                return new AXDTStatRecon(*ffa_, *ffb_, *a_tilde_, *b_tilde_, *absorp_op_, *axdt_op_,
                                         static_cast<index_t>(N_), recon_type_);
        }
    }

    template <typename data_t>
    bool AXDTStatRecon<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherAXDTStatRecon = downcast_safe<AXDTStatRecon>(&other);
        if (!otherAXDTStatRecon)
            return false;

        if (otherAXDTStatRecon->recon_type_ != recon_type_)
            return false;

        switch (recon_type_) {
            case Gaussian_log_d:
            case Gaussian_d:
                if (otherAXDTStatRecon->d_tilde_ != d_tilde_
                    || *(otherAXDTStatRecon->axdt_op_) != *(axdt_op_))
                    return false;
                else
                    return true;
            case Gaussian_approximate_racian:
            case Racian_direct:
                if (otherAXDTStatRecon->ffa_ != ffa_ || otherAXDTStatRecon->ffb_ != ffb_
                    || otherAXDTStatRecon->a_tilde_ != a_tilde_
                    || otherAXDTStatRecon->b_tilde_ != b_tilde_
                    || *(otherAXDTStatRecon->absorp_op_) != *(absorp_op_)
                    || *(otherAXDTStatRecon->axdt_op_) != *(axdt_op_)
                    || otherAXDTStatRecon->N_ != N_)
                    return false;
                else
                    return true;
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class AXDTStatRecon<float>;
    template class AXDTStatRecon<double>;

} // namespace elsa
