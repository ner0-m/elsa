#pragma once

#include "SiddonsMethod.h"
#include "JosephsMethod.h"

#ifdef ELSA_CUDA_PROJECTORS
#include "SiddonsMethodCUDA.h"
#include "JosephsMethodCUDA.h"
#endif

#include "Solver.h"
#include "CG.h"
#include "OGM.h"
#include "FGM.h"
#include "ADMM.h"
#include "ISTA.h"
#include "FISTA.h"
#include "GradientDescent.h"

#include "SoftThresholding.h"
#include "HardThresholding.h"

#include "Identity.h"

#include "CircleTrajectoryGenerator.h"
#include "PhantomGenerator.h"

#include <string_view>
#include <random>

namespace elsa
{
    namespace detail
    {
        /// Impl struct to get the name of the projector at compile time
        template <template <typename> class Op>
        struct ProjectorNameImpl;

        template <>
        struct ProjectorNameImpl<SiddonsMethod> {
            static constexpr std::string_view value = "SiddonCPU";
        };
        template <>
        struct ProjectorNameImpl<JosephsMethod> {
            static constexpr std::string_view value = "JosephCPU";
        };

#ifdef ELSA_CUDA_PROJECTORS
        template <>
        struct ProjectorNameImpl<SiddonsMethodCUDA> {
            static constexpr std::string_view value = "SiddonGPU";
        };
        template <>
        struct ProjectorNameImpl<JosephsMethodCUDA> {
            static constexpr std::string_view value = "JosephGPU";
        };
#endif
    } // namespace detail

    /// Get Projector name for given template at compile time
    template <template <typename> typename Op>
    static constexpr std::string_view ProjectorName_v = detail::ProjectorNameImpl<Op>::value;

    namespace detail
    {
        /// Impl struct to get the name of the projector at compile time
        template <template <typename> class Op>
        struct SolverNameImpl {
        };

        template <>
        struct SolverNameImpl<CG> {
            static constexpr std::string_view value = "CG";
        };
        template <>
        struct SolverNameImpl<OGM> {
            static constexpr std::string_view value = "OGM";
        };
        template <>
        struct SolverNameImpl<FGM> {
            static constexpr std::string_view value = "FMG";
        };
        template <>
        struct SolverNameImpl<GradientDescent> {
            static constexpr std::string_view value = "Grad Desc";
        };
        template <>
        struct SolverNameImpl<ISTA> {
            static constexpr std::string_view value = "ISTA";
        };
        template <>
        struct SolverNameImpl<FISTA> {
            static constexpr std::string_view value = "FISTA";
        };

        /// Complicated shit to get ADMM working here ...
        template <template <typename> typename X, template <typename> typename Z, typename data_t>
        struct SolverNameImplADMM;

        template <typename data_t>
        struct SolverNameImplADMM<CG, SoftThresholding, data_t> {
            static constexpr std::string_view value = "ADMM(CG+SThr)";
        };

        template <typename data_t>
        struct SolverNameImplADMM<CG, HardThresholding, data_t> {
            static constexpr std::string_view value = "ADMM(CG+HThr)";
        };
    } // namespace detail

    /// Get Solver name for given template at compile time
    template <template <typename> typename Solver>
    static constexpr std::string_view SolverName_v = detail::SolverNameImpl<Solver>::value;

    /// Get Solver name for given template at compile time
    template <template <typename> typename XSolver, template <typename> typename ZSolver,
              typename data_t>
    static constexpr std::string_view SolverNameADMM_v =
        detail::SolverNameImplADMM<XSolver, ZSolver, data_t>::value;

    /// Setup for Solvers in the benchmark. Create a specialization if solver needs specific
    /// setup
    template <typename Solver, typename data_t = real_t>
    struct SolverSetup {
    public:
        template <template <typename> typename Op>
        static auto setupSolver(const Op<data_t>& op, const DataContainer<data_t>& phantom)
        {
            auto sinogram = op.apply(phantom);
            WLSProblem problem(op, sinogram);
            Solver solver(problem);
            return std::move(solver.clone());
        }
    };

    namespace detail
    {
        // template <template <typename> typename Solver>
        template <typename Solver>
        struct LASSOSetup {
        public:
            template <template <typename> typename Op, typename data_t>
            static auto setupSolverWithLasso(const Op<data_t>& op,
                                             const DataContainer<data_t>& phantom)
            {
                auto sinogram = op.apply(phantom);

                // Setup WLSProblem
                WLSProblem problem(op, sinogram);

                // Create L1 regularization term
                L1Norm regFunc(op.getDomainDescriptor());
                RegularizationTerm regTerm(0.5f, regFunc);

                // Create LASSOProblem
                LASSOProblem lassoProb(problem, regTerm);
                Solver solver(lassoProb);

                return std::move(solver.clone());
            }
        };
    } // namespace detail

    template <typename data_t>
    struct SolverSetup<ISTA<data_t>, data_t> {
    public:
        template <template <typename> typename Op>
        static auto setupSolver(const Op<data_t>& op, const DataContainer<data_t>& phantom)
        {
            return detail::LASSOSetup<ISTA<data_t>>::setupSolverWithLasso(op, phantom);
        }
    };

    template <typename data_t>
    struct SolverSetup<FISTA<data_t>, data_t> {
    public:
        template <template <typename> typename Op>
        static auto setupSolver(const Op<data_t>& op, const DataContainer<data_t>& phantom)
        {
            return detail::LASSOSetup<FISTA<data_t>>::setupSolverWithLasso(op, phantom);
        }
    };

    namespace detail
    {
        // template <template <typename> typename Solver>
        template <typename Solver>
        struct ADMMSetup {
        public:
            template <template <typename> typename Op, typename data_t>
            static auto setupADMM(const Op<data_t>& op, const DataContainer<data_t>& phantom)
            {
                auto sinogram = op.apply(phantom);
                WLSProblem problem(op, sinogram);

                auto& desc = op.getDomainDescriptor();
                L1Norm<data_t> regFunc(desc);
                RegularizationTerm<data_t> regTerm(0.5f, regFunc);

                Identity<data_t> idOp(desc);
                Scaling<data_t> negativeIdOp(desc, -1);
                DataContainer<data_t> dCC(desc);
                dCC = 0;

                Constraint<data_t> constraint(idOp, negativeIdOp, dCC);

                SplittingProblem<data_t> splittingProblem(problem.getDataTerm(), regTerm,
                                                          constraint);

                Solver admm(splittingProblem);

                return std::move(admm.clone());
            }
        };
    } // namespace detail

    template <typename data_t>
    struct SolverSetup<ADMM<CG, SoftThresholding, data_t>, data_t> {
    public:
        template <template <typename> typename Op>
        static auto setupSolver(const Op<data_t>& op, const DataContainer<data_t>& phantom)
        {
            return detail::ADMMSetup<ADMM<CG, SoftThresholding, data_t>>::setupADMM(op, phantom);
        }
    };

    template <typename data_t>
    struct SolverSetup<ADMM<CG, HardThresholding, data_t>, data_t> {
    public:
        template <template <typename> typename Op>
        static auto setupSolver(const Op<data_t>& op, const DataContainer<data_t>& phantom)
        {
            return detail::ADMMSetup<ADMM<CG, SoftThresholding, data_t>>::setupADMM(op, phantom);
        }
    };

    /// Setup of projector. Create a specialization if projector nedds specific setup
    template <typename Op>
    struct ProjectorSetup {
    public:
        static auto setupProjector(const VolumeDescriptor& volDesc,
                                   const DetectorDescriptor& sinoDesc)
        {
            Op projector(volDesc, sinoDesc);
            return std::move(projector.clone());
        }
    };

    /// Setup of Circular Geometry
    struct CircularGeometrySetup {
        index_t _numAngles{180};
        index_t _arc = {360};

        auto operator()(IndexVector_t size, const DataDescriptor& phantomDesc) const
        {
            // generate circular trajectory
            auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
                _numAngles, phantomDesc, _arc, static_cast<real_t>(size(0)) * 100.f,
                static_cast<real_t>(size(0)));
            return sinoDescriptor;
        }
    };

    /// Setup of Circular Geometry
    template <typename data_t>
    struct SheppLoganPhantomSetup {
        index_t _numAngles{180};
        index_t _arc = {360};

        auto operator()(int dim, int size) const
        {
            const IndexVector_t coeffs = IndexVector_t(dim).setConstant(size);
            auto phantom = PhantomGenerator<data_t>::createModifiedSheppLogan(coeffs);
            return phantom;
        }
    };

    namespace detail
    {
        class RangeIterator
        {
        public:
            RangeIterator(int cur, int step, bool isEnd);

            auto operator++() -> RangeIterator;
            auto operator++(int) -> RangeIterator;

            auto operator*() -> int;

            auto operator==(RangeIterator rhs) const -> bool;
            auto operator!=(RangeIterator rhs) const -> bool;

        private:
            /// Check if `val` is in range of stop_val with step size `step_val`
            template <typename T>
            static constexpr auto isWithinRange(T val, T stopVal, [[maybe_unused]] T stepVal)
                -> bool;

            static auto notEqualToEnd(const RangeIterator& lhs, const RangeIterator& rhs) noexcept
                -> bool;

            void advance();

            int _cur;
            int _step;
            bool _isEnd;
        };
    } // namespace detail

    class Range
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = int;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type;

        Range(int stop) : Range(0, stop) {}
        Range(int start, int stop) : Range(start, stop, 1) {}
        Range(int start, int stop, int step) : _start(start), _end(stop), _step(step) {}

        auto begin() const -> detail::RangeIterator;
        auto end() const -> detail::RangeIterator;

    private:
        int _start{};
        int _end{};
        int _step{};
    };
} // namespace elsa
