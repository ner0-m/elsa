#pragma once

#include "Statistics.hpp"
#include "SetupHelpers.h"
#include "NoiseGenerators.h"
#include "LoggingHelpers.h"
#include "TypeCasts.hpp"

#include <thread>
#include <chrono>
#include <utility>
#include <random>
#include <string_view>

namespace elsa
{
    /// Stats for a single iteration of a run
    template <typename data_t = real_t>
    struct Stats {
        /// Typedef to make generic programming easier
        using Scalar = data_t;

        /// Store time as floating point representation that we can easily print in decimal notation
        using fsec = std::chrono::duration<real_t>;

        /// Time it took for the given iteration
        fsec::rep _time;

        /// Absolute error
        data_t _absError;

        /// Relative error
        data_t _relError;
    };

    /**
     * @brief Compute mean and standard deviation for each quantity we're tracking with `Stats`
     *
     * @param stats vector of `Stats` storing samples for each iteration of the benchmark
     * @return mean and standard deviation of each of the quantities stored in `Stats`
     */
    template <typename data_t = real_t>
    auto evaluateStats(const std::vector<Stats<data_t>>& stats)
    {
        std::vector<typename Stats<data_t>::fsec::rep> time;
        time.reserve(stats.size());

        std::vector<typename Stats<data_t>::Scalar> absError;
        absError.reserve(stats.size());

        std::vector<typename Stats<data_t>::Scalar> relError;
        relError.reserve(stats.size());

        for (const auto& s : stats) {
            time.push_back(s._time);
            absError.push_back(s._absError);
            relError.push_back(s._relError);
        }

        auto [timeMean, timeStddev] = calculateMeanStddev(time);
        auto [absMean, absStddev] = calculateMeanStddev(absError);
        auto [relMean, relStddev] = calculateMeanStddev(relError);

        return std::make_tuple(timeMean, timeStddev, absMean, absStddev, relMean, relStddev);
    }

    /// Declaration of benchmark driver
    namespace detail
    {
        template <typename Solver, template <typename> typename Op, typename data_t = real_t,
                  typename NoiseGenerator = NoNoiseGenerator,
                  typename PhantomSetup = SheppLoganPhantomSetup<data_t>,
                  typename Logging = ConsoleLogging, typename GeometrySetup = CircularGeometrySetup>
        void benchDriverImpl(int dim, int size, std::size_t benchIters, std::string_view solverName,
                             Range range, NoiseGenerator&& noiseGen = {},
                             PhantomSetup&& phantomSetup = {},
                             GeometrySetup&& trajectorySetup = {});
    }

    /// benchDriver overload for all solvers that expect one template argument (e.g. CG,
    /// GradientDescent). Deduces name of solver and calls detail::benchDriverImpl.
    template <template <typename> typename Solver, template <typename> typename Op,
              typename data_t = real_t, typename NoiseGenerator = NoNoiseGenerator,
              typename PhantomSetup = SheppLoganPhantomSetup<data_t>,
              typename Logging = ConsoleLogging, typename GeometrySetup = CircularGeometrySetup>
    void benchDriver(int dim, int size, std::size_t benchIters = 5, Range range = {50, 300, 50},
                     NoiseGenerator&& noiseGen = {}, PhantomSetup&& phantomSetup = {},
                     GeometrySetup&& trajectorySetup = {})
    {
        auto solName = SolverName_v<Solver>;
        detail::benchDriverImpl<Solver<data_t>, Op, data_t, NoiseGenerator, PhantomSetup, Logging,
                                GeometrySetup>(dim, size, benchIters, solName, range,
                                               std::move(noiseGen), std::move(phantomSetup),
                                               std::move(trajectorySetup));
    }

    /// benchDriver overload for solvers that expect multiple template template parameters (e.g.
    /// ADMM). Deduces name of solver and calls detail::benchDriverImpl.
    template <template <template <typename> typename, template <typename> typename, typename>
              typename Solver,
              template <typename> typename XSolver, template <typename> typename ZSolver,
              template <typename> typename Op, typename data_t = real_t,
              typename NoiseGenerator = NoNoiseGenerator,
              typename PhantomSetup = SheppLoganPhantomSetup<data_t>,
              typename Logging = ConsoleLogging, typename GeometrySetup = CircularGeometrySetup>
    void benchDriver(int dim, int size, std::size_t benchIters = 5, Range range = {50, 300, 50},
                     NoiseGenerator&& noiseGen = {}, PhantomSetup&& phantomSetup = {},
                     GeometrySetup&& trajectorySetup = {})
    {
        auto solName = SolverNameADMM_v<XSolver, ZSolver, data_t>;
        detail::benchDriverImpl<Solver<XSolver, ZSolver, data_t>, Op, data_t, NoiseGenerator,
                                PhantomSetup, Logging, GeometrySetup>(
            dim, size, benchIters, solName, range, std::move(noiseGen), std::move(phantomSetup),
            std::move(trajectorySetup));
    }

    namespace detail
    {
        /**
         * @brief Driver for the integration tests. Runs a loop with a given solver collecting
         * statistics on it and printing it out
         *
         * @author David Frank - initial code
         *
         * @param dim - dimension of problem
         * @param size - size of the problem
         * @param benchIters - number of iterations the complete reconstruction is run with a fixed
         * set of parameters
         * @param solName - Name of solver, passed in as ADMM needs special handling
         * @param range - range of iterations done by the solver
         * @param noiseGen - Add noise to the ground truth phantom
         * @param phantomSetup - function object to setup the phantom
         * @param trajectorySetup - function object to setup the trajectory
         *
         * @tparam Solver - The solver used in this integration run
         * @tparam Op - The projector used in this integration run
         * @tparam data_t - data representation type (default `real_t`)
         * @tparam NoiseGenerator - Add noise to initial phantom, Function object with a single
         * function `operator()(DataContainer) -> DataContainer` (default `NoNoiseGenerator`)
         * @tparam PhantomSetup - Setup phantom, Function object with single function
         * `operator()(int dim, int size) -> DataContainer` (default Shepp-Logan phantom)
         * @tparam GeometrySetup - Setup trajectory, Function object with single function
         * `operator()(IndexVector_t coeffs, VolumeDescriptor volDesc) -> DataDescriptor`
         *
         */
        template <typename Solver, template <typename> typename Op, typename data_t,
                  typename NoiseGenerator, typename PhantomSetup, typename Logging,
                  typename GeometrySetup>
        void benchDriverImpl(int dim, int size, std::size_t benchIters, std::string_view solName,
                             Range range, NoiseGenerator&& noiseGen, PhantomSetup&& phantomSetup,
                             GeometrySetup&& trajectorySetup)
        {
            /// Float representation of seconds
            using fsec = std::chrono::duration<real_t>;

            // Short type for the clock
            using clock = std::chrono::system_clock;

            // Short for nanoseconds
            using std::chrono::nanoseconds;

            // Log header
            Logging::logHeader();

            // Some statistics we want to save
            std::vector<Stats<data_t>> stats(benchIters);

            // Setup a phantom, default a Modified Shepp Logan
            auto phantom = phantomSetup(dim, size);
            // Calculate the norm of the ground truth
            const auto phantomNorm = phantom.l2Norm();

            // Add noise, default NoNoiseGenerator
            phantom = noiseGen(phantom);

            const auto& volDesc = downcast<VolumeDescriptor>(phantom.getDataDescriptor());
            const IndexVector_t coeffs = volDesc.getNumberOfCoefficientsPerDimension();

            // setup trajectory
            auto sinoDesc = trajectorySetup(coeffs, volDesc);

            // setup Projector
            auto op = ProjectorSetup<Op<data_t>>::setupProjector(volDesc, *sinoDesc);

            constexpr auto opName = ProjectorName_v<Op>;

            // loop over given range
            for (auto noIters : range) {
                // turn logger off, so we don't get any noise
                Logger::setLevel(Logger::LogLevel::OFF);

                // Not quite sure why this needs to be here, but if it's outside the previous
                // for loop, it doesn't work after one iterations...
                auto logger = Logger::get("Benchmark");
                logger->set_pattern("%v");
                logger->set_formatter(std::make_unique<CarriageReturnFormatter>());

                // Repeat the solver a couple of times
                for (unsigned long i = 0; i < benchIters; ++i) {
                    // Setup solver new each time
                    auto solver = SolverSetup<Solver, data_t>::setupSolver(*op, phantom);

                    Logger::setLevel(Logger::LogLevel::INFO);
                    logger->info("Benchmark iterations {}/{}", i + 1, benchIters);
                    Logger::setLevel(Logger::LogLevel::OFF);

                    // Run the solver
                    const auto start = clock::now();
                    const auto rec = solver->solve(noIters);
                    const auto end = clock::now();

                    // Gather some statistics
                    const auto time = std::chrono::duration_cast<fsec>(end - start).count();

                    // Calculate error
                    DataContainer diff = rec - phantom;
                    const auto absError = diff.l2Norm();
                    const auto relError = absError / phantomNorm;

                    stats[i] = {time, absError, relError};
                }

                // Calculate error and standard deviation of statistics
                auto [timeMean, timeStddev, absErrMean, absErrStddev, relErrMean, relErrStddev] =
                    evaluateStats(stats);

                auto [lower, upper] = confidenceInterval95(benchIters, timeMean, timeStddev);

                // Log a laps
                Logging::template logLaps<data_t>(dim, size, benchIters, opName, solName, noIters,
                                                  timeMean, timeStddev, lower, upper, absErrMean,
                                                  relErrMean);
            }
        }
    } // namespace detail
} // namespace elsa
