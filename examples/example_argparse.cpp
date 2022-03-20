#include "elsa.h"
#include "LutProjector.h"
#include "Utilities/Statistics.hpp"
#include "IO.h"

#include <argparse/argparse.hpp>

std::pair<elsa::DataContainer<elsa::real_t>, elsa::DataContainer<elsa::real_t>>
    recon2d(elsa::index_t s, elsa::index_t numAngles, elsa::index_t arc, elsa::index_t iters)
{
    elsa::IndexVector_t size({{s, s}});

    const auto phantom = elsa::PhantomGenerator<elsa::real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // write the phantom out
    elsa::io::write(phantom, "2dphantom.pgm");

    // generate circular trajectory
    const auto distance = static_cast<elsa::real_t>(size(0));
    auto sinoDescriptor = elsa::CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as PhantomGenerator returns a
    // VolumeDescriptor
    elsa::Logger::get("Info")->info("Create BlobProjector");
    elsa::BlobProjector projector(dynamic_cast<const elsa::VolumeDescriptor&>(volumeDescriptor),
                                  *sinoDescriptor);

    // simulate the sinogram
    elsa::Logger::get("Info")->info("Calculate sinogram");
    auto sinogram = projector.apply(phantom);
    elsa::io::write(sinogram, "2dsinogram.pgm");

    // setup reconstruction problem
    elsa::WLSProblem wlsProblem(projector, sinogram);

    elsa::CG cgSolver(wlsProblem);

    auto cgReconstruction = cgSolver.solve(iters);
    elsa::PGM::write(cgReconstruction, "2dreconstruction_cg.pgm");

    return {cgReconstruction, phantom};
    // return {phantom, phantom};
}

std::pair<elsa::DataContainer<elsa::real_t>, elsa::DataContainer<elsa::real_t>>
    recon3d(elsa::index_t s, elsa::index_t numAngles, elsa::index_t arc, elsa::index_t iters)
{
    elsa::IndexVector_t size({{s, s}});

    const auto phantom = elsa::PhantomGenerator<elsa::real_t>::createModifiedSheppLogan(size);

    return {phantom, phantom};
}

template <typename data_t>
data_t covariance(elsa::DataContainer<data_t> a, elsa::DataContainer<data_t> b, data_t mean_a,
                  data_t mean_b)
{
    data_t sum = 0.0;
    for (int i = 0; i < a.getSize(); ++i) {
        sum += ((a[i] - mean_a) * (b[i] - mean_b));
    }
    return sum / a.getSize();
}

template <typename data_t>
data_t covariance(elsa::DataContainer<data_t> a, elsa::DataContainer<data_t> b)
{
    const auto mean_a = a.sum() / a.getSize();
    const auto mean_b = b.sum() / b.getSize();

    return covariance(a, b, mean_a, mean_b);
}

template <typename data_t>
data_t mean(elsa::DataContainer<data_t> data)
{
    return data.sum() / data.getSize();
}

template <typename data_t>
data_t variance(elsa::DataContainer<data_t> a)
{
    const auto mean_a = mean(a);

    data_t sum = 0.0;
    for (int i = 0; i < a.getSize(); ++i) {
        sum += std::pow(a[i] - mean_a, 2);
    }
    return sum / a.getSize();
}

template <typename data_t>
data_t standardDeviation(elsa::DataContainer<data_t> a)
{
    return std::sqrt(variance(a));
}

template <typename data_t>
data_t meanSquaredError(elsa::DataContainer<data_t> a, elsa::DataContainer<data_t> b)
{
    return elsa::DataContainer(a - b).l2Norm() / a.getSize();
}

template <typename data_t>
data_t rootMeanSquaredError(elsa::DataContainer<data_t> a, elsa::DataContainer<data_t> b)
{
    return std::sqrt(meanSquaredError(a, b));
}

template <typename data_t>
data_t peakSignalToNoiseRation(elsa::DataContainer<data_t> a, elsa::DataContainer<data_t> b)
{
    const auto mse = meanSquaredError(a, b);
    return 20 * std::log10(a.maxElement()) - 10 * std::log10(mse);
}

template <typename data_t>
data_t structuralSimilarityIndex(elsa::DataContainer<data_t> a, elsa::DataContainer<data_t> b)
{
    auto mean_a = mean(a);
    auto mean_b = mean(b);

    const auto var_a = variance(a);
    const auto var_b = variance(a);

    auto covar = covariance(a, b);

    const auto L = std::pow(2, sizeof(data_t)) - 1;
    const auto k1 = 0.01;
    const auto k2 = 0.03;

    const auto c1 = std::pow(k1 * L, 2);
    const auto c2 = std::pow(k2 * L, 2);

    const auto q1 = 2 * mean_a * mean_b + c1;
    const auto q2 = 2 * covar + c1;
    const auto q3 = mean_a * mean_a + mean_b * mean_b + c1;
    const auto q4 = var_a * var_a + var_b * var_b + c2;

    return (q1 * q2) / (q3 * q4);
}

template <typename data_t>
void analyze(elsa::DataContainer<data_t> phantom, elsa::DataContainer<data_t> recon)
{
    // Compute mean squared difference
    auto mse = meanSquaredError(phantom, recon);
    auto rmse = rootMeanSquaredError(phantom, recon);

    elsa::Logger::get("Analyze")->info("MSE: {}", mse);
    elsa::Logger::get("Analyze")->info("RMSE: {}", rmse);

    auto psnr = peakSignalToNoiseRation(phantom, recon);
    elsa::Logger::get("Analyze")->info("PSNR: {} dB", psnr);

    const auto ssim = structuralSimilarityIndex(phantom, recon);
    elsa::Logger::get("Analyze")->info("SSIM: {}", ssim);
};

int main(int argc, char* argv[])
{
    argparse::ArgumentParser args("elsa", "0.7");

    args.add_argument("--dims").help("Dimension of the problem").default_value(2).scan<'i', int>();
    args.add_argument("--size").help("Size of the problem").default_value(256).scan<'i', int>();

    args.add_argument("--angles")
        .help("Number of poses for trajectory")
        .default_value(0)
        .scan<'i', int>();

    args.add_argument("--arc")
        .help("Arc for trajectory (in degree)")
        .default_value(360)
        .scan<'i', int>();

    args.add_argument("--iters")
        .help("Number of iterations for solver")
        .default_value(10)
        .scan<'i', int>();

    args.add_argument("--analyze")
        .help("Analyze reconstruction")
        .default_value(false)
        .implicit_value(true);

    try {
        args.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << args;
        std::exit(1);
    }

    const elsa::index_t dims = args.get<int>("--dims");
    const elsa::index_t size = args.get<int>("--size");

    const auto num_angles = [&]() {
        if (args.is_used("--angles")) {
            return static_cast<elsa::index_t>(args.get<int>("--angles"));
        }
        return size;
    }();
    const elsa::index_t arc = args.get<int>("--arc");

    const elsa::index_t iters = args.get<int>("--iters");

    auto [recon, phantom] = [&]() {
        if (dims == 2) {
            return recon2d(size, num_angles, arc, iters);
        } else {
            return recon3d(size, num_angles, arc, iters);
        }
    }();

    if (args["--analyze"] == true) {
        analyze(phantom, recon);
    }

    return 0;
}
