#include "elsa.h"
#include "LutProjector.h"
#include "Utilities/Statistics.hpp"
#include "IO.h"
#include "spdlog/fmt/bundled/core.h"

#include <argparse/argparse.hpp>

elsa::DataContainer<elsa::real_t> get_phantom(elsa::index_t dims, elsa::index_t s,
                                              std::string phantom_kind)
{
    const auto size = elsa::IndexVector_t::Constant(dims, s);

    if (phantom_kind == "SheppLogan") {
        return elsa::phantoms::modifiedSheppLogan(size);
    } else if (phantom_kind == "Rectangle") {
        auto quarter = s / 4;
        const auto lower = elsa::IndexVector_t::Constant(size.size(), quarter);
        const auto upper = elsa::IndexVector_t::Constant(size.size(), s - quarter);
        return elsa::phantoms::rectangle(size, lower, upper);
    } else if (phantom_kind == "Circle") {
        return elsa::phantoms::circular(size, s / 4.f);
    }
    throw elsa::Error("Unknown phantom kind {}", phantom_kind);
}

std::unique_ptr<elsa::LinearOperator<elsa::real_t>>
    get_projector(std::string projector_kind, const elsa::DataDescriptor& volume,
                  const elsa::DataDescriptor& sinogram)
{
    const auto& vol = dynamic_cast<const elsa::VolumeDescriptor&>(volume);
    const auto& sino = dynamic_cast<const elsa::DetectorDescriptor&>(sinogram);

    if (projector_kind == "Blob") {
        elsa::BlobProjector<elsa::real_t> projector(vol, sino);
        return projector.clone();
    } else if (projector_kind == "BSpline") {
        elsa::BSplineProjector<elsa::real_t> projector(vol, sino);
        return projector.clone();
    } else if (projector_kind == "Siddon") {
        elsa::SiddonsMethod<elsa::real_t> projector(vol, sino);
        return projector.clone();
    } else if (projector_kind == "Joseph") {
        elsa::JosephsMethod<elsa::real_t> projector(vol, sino);
        return projector.clone();
    }
    throw elsa::Error("Unknown projector {}", projector_kind);
}

elsa::DataContainer<elsa::real_t> compute_sinogram(const elsa::DataContainer<elsa::real_t> phantom,
                                                   elsa::index_t numAngles, elsa::index_t arc,
                                                   std::string forward_projector)
{
    const auto size = phantom.getDataDescriptor().getNumberOfCoefficientsPerDimension();
    auto& volumeDescriptor = phantom.getDataDescriptor();

    const auto dims = size.size();

    // generate circular trajectory
    const auto distance = static_cast<elsa::real_t>(size(0));
    auto sinoDescriptor = elsa::CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // Don't commit the inverse crim (i.e, use a different projector for the initial forward
    // projection)
    auto forward_ptr =
        get_projector(forward_projector, phantom.getDataDescriptor(), *sinoDescriptor);
    auto& forward = *forward_ptr;

    // simulate the sinogram
    elsa::Logger::get("Info")->info("Calculate sinogram using {}-Projector", forward_projector);
    auto sinogram = forward.apply(phantom);

    elsa::io::write(sinogram, fmt::format("{}dsinogram_{}.edf", dims, forward_projector));

    if (dims == 2) {
        elsa::io::write(sinogram, fmt::format("{}dsinogram_{}.pgm", dims, forward_projector));
    } else if (dims == 3) {
        for (int i = 0; i < size[dims - 1]; ++i) {
            elsa::io::write(sinogram.slice(i),
                            fmt::format("{}dsinogram_{:02}_{}.pgm", dims, i, forward_projector));
        }
    }

    return sinogram;
}

std::unique_ptr<elsa::Solver<elsa::real_t>>
    get_solver(std::string solver_kind, const elsa::LinearOperator<elsa::real_t>& projector,
               const elsa::DataContainer<elsa::real_t>& sinogram)
{
    if (solver_kind == "CG") {
        elsa::TikhonovProblem problem(projector, sinogram, 0.05);
        elsa::CG solver(problem);
        return solver.clone();
    } else if (solver_kind == "ISTA") {
        elsa::LASSOProblem problem(projector, sinogram);
        elsa::ISTA solver(problem);
        return solver.clone();
    } else if (solver_kind == "FISTA") {
        elsa::LASSOProblem problem(projector, sinogram);
        elsa::FISTA solver(problem);
        return solver.clone();
    } else {
        throw elsa::Error("Unknown Solver {}", solver_kind);
    }
}

elsa::DataContainer<elsa::real_t> reconstruct(const elsa::DataContainer<elsa::real_t>& sinogram,
                                              const elsa::LinearOperator<elsa::real_t>& projector,
                                              std::string projector_kind, elsa::index_t iters,
                                              std::string solver_kind)
{
    const auto size = sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension();
    const auto dims = size.size();

    // setup reconstruction problem
    elsa::Logger::get("Info")->info("Setting up solver {}", solver_kind);
    auto solver_ptr = get_solver(solver_kind, projector, sinogram);
    auto& solver = *solver_ptr;

    elsa::Logger::get("Info")->info("Start reconstruction");
    auto reconstruction = solver.solve(iters);

    elsa::io::write(reconstruction, fmt::format("{}dreconstruction_{}.edf", dims, projector_kind));

    if (dims == 2) {
        elsa::io::write(reconstruction,
                        fmt::format("{}dreconstruction_{}.pgm", dims, projector_kind));
    } else if (dims == 3) {
        for (int i = 0; i < size[dims - 1]; ++i) {
            elsa::io::write(reconstruction.slice(i),
                            fmt::format("{}dreconstruction_{:02}_{}.pgm", dims, i, projector_kind));
        }
    }

    return reconstruction;
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
    elsa::DataContainer diff = a - b;
    return diff.l2Norm() / a.getSize();
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

    args.add_argument("--verbose")
        .help("increase output verbosity")
        .default_value(false)
        .implicit_value(true);

    args.add_argument("--no-recon")
        .help("Only perform forward projection and no reconstruction")
        .default_value(false)
        .implicit_value(true);

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

    args.add_argument("--projector")
        .help("Projector to use for reconstruction (\"Blob\", \"Siddon\", \"Joseph\")")
        .default_value(std::string("Blob"));

    args.add_argument("--forward")
        .help("Choose different projector for forward proj (\"Blob\", \"Siddon\", \"Joseph\")")
        .default_value(std::string("Joseph"));

    args.add_argument("--solver")
        .help("Choose different solver (\"CG\", \"ISTA\", \"FISTA\")")
        .default_value(std::string("CG"));

    args.add_argument("--phantom")
        .help("Choose different solver (\"SheppLogan\", \"Rectangle\", \"Circle\")")
        .default_value(std::string("SheppLogan"));

    args.add_argument("--analyze")
        .help("Analyze reconstruction")
        .default_value(false)
        .implicit_value(true);

    args.add_argument("--baseline")
        .help("Give a baseline file with which the current reconstruction is compared");

    try {
        args.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << args;
        std::exit(1);
    }

    if (args["--verbose"] == true) {
        elsa::Logger::setLevel(elsa::Logger::LogLevel::DEBUG);
        elsa::Logger::get("main")->info("Verbose output enabled");
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

    const auto projector_kind = args.get<std::string>("--projector");

    const auto forward_projector = [&]() {
        if (args.is_used("--forward")) {
            return args.get<std::string>("--forward");
        }
        return projector_kind;
    }();

    const auto solver_kind = args.get<std::string>("--solver");

    const auto phantom_kind = args.get<std::string>("--phantom");

    /// reconstruction setup
    const auto phantom = get_phantom(dims, size, phantom_kind);

    // write the phantom out
    if (dims == 2) {
        elsa::io::write(phantom, fmt::format("{}dphantom.pgm", dims));
    }
    elsa::io::write(phantom, fmt::format("{}dphantom.edf", dims));

    auto sinogram = compute_sinogram(phantom, num_angles, arc, forward_projector);

    if (args["--no-recon"] == false) {

        // Create projector for reconstruction
        elsa::Logger::get("Info")->info("Create {}-Projector", projector_kind);
        auto projector_ptr = get_projector(projector_kind, phantom.getDataDescriptor(),
                                           sinogram.getDataDescriptor());
        auto& projector = *projector_ptr;

        auto recon = reconstruct(sinogram, projector, projector_kind, iters, solver_kind);

        // if (args["--analyze"] == true) {
        //     analyze(phantom, recon);
        //
        //     if (args.is_used("--baseline")) {
        //         const auto baseline_file = args.get<std::string>("--baseline");
        //         const auto baseline = elsa::io::read<elsa::real_t>(baseline_file);
        //         analyze(baseline, recon);
        //     }
        // }
    }

    return 0;
}
