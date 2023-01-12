# Create binaries for benchmarks in bin/benchmarks
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks)

CPMAddPackage(
    NAME nanobench
    GITHUB_REPOSITORY martinus/nanobench
    VERSION 4.3.7
    GIT_SHALLOW TRUE
)

add_custom_target(benchmark)

add_executable(bench_forward benchmark_forward.cpp)
target_link_libraries(bench_forward PRIVATE nanobench elsa::all)
add_dependencies(benchmark bench_forward)

add_executable(bench_backward benchmark_backward.cpp)
target_link_libraries(bench_backward PRIVATE nanobench elsa::all)
add_dependencies(benchmark bench_backward)

add_executable(bench_norms benchmark_norms.cpp)
target_link_libraries(bench_norms PRIVATE nanobench elsa::all)
add_dependencies(benchmark bench_norms)

add_executable(bench_shepp_logan benchmark_shepp_logan.cpp)
target_link_libraries(bench_shepp_logan PRIVATE nanobench elsa::all)
add_dependencies(benchmark bench_shepp_logan)

add_executable(bench_forbild benchmark_forbild.cpp)
target_link_libraries(bench_forbild PRIVATE nanobench elsa::all)
add_dependencies(benchmark bench_forbild)

add_executable(bench_shapes benchmark_shapes.cpp)
target_link_libraries(bench_shapes PRIVATE nanobench elsa::all)
add_dependencies(benchmark bench_shapes)
