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

add_executable(bench_lincomb benchmark_lincomb.cpp)
target_link_libraries(bench_lincomb PRIVATE nanobench elsa::all)
add_dependencies(benchmark bench_lincomb)

add_executable(bench_cublas benchmark_cublas.cpp)
target_link_libraries(bench_cublas PRIVATE nanobench elsa::all)
set_source_files_properties(benchmark_cublas.cpp PROPERTIES LANGUAGE CUDA)
add_dependencies(benchmark bench_cublas)
