elsa offers a fully GPU based DataContainer. This option requires compilation with
```
-DELSA_CUDA_VECTOR=ON
```
Additionally, a matching version of CUDA as well as clang has to be available. This is necessary because of C++17 in device code which is not supported by nvcc. Two tested combinations on Ubuntu 18.04 are:

| clang | CUDA | compute capability |
| ------ | ------ | ----- |
| 8 | 10.0 | up to 7.5 |
| 8 | 9.2 | up to 7 |

Similarly to the `DataHandlerGPU` which uses Eigen, the `DataHandlerGPU` uses the [Quickvec](https://gitlab.lrz.de/IP/quickvec) project. The compile flags will be set automatically.

Enabling the compile-time option `ELSA_CUDA_VECTOR` will set the default `DataContainer` type to use the GPU.

Keep in mind that running full reconstruction examples with large volumes requires large amounts of memory (which the GPU might not have depending on your hardware). As a rough estimate, running a reconstruction task with $700^3$ volume elements requires around 10 GB of memory, as multiple copies have to be created.

On newer GPUs, oversubscription of the GPU memory through the unified memory architecture is possible. However, this will slow down the task at hand considerably to the point that a CPU-only version in main memory might be faster.
