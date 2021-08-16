Quickstart
----------

To demonstrate our framework, we're working through a simple exampled for Computed Tomography.
We'll be using the C++ interface of our framework. Please note, that this is a brief peak at the
code usage, not an introduction to tomographic reconstruction.

### 2D example

To make the examples short and less noisy, we first include all the headers of elsa and
pull the elsa namespace into scope.

```cpp
#include "elsa.h"
using namespace elsa;
```

The first thing, we have to do is set up some phantom data. Working with phantom data is quite
useful for many applications, as we have a ground truth and compute error norms.

```cpp
const auto size = IndexVector_t::Constant(2, 1, 256);
const auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
```

Here we settle on a 2D Shepp-Logan phantom of size 256x256. To have a look at the phantom,
you can write it to disk using the [PGM](https://de.wikipedia.org/wiki/Portable_Anymap) format,
like this:

```cpp
PGM::write(phantom, "phantom.pgm");
```

Next, we have to define a trajectory. The trajectory defines the position of the source and the
detector. In our case, this is a fan-beam geometry setup.

```cpp
const index_t numAngles = 180;
const index_t arc = 360;
const real_t distance = size(0);
const auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
    numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);
```

The trajectory has 180 positions (`numAngles`) over the whole circle (`arc`). The two distances are
the distance from the source to the center of the volume and the distance from the center of the volume
to the detector.
It's usually a good idea to move the source further away from the volume, and the detector closer to it.
This way also the edges of the phantom are covered nicely, and no artifacts appear during
reconstruction (but don't trust me, try it!).

Now, we need to create the sinogram. The sinogram is the result of projecting the phantom.

```cpp
const auto volumeDescriptor = phantom.getDataDescriptor();
SiddonsMethod projector(volumeDescriptor, *sinoDescriptor);
auto sinogram = projector.apply(phantom);

// Also write it to disk and have a look at it!
PGM::write(phantom, "sinogram.pgm");
```

The `projector` is the approximation of the Radon transform. Our current projectors are implemented
based on ray traversals.

With this setup, we finally have everything to setup the tomographic problem.

```cpp
const WLSProblem wlsProblem(projector, sinogram);
const CG solver(wlsProblem);
```

Without getting to into the math behind it (checkout the [paper](#citation), if you want to dive
into it), this sets up an optimization problem. We want to find an solution to this problem
In this case, we'll find it it using the
[Conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method), or CG for short.
It's an iterative algorithm to solve. As this is still a quite simple example, we don't need
to let it run for too many iterations.

```cpp
const index_t noIterations{20};
const auto reconstruction = solver.solve(noIterations);

PGM::write(reconstruction, "reconstruction.pgm");
```

Now, you have the reconstruction! In the best case, this should already look quite similar to the
original phantom. We can have a look at the difference and the L2-Norm:

```cpp
PGM::write(phantom - reconstruction, "reconstruction.pgm");
infoln("L2-Norm of the phantom: {:f.5}", phantom.l2Norm());
infoln("L2-Norm of the reconstruction: {:f.5}", reconstruction.l2Norm());
```

### 3D example

If you want to run a 3D reconstruction, all you need to change, is the initial size at the beginning.

```cpp
const auto size = IndexVector_t::Constant(3, 1, 64);
const auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
```

In this case, we create a 64x64x64 sized phantom, as it's already 4 times larger than the 2D 256x256
phantom, and therefore, will also take longer to compute.

With the current capabilities of the framework, you'll also have to remove or comment out the `PGM::write`
calls. We don't support 3D phantoms to be printed with it. If you still want to look at it, you can
replace the calls with `EDF::write`, which is better suited.

### CUDA projectors

To speed up the reconstruction up, we'll leverage CUDA in the next step. Be sure that you've set it up
and have a CUDA capable card.

elsa by default checks if you've got a working CUDA compiler, therefore, you should be good to go.
If something went wrong, pass CMake the flag: `ELSA_BUILD_CUDA_PROJECTORS=ON`. This should enable
the CUDA projectors.

In the above example, the `SiddondsMethod` projector was used. We'll replace it with the
`JosephsMethodCUDA`:

```cpp
JosephsMethodCUDA projector(volumeDescriptor, *sinoDescriptor);
```

The reconstruction should now be quite a bit faster.
