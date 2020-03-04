This should help you getting started setting up very simple problems and give you a first look at our API.

##### What is this framework for?

Our intended use is tomographic reconstruction. Tomographic reconstruction is a type of problem where, from a finite number of projections, a result has to be computed. A typical application is X-ray computed tomography (CT). But other applications such as light field microscopy, or different image processing techniques (deconvolution, deblurring, multiple view reconstruction) fall into a similar category of problem.

The framework currently supports, so called operators for, X-ray CT out of the box. Different applications could be supported by implementing appropriate operators (and for optimal results some other functionality as well).

##### 2D Example

Let's start with the complete sample code and then break it down.

```cpp
#include "elsa.h"

using namespace elsa;

int main()
{
    // Generate 2D Sheep Logan phantom
    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

    // Generate circular trajectory
    index_t numAngles{180}, arc{360};
    auto [geometry, sinoDescriptor] = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, 0 * 100, 0);

    // Setup operator for 2D X-ray transform
    Logger::get("Info")->info("Simulating sinogram using Siddon's method");
    SiddonsMethod projector(phantom.getDataDescriptor(), *sinoDescriptor, geometry);

    // Simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // Setup reconstruction problem
    WLSProblem problem(projector, sinogram);

    // Solve the reconstruction problem
    CG solver(problem);

    index_t noIterations{20};
    auto reconstruction = solver.solve(20);

    // Write the reconstruction out
    EDF::write(reconstruction, "2dreconstruction.edf");
}
```
###### Walkthrough

Let's start at the top

```cpp
#include "elsa.h"

using namespace elsa;
```

The `elsa.h` header is our convenience header. It includes all important parts of elsa. For the example here, we're also introducting all names from elsa into our scope, to not clutter up everything in the example code.

```cpp
IndexVector_t size(2);
size << 128, 128;
auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
```

In `main`, we create [Shepp Logan phantom](https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom) with the size of 128x128. This phantom is widely used in research to compare different algorithms. As a ground truth is known, error margins can be computed and it's easier to compare with other research. Further, it's easy to change the size, to keep run time down during development. 

The type returned by the function is a `DataContainer`. It is an important types of our framework, as it handles our data storage, layout and much of it's computation. It is basically a mathematical vector, but it further takes care of transfers from and to GPU and lazy evaluation (through expression templates).

The phantom mimics the actual scanned object. In the code, and comments, we often refer to the space of this object as _domain_. Next, we have to setup the trajectory a real CT scanner would take:


```cpp
index_t numAngles{180}, arc{360};
auto [geometry, sinoDescriptor] = CircleTrajectoryGenerator(`CircleTrajectoryGenerator`::createTrajectory(
    numAngles, phantom.getDataDescriptor(), arc, size(0) * 100, size(0));
```

In this case, we create a 360° (`arc`) circular trajectory (`CircleTrajectoryGenerator`), with 180 projections (`numAngles`). Figuratively speaking, over a full circle we're take 180 'pictures' with the CT scanner.

With this setup, we're also fixing the size of our sinogram. A sinogram is the visual representation of the raw data obtained from a CT scan. We usually refer to this space as the _range_. The size of the sinogram mostly depends on the size of the domain and the number of projections.

```cpp
auto sinogram = projector.apply(phantom);
```

This actually computes the sinogram. It is performing an approximation of the [Radon transform](https://en.wikipedia.org/wiki/Radon_transform). This setup was necessary to work with fake data. Now the sinogram corresponds to something we could get from a CT scanner and read into our framework.

Now, we can work on our reconstruction. First, let's setup a problem.

```cpp
WLSProblem problem(projector, sinogram);
```

This describes a weighted least squares problem. In the next chapter of the getting started guide, we have more stuff on the background, you can look :ref:`check it out <elsa_background>` for more formal description. The `WSLProblem` is one way of defining the problem formulation. Without it, we could not know, what problem we want to solve.

```cpp
CG solver(problem);

index_t noIterations{20};
auto reconstruction = solver.solve(20);
```

This creates a solver. In this particular case we use the [conjugate gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method). Others could be used. Then we run the algorithm for 20 iterations and get a the reconstructed object as output.
    
```cpp
EDF::write(reconstruction, "2dreconstruction.edf");
```

This is the final step of the getting started. We write out the reconstructed object as a edf file.