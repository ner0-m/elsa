Now with this foundation, let's dive a little deeper into some parts of our framework. Where appropriated, we'll give a few examples, which can we worked into the :ref:`previous example <2D Example>`.

##### Data Storage

As mentioned all data is stored in our :ref:`DataContainer <elsacore_datacontainer>`. All data is stored linearized. This linearizion is described by a :ref:`DataDescriptor <elsacore_datadescriptor>`. There are many ways we can store our data. For simple X-ray CT problems, it is usually sufficient to have one block. But for other problems or applications, it sometimes is necessary to have blocked data. By blocked, the data is still stored continuously in memory, but the interpretation and/or linearizion of the block can mean different things.

In general a `DataContainer` behaves very similar to a many vector classes from other linear algebra libraries. You can randomly access the container, you have many common math functions (dot product, some norms and others) and you can do many coefficient wise operations with multiple containers (i.e. sum, multiplication).

For some of the math operations, we support :ref:`expression templates <elsacore_expression>`. These enable us to have complicated expressions, and we have a minimum number of copies, resulting in an optimal memory usage and run time performance. :ref:`Check this out <elsacore_expression>`, for some more information on this topic, with an in-depth explanation of this feature.

##### Operator

Operators are the discretized representation :math:`\mathbf A` of :math:`\mathbf A(x) = b`. Therefore, it inherently describes the kind of problem we want to solve. For a X-ray tomography reconstruction you'll need a different one, compared to light field, to deblurring an image.

Operators are implemented using so called projectors. As they project the wanted object into the target space. Two very important aspects are performance in terms of time and quality. The projector can greatly influence the quality of reconstruction and the time it takes to perform it.

All of our current projectors are ray based. In other words, we approximate the projections as a straight line integral and approximate the line integral in different ways. Practically, the data is interpreted as a grid, through which rays are casted, traversed and the line integral approximated for each visited voxel[^fn]. Depending on the projector, the traversal and approximation is different. Currently, we support a simple binary projector, a Siddon based projector and a Josephs based projectors.

[^fn]: In this context, by voxel we mean a "3D pixel". But instead of naming both, we stick to the voxel formulation.

:Binary method:

The binary method is a very basic method. While traversing the grid, a simple voxel was hit, or not is being detected. This leads to quick run times, but to quite a lot of artifacts.

:Siddon's method:

Siddon's algorithms[^1] improves on the binary method, by taking the intersection length between the voxel and thew ray into consideration. This means the distance between the "entry" and "exit" of the current voxel and the ray. This leads to an improved image quality.

[^1]: Siddon, R. L. (1985). Fast calculation of the exact radiological path for a three-dimensional CT array. Medical Physics, 12(2), 252–255. doi:10.1118/1.595715

:Jospeh's method:

A more involved projector is based on the Joseph's algorithm [^2]. Based on the principal direction of the ray, an orthogonal interpolation direction is chosen. Then only steps in the principal direction are taken and the distance between the ray and neighboring voxel centers are used as a interpolation weight. This is a great amount of computation, however the results are improved. It often appears smoother compared to Siddon's method and with less artifacts. However, depending on the implementation, at certain angles, it can still lead to artifacts, as the principal direction is switched.

If you have a CUDA capable GPU, you can also use a the CUDA version of the Siddon's and Joseph's projectors. Especially, for bigger 3D problems, and some more iterations on the solver, this can greatly improve performance. For simple 2D cases with not to many iterations on the solver, it is often not necessary.

[^2]: Joseph, P. M. (1982). An Improved Algorithm for Reprojecting Rays through Pixel Images. IEEE Transactions on Medical Imaging, vol. 1, 192-196

So going back to our :ref:`original example <2D Example>`. You could swap the `SiddonsMethod` with `JosephsMethod`, or with `JosephsMethodCUDA` (make sure you compile with the `ELSA_BUILD_CUDA_PROJECTORS` CMake flag set to `ON`, and be sure to have CUDA set up properly).


##### Problem

You already saw the `WLSProblem` in action. Let's get a little more involved here as well.

>  :math:`\underset{x}{\operatorname{argmin}} \frac{1}{2} \| \mathbf A x - b \|_2^2`

This is the same equation as we had above. Let's look at this a little closer. :math:`Ax - b` is called a residual. A residual computes the error between two entities for each component. In this case between our measurements and :math:`x` projected into the measurement space. The :math:`\frac{1}{2} \| \cdotp \|_2^2` is a so called functional, in this case the `L2NormPow2`. A functional maps a vector to a scalar value.

Least squares are interesting because, you can 'append' [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics) terms. Let's do exactly that.

First let's create a functional (this code should be above the `WSLProblem problem = ...` line):

```cpp
L2NormPow2 regFunctional(phantom.getDataDescriptor()); // reg is short for regularization
real_t lambda = 0.5f;

RegularizationTerm regTerm (lambda, regularizationFunctional);
```

So in the first line, we create a functional, which maps x, to itself. `Lambda` is the so called regularization weight. Let's write the formulation to see, what is meant.

>  :math:`\underset{x}{\operatorname{argmin}} \frac{1}{2} \| \mathbf A x - b \|_2^2 + \lambda \| \mathbf x \|_2^2`

This is the so called :math:`\mathbf L_2` regularization, a special case of the Tikhonov regularization.

So let's use it instead of the `WSLProblem`:

```cpp
WLSProblem wlsProblem(projector, sinogram);
TikhonovProblem tikhonov(wlsProblem, regTerm);
```

And finally, construct the CG with the Tikhonov problem:

```cpp
CG solver(tikhonov);
```

Now there are some parameters to play with. You can compare it to the original problem, then you can also play with the :math:`\lambda` value. This adjusts the 'strength' of the regularization and is unique for every reconstruction. It can be a lot of work, to find an optimal value.

There are many forms of regularization, elsa let's you freely choose which functionals you arrange them according to your specific problem.

##### Solver

Our recommendation is to stick with the :ref:`CG <elsasolvers_cg>` algorithm. The algorithm assumes the problem formulated in quadric form:

>  :math:`\frac{1}{2} x^t \mathbf A x - x^tb`

and a symmetric positive-definite operator. This is different compared to the previous formulations. However, weighted least squares with and without Tikhonov regularization can be reformulated to quadric form. 

---
