Quickstart
##########

To demonstrate our framework, we're working through a simple exampled for Computed Tomography.
We'll be using the C++ interface of our framework. Please note, that this is a brief peak at the
code usage, not an introduction to tomographic reconstruction.

2D example
**********

To make the examples short and less noisy, we first include all the headers of elsa and
pull the elsa namespace into scope.

.. literalinclude:: ../../examples/simple_recon2d.cpp
   :language: cpp
   :start-after: simplerecon header begin
   :end-before: simplerecon header end

Next we'll create the template to put all the code inside:

.. code-block:: cpp

   // The code from above
   #include "elsa.h"
   using namespace elsa

   void example2d() {
       // From now all the code goes here
   }

   int main() {
       example2d();
   }

The first thing, we have to do is set up some phantom data. Working with phantom data is quite
useful for many applications, as we have a ground truth and compute error norms.

.. literalinclude:: ../../examples/simple_recon2d.cpp
   :language: cpp
   :start-after: simplerecon phantom create
   :end-before: simplerecon phantom create

Here we settle on a 2D Shepp-Logan phantom. To have a look at the phantom, you can write it to
disk using the `PGM <https://de.wikipedia.org/wiki/Portable_Anymap>`_ format, like this:

.. literalinclude:: ../../examples/simple_recon2d.cpp
   :language: cpp
   :start-after: simplerecon phantom write
   :end-before: simplerecon phantom write

But if you change the file extension to `edf`, you can write it - well - to the `edf` file format.
`EDF <https://en.Wikipedia.org/wiki/European_Data_Format>`_ files can be read back in to elsa, PGM
files can not. PCM files are just meant to as a quick and easy way to debug reconstructions. If you
plan to further use the output, you should use EDF files.

From here, you can write all :code:`DataContainer`'s, like the sinogram or the reconstruction, we will
not explicitly add it here in the code, but you are encouraged to write the results to disc hand
have a look.

Next, we have to define a trajectory. The trajectory defines the position of the source and the
detector. In our case, this is a fan-beam geometry setup.

.. literalinclude:: ../../examples/simple_recon2d.cpp
   :language: cpp
   :start-after: simplerecon trajectory
   :end-before: simplerecon trajectory

The trajectory has 180 positions (`numAngles`) over the whole circle (`arc`). The two distances are
the distance from the source to the center of the volume and the distance from the center of the volume
to the detector. In this case we make the distance from the center to detector and source dependent
on the size of the volume. The values provided here are okay defaults.

It's usually a good idea to move the source further away from the volume, and the detector closer to it.
This way also the edges of the phantom are covered nicely, and no artifacts appear during
reconstruction (but don't trust me, try it!).

Now, we need to create the sinogram. The sinogram is the result of projecting the phantom.

.. literalinclude:: ../../examples/simple_recon2d.cpp
   :language: cpp
   :start-after: simplerecon sinogram
   :end-before: simplerecon sinogram

The `projector` is the approximation of the Radon transform. Our current projectors are implemented
based on ray traversals.

With this setup, we finally have everything to setup the tomographic problem.

.. literalinclude:: ../../examples/simple_recon2d.cpp
   :language: cpp
   :start-after: simplerecon solver
   :end-before: simplerecon solver

Without getting to into the math behind it (checkout the
`paper <https://doi.org/10.1117/12.2534833>`_ , if you want to dive into it), this sets up an
optimization problem. We want to find an solution to this problem In this case, we'll find it it
using the `Conjugate gradient method <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_,
or CGLS for short. It's an iterative algorithm to solve. As this is still a quite simple example, we
don't need to let it run for too many iterations.

Now, you have the reconstruction! In the best case, this should already look quite similar to the
original phantom. We can have a look at the difference and the L2-Norm:

.. literalinclude:: ../../examples/simple_recon2d.cpp
   :language: cpp
   :start-after: simplerecon analysis
   :end-before: simplerecon analysis

