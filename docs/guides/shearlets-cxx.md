Shearlets
---------

# WIP

Shearlets are a ... Its predecessors are wavelets and curevelets. Shearlets offer an optimally sparse representation,
meaning that ...

We offer band-limited cone-adapted shearlets meaning that we have compact support in the Fourier domain [1].

# TODO by spectra we refer to the generating functions psi and phi, related to the frequency domain (explain this well)

We will briefly explain the `ShearletTransform` component and go through an example.

# TODO say that it works only on one channel images

We use dyadic dilation (parabolic scaling) and shearing, meaning that a = X and s = Y. Also, translation invariant.

```c++
// TODO use another image? Maybe a free to use one https://commons.wikimedia.org/wiki/File:Raccoon_(Procyon_lotor)_3.jpg ?
// generate 2d phantom
IndexVector_t size(2);
size << 511, 511;
auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
const auto& volumeDescriptor = phantom.getDataDescriptor();

ShearletTransform<real_t> shearletTransform(size[0], size[1]);

Logger::get("Info")->info("Applying shearlet transform");
DataContainer<real_t> shearletCoefficients = shearletTransform.apply(phantom);

Logger::get("Info")->info("Applying inverse shearlet transform");
DataContainer<real_t> reconstruction = shearletTransform.applyAdjoint(shearletCoefficients);

// write the reconstruction out
EDF::write(reconstruction, "2dreconstruction_shearlet.edf");
```

# TODO applyAdjoint returns complex numbers, but we cut it out, refer to a paper that addresses this

The shearlet transform does not compute anything when first creating the object.

# TODO talk here about the other types of shearlets that we're going to offer, e.g. smooth shearlets (make these an enum?)

Applying the shearlet transform to a 2D image, generates the L layers of the same shape.

# TODO add here image of stacked spectra and stacked shearlet coefficients

Given that the spectra are only related to the shape of the image and number of scales, one can reuse such an object
depending on the context.

# TODO add here two images of frame correctness and reconstruction difference

### References

[1]

https://www.math.uh.edu/~dlabate/SHBookIntro.pdf
https://www.math.uh.edu/~dlabate/Athens.pdf
https://arxiv.org/pdf/1202.1773.pdf
