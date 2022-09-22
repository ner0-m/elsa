Filtered backprojection
-----------------------

### Background

The most common reconstruction method is the so called _filtered backprojection_ (FBP). The idea is
simple, the basic reconstruction using only a backprojection is blurry, hence perform a filtering
step. That's it, but there is on key aspect. Filter the projection data, i.e. the sinogram, instead
of the backprojection. Then backproject the filtered sinogram.

### Implementation

Let us implement this method quickly. First, we need a fast Fourier transformation (FFT)
implementation. _elsa_ has one, but it's a little more cumbersome than what is necessary here. Hence,
we will use SciPy's FFT. Import it with:

```python
from scipy.fft import fft, ifft
```

Now we need to perform some padding to ensure the FFT works properly:

```python
# Convert to numpy array
np_sinogram = np.array(sinogram)
sinogram_shape = np_sinogram.shape[0]

# Add padding
projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * sinogram_shape))))
pad_width = ((0, projection_size_padded - sinogram_shape), (0, 0))
padded_sinogram = np.pad(np_sinogram, pad_width, mode="constant", constant_values=0)
```

Note that this step is to some extent optional. It would be sufficient to round to the next even
number for the padding. However, the reconstruction will be worse. Try it for yourself, replace
the line regarding padding with something like: `int(np.ceil(sinogram_shape / 2)) * 2`, and check
the reconstruction.

The next step is to create the filter. This will determine which frequencies one wishes to suppress
and by how much. The simplest filter, is the ideal Ramp filter. It will suppress high frequency
noise, compared to lower frequency noise. Interestingly, from a mathematical point of view, this is
the exact solution. So let us create that

```python
def ramp_filter(size):
    n = np.concatenate(
        (
            # increasing range from 1 to size/2, and again down to 1, step size 2
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # See "Principles of Computerized Tomographic Imaging" by Avinash C. Kak and Malcolm Slaney,
    # Chap 3. Equation 61, for a detailed description of these steps
    return 2 * np.real(fft(f))[:, np.newaxis]

# Ramp filter
fourier_filter = ramp_filter(projection_size_padded)
```

Interested readers are encouraged to read the given reference to see more details. These details
would be out of scope for this guide. Now, we need to perform the convolution between the sinogram
and the filter. In Fourier space, this is just a component wise multiplication, then apply the
inverse Fourier transform:

```python
projection = fft(padded_sinogram, axis=0) * fourier_filter
filtered_sinogram = np.real(ifft(projection, axis=0)[:sinogram_shape, :])
```

Now, finally we perform the backprojection with the filtered sinogram.

```python
# Go back into elsa space
filtered_sinogram = elsa.DataContainer(filtered_sinogram, sino_descriptor)

# Do the backprojection:
filtered_backprojection = projector.applyAdjoint(filtered_sinogram)

plt.imshow(np.array(filtered_backprojection))
plt.show()
```

This is a good enough looking reconstruction. Different filters, can improve the results further,
especially certain scenarios of noise might be handled better.

### Flaws of the Filtered Backprojection

The FBP is the de facto standard in commercial X-ray CT scanners. This is due to its efficiency. The
first algorithms developed for X-ray tomographic reconstruction were iterative reconstruction
algorithms. However, due to the immense amount of computation necessary for iterative reconstructions
to be successful, they were not practical. Hence, the efficiency of the FBP was key in the beginning.

However, the FBP breaks down in so-called _low dose_ scenarios. These are, for example, scenarios where
the number of projection angles is low. You can try this out, change the trajectory given above, to
instead of 420 angles, only use, say, 60. The artifacts in the reconstruction take over quickly, and
as you will see in the next chapters, iterative reconstruction algorithms can deal with these kinds
of scenarios better.

Another low dose scenario is reducing the tube current, resulting in very noise projection images.
Also for this scenario, iterative reconstruction algorithms, particular ones with regularization,
typically perform better than FBP.

Low dose reconstruction is tremendously important, as a high dosage of X-ray radiation is dangerous
and has health risks associated with it. Hence, in the medical field it is important to reduce
the dosage to a necessary minimum.

Another aspect is that the FBP as described here is 2D only. There exists a 3D extension (FDK) to it,
which works similarly to the FBP and is still quite efficient. However, iterative algorithms are
agnostic of the dimensions. And with the rise of GPU programming, iterative algorithms have become
practically more feasible. Much research in the field is focused in iterative reconstructions, however,
as mentioned above, until today, commercial CT scanners still mostly rely on the FBP (FDK).
