# Reconstruction of X-ray CT data

So far, this guide as only dealt with synthetic data. Synthetic data is useful for many experiments
and analysis. However, in the end real datasets needs to be reconstructed. They are noisy, imperfect
and most of them need some form of special handling. In this guide, we will walk through an example
dataset and how to reconstruct it using iterative algorithms.


#### Dataset

The dataset used in this guide, is [this seashell](https://zenodo.org/record/6983008) dataset
created by the Finnish Inverse Problems Society and is published under the Creative Commons
Attribution 4.0 International. To following along, please download the dataset and extract the zip
file to some location you can access.

In principal any other dataset can be used, and most of the concepts discussed here are important
for all of them. However, many dataset are slightly different and might need special handling.
Hence, I recommend following along with the same dataset.

The dataset was created by a cone-beam CT scanner. The detector is a flat plane detector with
2240x2378 pixel. In total 721 different positions in a circular trajectory were acquired with an
angle increment of 0.5 degrees between each pose. In this particular case, some post-processing was
already perform. 100 Dark current images (i.e. images without any sample in the detector) were
acquired, then averaged and applied to the projection data. Additionally, flat-field corrections
were applied.

#### Preprocessing

The first step is to load the dataset. This particular dataset is given in 'tiff' files.
We will create a function, which will load each file and apply the preprocessing steps. If you do
not want to load the complete dataset, you can increase the step size (i.e. only loading every
second, third and so on image).

```py
def load_dataset(path=".", step=1, binning=None, reload=False):
    num_images = 722
    projections = None

    # Load dataset indead
    if not reload and os.path.exists(f"{path}/sinogram.npy"):
        print("Reading sinogram from disk")
        return np.load("sinogram.npy")

    for i in range(1, num_images, step):
        filename = f"{path}/20211124_seashell_{str(i).zfill(4)}.tif"

        print(f"Processing {filename}")
        raw = plt.imread(filename).astype(np.float16)

        # Binning reduces the resolution of the image
        if projections is None:
            shape = None
            lastdim = num_images // stepeverything is stored as 16-bit floating point values to
            reduce the
            if binning:
                shape = tuple(np.asarray(raw.shape) // binning) + (lastdim,)
            else:
                shape = raw.shape + (lastdim,)

            projections = np.empty(shape, dtype=np.float16)

        projections[:, :, i] = preprocess(raw, binning=binning)

    np.save("{path}/sinogram.npy", projections)

    return projections

def preprocess(proj, binning=None):
    return proj
```

The `load_dataset` function as mentioned loads every tiff file and calls the preprocess method on
it. Some steps are already present here, which will be discussed later, such as binning. Also, a
simple reload mechanism is buildin, to quickly iterate on the reconstruction part later. Also note,
everything is stored as 16-bit floating point values to reduce the overall memory consumption.
Loading the complete dataset using 16-bit floating point values, already requieres around 7.6 GB of
data. Single precision floating point values would then already be over 15 GB of data. Hence, it is
recommended to work with a machine, with at least 32GB of RAM. Finally, the `preprocess` function
currently just returns the projection argument for now.

###### Negative Log-transform

First, have a look at any image:

```py
# Adjust path if necessary, if the process gets killed because of memory, add a step parameter
projections = load_dataset()

I = plt.imread(projections[:, :, 123], cmap="gray", norm="log")

plt.imshow(I)
plt.show()
```

You will notice something weird. The shell is dark and the background is bright, i.e. the dark parts
have low signal and the bright part have a high intensity. For a projection you'd expect it the
other way around. The shell is more absorbent and hence it should be brighter, and the background
dark. The detector measures the amount of photons arriving at the detector, not the attenuation.
So, we need to deal with that somehow. A simple inversion is wrong, So, how is it done correctly?
Let's look at a measurement $I_1$ and how it connects to attenuation $\mu$. They are given by the
line integral along the line $L$:


```math
\int_L \mu(x) dx
```

The measured intensity is connected to the initial intensity and the line integral the following
way:

```math
I_1 = I_0 exp(- \int_0^s \mu(x)dx)
```

Rearranging a bit and taking the logarithm, you get the following formulation:
```math
- \log ( \frac{I_1}{I_0} ) = \int_L \mu(x)dx
```

Doing this formulation for every pixel of the detector will result in the actual projection data.
To get the initial intensity, we will just use a patch from the raw measurement data. Let's have a
look at the processing step for a single projection image:


```py
def preprocess(proj, binning=None):
    # Take a part of the background
    background = proj[:128, :128]
    I0 = np.mean(background)

    # reduce some noise
    proj[proj > I0] = I0

    # log-transform of projection
    proj = -np.log(proj / I0)

    return proj
```

First, we extract a patch from the background to compute $I_0$. Next, we clamp all values below
$I_0$ to that. This just reduces some noise, and prevents unphysical quantities of negative values.
Finally, we perform the negative log transformation and that is it. Have a look at the output image.
Now, you will see the background is dark, as it does not attenuate anything, but the seashell is
bright. Also, now you do not need the `norm=log` parameter to visualize the projection data properly
anymore. The `binning` parameter will be used in a second.

###### Normalization

The next very simple step will just normalize all projection images to a range from 0 to 1. Add the
following snippet in the `load_dataset()` function, after the loop, right before saving the
projection data:

```py
projections /= np.max(projections)
```

###### Binning

Another preprocessing step introduced here is binning. Binning reduces the spatial resolution of the
projection data and lowers the electron noise in the signal. Binning basically averages multiple
detector pixel to one virtual detector pixel.

So for each measurement, we average a group of `n x n` pixels, where `n` is the binning factor. This
usually is a power of two. The `preprocess` function should be expanded the following way:

```py
def preprocess(proj, binning=None):
    # Take a part of the background
    background = proj[:128, :128]

    # If binning should be performed, do it now
    if binning:
        proj = rebin2d(proj, binning)
        background = rebin2d(background, binning)

    # Same as previously
    #...

    return proj
```

The actual binning is performed the following way:

```py
def rebin2d(arr, binfac):
    """Credits to: https://scipython.com/blog/binning-a-2d-array-in-numpy/ """

    nx, ny = arr.shape
    bnx, bny = nx // binfac, ny // binfac
    shape = (bnx, arr.shape[0] // bnx, bny, arr.shape[1] // bny)
    return arr.reshape(shape).mean(-1).mean(1)
```

This function reshapes the array, and then means first over the last, and then over the second
dimension. This is quite nifty NumPy code, convince yourself that is does what you'd expect!

###### Shift

A final small correction noted in the dataset is needed. Due to a misaligned center of rotation of
the scanner, a shift to the left by 4 pixels is needed to produce better results. This has been
observed empirically and is nothing you can show mathematically. Add the following line to the
beginning of the `preprocess` function:

```py
proj = np.roll(proj, -4, axis=1)
```

#### Geometric setup

After all of that, we need to simulate the geometry setup and the volume we want to reconstruct.
First, we need to decide for a resolution of our reconstruction volume. For *elsa*, we need to
describe it using the `VolumeDescriptor`:

```py
volume_descriptor = elsa.VolumeDescriptor([128] * 3, [0.02] * 3)
```

Here, we settle on a relatively small volume (such that computation doesn't take hours or we run out
of memory). The spacing (second argument) is set such that the region of interest is well covered
in the reconstruction.

The measurements were taken in a circular trajectory of a full circle, hence we will be using the
`CircleTrajectoryGenerator`. Next, we need the number of position images were captured at. We can
read this of the size of the last dimension of the sinogram data. Then we need the distance from
source to origin, and distance from origin to the principal point of the detector. The origin is the
center of the volume. And the principal point is the center point of the detector. In this case, we
are given the distance from source to origin and source to detector, so we just need to calculate
the distance from origin to detector. Finally, we also need the size of the detector and the
spacing. The former is just the first two dimensions of the shape, the later is given in the text
file of the dataset. A function to create this looks like this:

```py
def create_sinogram(projections, volume_descriptor, binning=2):
    num_angles = projections.shape[-1]

    # Distances given by txt file
    sourceToDetector = 553.74
    sourceToOrigin = 210.66

    detectorSpacing = np.asarray([0.05, 0.05]) * binning
    detectorSize = projections.shape[:-1]

    originToDetector = sourceToDetector - sourceToOrigin

    sino_descriptor = elsa.CircleTrajectoryGenerator.fromAngularIncrement(
        num_angles,
        volume_descriptor,
        0.5,               # Angular increment
        sourceToOrigin,
        originToDetector,
        [0, 0],            # Offset of principal point
        [0, 0, 0],         # Offset of origin
        detectorSize,
        detectorSpacing,
    )

    return elsa.DataContainer(projections[:, :, :num_angles], sino_descriptor)

```

A small note, it might be a good idea to release the projection data as we have it stored in an elsa
container now.


#### Reconstruction

Now, we need to perform a reconstruction. A couple of aspects need to be determined first. We need
to choose a projector (simulating the X-ray forward projection). For such a large 3D dataset you
should prefer a CUDA projector. Here, we will choose the `JosephsMethodCUDA`:

```py
projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor)
```

Next, we must choose the kind of reconstruction we want. We could again use an analytical
filtered back-projection (FBP) similar to the one implemented in the previous parts of this guide.
But we will choose an iterative reconstruction algorithms. They generally outperform the FBP, at the
cost of longer run-times. There are many different aspect to choose an algorithm and problem
formulation. Here, we will choose a least squares formulation with total variation (TV)
regularization.
TV regularization favours few transitions in the gradient of the reconstructed image, i.e. it
penalizes many transitions in the gradient. This leads to a piecewise smooth or constant image. In
extreme cases, it might look a little cartoonish. However, it is an excellent regularization method,
and considered state-of-the-art.

The exact problem formulation is the following:

```math
\min_x \frac{1}{2} || Ax - b ||_2^2 + \lambda || \nabla x ||_1
```

Here, $A$ is the projector (`JosephsMethodCUDA`), $x$ is the unknown volume we try to reconstruct,
$b$ is the measured data, $\nabla$ the (numerical) gradient of the unknown, and $\lambda$ is the
regularization parameter, balancing the contribution of the regularization.

Such a problem, is often referred to as $L^1$ regularization or *LASSO*, and many different
algorithms exists, which solve this kind of problem. For this, we will choose `FISTA`. A function
taking a couple of parameters and returning a reconstruction using TV-regularization looks like this:

```py
def reconstruct(projector, sinogram, iters=5, weight=50):
    # Create the dataterm
    wls = elsa.WLSProblem(projector, sinogram)

    # setup regularization term
    finite_differences = elsa.FiniteDifferences(projector.getDomainDescriptor())
    tv = elsa.L1Norm(elsa.LinearResidual(finite_differences))
    tvreg = elsa.RegularizationTerm(weight, tv)

    # setup LASSO problem
    problem = elsa.LASSOProblem(wls, tvreg)

    # set solver and return a NumPy array of the reconstruction
    solver = elsa.FISTA(problem)
    return np.asarray(solver.solve(iters))
```

The call size will look like this:

```py
recon = reconstruct(projector, sinogram)
```

You will need to play around with the `weight` parameter a bit. A to lower value will not regularize
the problem, however a too large value will suppress important details. Usually, good preliminary
results can be seen after 5 - 10 iterations.
