Forward projection
------------------

A core part of tomographic reconstruction is the modality specific forward projection or forward
model. It is the abstraction of the physical imaging process. In the context of X-ray tomography, this models
the interaction of X-rays with matter. For X-ray attenuation computed tomography (CT), this forward
model describes the attenuation process of X-rays traversing matter.

Taking a step back, a typical setup for X-ray attenuation CT, like you find in the medical field,
consists of an X-ray source, the X-ray detector and the object of interest in between the two.
Then for a number of different positions of the X-ray source and detector, images are acquired. A
very typical trajectory of X-ray source and detector is a circular trajectory over an arc of 180°.

As we are doing some experiments, we will start with a typical example phantom. The [modified
Shepp-Logan phantom](https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom). In _elsa_, you can
create one simply by:

```python
# determine size
size = np.array([128, 128])
phantom = elsa.PhantomGenerator.createModifiedSheppLogan(size)

# This is used throughout the examples so we store it to a variable
volume_descriptor = phantom.getDataDescriptor()
```

Let us now create the circular trajectory of the X-ray imaging setup around the object:

```python
num_angles = 420
arc = 180

sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
    num_angles, phantom.getDataDescriptor(), arc, size[0] * 100, size[0])
```

The trajectory will consist of 420 positions over the arc of 180°. The last two arguments to the
trajectory are the distance of the source and the detector to the volume. In this case, the source
is quite a bit further away than the detector is from the volume. This will ensure proper
reconstruction of the corners of the volume (feel free to try using a smaller distance from the detector).

The next step is the simulation. As we work here in an experimental environment, with no real data,
we perform the forward projection manually to simulate the imaging process. However, with a real data set 
this step would not be necessary.

```python
projector = elsa.SiddonsMethod(volume_descriptor, sino_descriptor)

# simulate the sinogram
sinogram = projector.apply(phantom)
```

The measurements of an X-ray attenuation CT are often referred to as a _sinogram_. This is due the
sinusoidal shape of the projection of a single point in the object of interest. The sinogram is not
an image, rather, it is a stack of measurements. In the two-dimensional case, each column is an
actual measurement. Let's have a look at the complete sinogram:

```python
plt.imshow(np.array(sinogram))
plt.show()
```

Though the output is quite busy, you will see the sinusoidal patterns in the sinogram. Bright
colors represent parts of high attenuation and darker colors areas of lower attenuation.

#### GPU projection

For 2D examples this section is not as important, however for 3D examples it really is. The projection 
can be efficiently implemented using GPUs and is easily supported in _elsa_. In the above example, all
you would need to change is the projector setup to:

```python
projector = elsa.SiddonsMethodCUDA(volume_descriptor, sino_descriptor)
```

To check if you can use the GPU projectors on your machine, you can use the
`cudaProjectorsEnabled()` function, which will return `True` if you can use the projectors
(in other words: if you have a CUDA-capable GPU with working drivers, and elsa compiled with
the appropriate CUDA settings).

#### 3D projection

From the above examples, all you would need to change is the input size of the phantom. All other
aspects are adapted automatically, such that minimal code changes are necessary.

```python
# determine size
size = np.array([128, 128, 128])
phantom = elsa.PhantomGenerator.createModifiedSheppLogan(size)
```

Now the returned sinogram is a stack of 2D slices. Each slice is a measurement taken at a
certain position. However, visualization of such a 3D stack is a little tricky. For the sinogram, 
you can use a simple
[image slice viewer](https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html)
like this:

```python
class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title("use scroll wheel to navigate images")

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def on_scroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 10) % self.slices
        else:
            self.ind = (self.ind - 10) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel(f"Pose {self.ind} of {self.slices}")
        self.im.axes.figure.canvas.draw()
```

Then to show the 3D sinogram:

```python
fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, np.array(sinogram))
fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
plt.show()
```

Now you can scroll through the sinogram and have a look at each of the measurements.

#### Projection methods

So far we have used one particular projection method, often called Siddon's method. This method
computes the radiological path of X-rays, i.e. the intersection length of the X-ray and the object
of interest. This is a discretized mathematical description of the forward model. However, the
forward model assumes infinitely many X-rays, which is computationally infeasible, of course.

Therefore, other methods have been proposed. One other very frequently used one is the so-called
Joseph's method. This method yields better results as it considers adjacent pixels and hence
doesn't assume an infinitely thin X-ray. The above examples only need one line of change:

```python
projector = elsa.JosephsMethod(volume_descriptor, sino_descriptor)
```

Of course, a CUDA version is available as well.

You are encouraged to try the different methods and see if you can sport accuracy or runtime
differences!
