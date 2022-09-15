Python Bindings
-----------------------------

_elsa_ also comes with Python bindings for almost all aspects of the framework.
This short guide aims to give an introduction into some simple use cases and explain how one can
easily translate C++ code into Python code for faster prototyping and experimenting.
One major benefit that comes with the Python bindings is that we are able to natively
use numpy arrays with our elsa data containers, making it easy to work with other tools such as
matplotlib.

### Setup the python bindings
Once you have cloned _elsa_, change into the directory and simply run `pip install .`. This will build
_elsa_ and install it including the Python bindings. In case you wonder what is happening, add the
`--verbose` flag to see the progress.

Once everything is set up, simply open a Python interpreter and run
```python
import pyelsa as elsa
```
to check that everything is working.

### 2D example
To give a short introduction into the Python usage of elsa we will recreate the 2D example of the
:doc:`./quickstart-cxx` section in Python.

```python
import pyelsa as elsa
import numpy as np
import matplotlib.pyplot as plt

size = np.array([128, 128])
phantom = elsa.PhantomGenerator.createModifiedSheppLogan(size)
volume_descriptor = phantom.getDataDescriptor()

# generate circular trajectory
num_angles = 180
arc = 360

sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
    num_angles, phantom.getDataDescriptor(), arc, size[0] * 100, size[0])

# setup operator for 2d X-ray transform
projector = elsa.SiddonsMethod(volume_descriptor, sino_descriptor)

# simulate the sinogram
sinogram = projector.apply(phantom)

# setup reconstruction problem
problem = elsa.WLSProblem(projector, sinogram)

# solve the problem
solver = elsa.CG(problem)
n_iterations = 20
reconstruction = solver.solve(n_iterations)

# plot the reconstruction
plt.imshow(np.array(reconstruction))
plt.show()
```

As you can see the code is essentially equivalent to the C++ code shown in the Quickstart guide :doc:`./quickstart-cxx`.
All the top level elsa C++ modules normally available through
```cpp
#include "elsa.h"
```
are also available under the top level `pyelsa` Python module.
All C++ functions and classes essentially have the same signatures in Python.
One important aspect of the Python bindings is, however, that in places where the C++ code would expect
`Eigen` vectors or matrices we can natively use `numpy` arrays as well as convert elsa `DataContainer` back to numpy
via

```python
data_container: elsa.DataContainer = ...
back_to_numpy = np.array(data_container)
```

This is important if we, for example, want to show a reconstruction image using matplotlib as it does not support the elsa
data containers directly.

### 3D reconstruction viewing
The tight integration with numpy and matplotlib also enables us to directly implement a 3D slice viewer
within our Python code.
Let us take a simple 3D phantom reconstruction example using a CUDA projector to be more performant:

```python
import pyelsa as elsa
import numpy as np
import matplotlib.pyplot as plt

size = np.array([64, 64, 64])  # 3D now
phantom = elsa.PhantomGenerator.createModifiedSheppLogan(size)
volume_descriptor = phantom.getDataDescriptor()

# generate circular trajectory
num_angles = 180
arc = 360

sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
    num_angles, phantom.getDataDescriptor(), arc, size[0] * 100, size[0])

# setup operator for 2d X-ray transform
projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor)

# simulate the sinogram
sinogram = projector.apply(phantom)

# setup reconstruction problem
problem = elsa.WLSProblem(projector, sinogram)

# solve the problem
solver = elsa.CG(problem)
n_iterations = 20
reconstruction = np.array(solver.solve(n_iterations))
```

We can now implement a simple matplotlib viewer plugin to scroll through our 3D reconstruction using the mouse wheel as shown in
the [matplotlib documentation](https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html)
```python
class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
```
We then simply setup our viewer
```python

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, reconstruction)
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()
```
and scroll through our 3D reconstruction.
