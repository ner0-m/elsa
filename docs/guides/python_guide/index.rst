*********************
Python Bindings Guide
*********************

.. toctree::
  install.md
  forward_projection.md
  backward_projection.md
  filtered_backprojection.md

*elsa* also comes with Python bindings for almost all aspects of the framework.
This short guide aims to give an introduction into some simple use cases and explains how one can
easily translate C++ code into Python code for faster prototyping and experimenting.
One major benefit that comes with the Python bindings is that we are able to natively
use numpy arrays with our elsa data containers, making it easy to work with other tools such as
matplotlib.

This small tutorial will cover mostly topics related to X-ray tomography. Though some of the later
topics do not depend on the underlying applications, the first couple of steps are applications
specific, and hence will deal with X-ray attenuation computed tomography (CT).

Throughout the guide the following imports are assumed to be present:

.. code-block:: python

   import pyelsa as elsa
   import numpy as np
   import matplotlib.pyplot as plt
