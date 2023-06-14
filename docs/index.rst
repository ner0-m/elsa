**********************************************************
elsa - an elegant framework for tomographic reconstruction
**********************************************************

**elsa** is an operator- and optimization-oriented framework for tomographic
reconstruction, with a focus on iterative reconstruction algorithms.
It is usable from Python and C++.

By design, **elsa** provides a flexible description of multiple imaging modalities.
The current focus is X-ray based computed tomography (CT) modalities such as
attenuation X-ray CT, phase-contrast X-ray CT based on grating interferometry
and (anisotropic) Dark-field X-ray CT. Other imaging modalities can be
supported easily and can leverage our extensive suite of optimization algorithms.

CUDA implementations for the computationally expensive forward models, which
simulate the physical measurement process of the imaging modality, are available
in **elsa**.

The framework is mostly developed by the Computational Imaging and Inverse Problems
(CIIP) group at the Technical University of Munich. For more info about our research
checkout our at https://ciip.cit.tum.de/.

The source code of **elsa** is hosted at
https://gitlab.lrz.de/IP/elsa. It is available under the
Apache 2 open source license.

Check the readme at the source repository for current build and installation
instructions. A good point to get started are our guides, which can be found
:doc:`here <guides/index>`. Another good starting point, is the example folder
of the repository. There many different scenarios are covered and well documented.
The C++ API reference is found :doc:`here <modules/index>`.
