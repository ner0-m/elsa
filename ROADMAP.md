# elsa Project Roadmap

### How should I use this document?

This document serves to highlight certain areas of interest as well as the 
general direction of the *elsa* project. 
This includes the prioritization of certain features or designs. 
Together with the [Milestones](https://gitlab.lrz.de/IP/elsa/-/milestones) this 
should serve new contributors in finding a project in both their and our 
interest.

## Guiding Principles

First and foremost, *elsa* is a research project of the *Computational Imaging 
and Inverse Problems* (CIIP) group (head: Tobias Lasser) at the Technical 
University of Munich, Germany. 
Hence, one of the main objectives of *elsa* is to aid in the research of that 
group.
As such it should support active research areas, such as methods in X-ray 
dark-field tomography and light field microscopy.
However, the framework should be general enough to accommodate other fields of
interest.

Another objective is teaching. 
As part of the academic process, students should be able to do projects at our 
group using *elsa*. 
They should be able to contribute to and/or use the library. 
This implies a couple of things. 
First, *elsa* should be approachable from a technical standpoint. 
Second, it (or rather the community and processes around it) should be open and 
welcoming. 
Third, interoperability with other tools is highly important (this is also true
for other use cases).

As a last principle: the contributors should have fun doing what they do.
Staying motivated and engaged for long periods of time is challenging. 
And it is our belief that fun and challenging work can be a meaningful way to 
stay motivated and engaged. 
Of course, what might be dull and tedious for one, might be interesting to 
another. 
So, *elsa* should ideally provide a playground for fun experimentation for a 
diverse group. 
Even if that might somewhat contradict one of the other guiding principles.


## Long-term Goal (~2 years)

- Provide multi-GPU capable forward and backward projection models for multiple
imaging modalities. 
A minimal set of models would include attenuation X-ray computed tomography 
(CT), phase-contrast CT, dark-field CT and light field microscopy.
Ideally, they should be self-contained, in the sense that other libraries or
applications can utilize them easily.
- Provide an idiomatic Python package. This specifically includes the
interoperability with Python frameworks (i.e. array libraries such as NumPy or
CuPy, and deep learning frameworks such as TensorFlow and PyTorch). 
As a minimum, the Python package should expose the forward and backward 
projection models. 
A more comprehensive package would include typical reconstruction
methods (e.g. filtered-backprojection for X-ray attenuation CT), pre- and
post-processing methods and visualization.

There exists a further requirement for the above goals. 
They should be solved in such a way that the consumer of the Python package, 
cannot tell (and should not care), whether or not the feature is implemented in 
Python or C++.
This should enable us enough freedom to have a playground for interesting
ideas in C++. 
But on the other hand, have an ease of implementation in certain areas by 
relying on Python as well.

As this is an important point, one concrete example. 
The projection models for X-ray CT are computationally expensive and as such 
should be implemented in C++ with GPU support. 
But (at least a simple) projection model for light field microscopy is based on
the deconvolution of two signals. 
And as such might be easily implemented in Python without a significant cost in 
runtime performance.
Of course, this must be evaluated, but the basic idea should be clear.


## Midterm Goals (6-12 months)

Once the midterm goals are accomplished, a solid foundation of the long term
goals should be fulfilled. This should also mark the point in which *elsa* 
should be able to gain and seek a user base also outside of CIIP.

- [ ] Basic support for X-ray phase-contrast and dark-field CT on medical
devices, such that biomedical physicists can (and want) to use elsa. At a
minimum the projectors.
- [ ] Enable experimentation and different directions of research for light
field microscopy


## Short-term Goals (~1-3 months)

Not all goals here need to be addressed in the next 1 to 3 months. 
But they should be solvable withing such a time frame and should steer towards 
both the mid- and long term goals.

Functional important features are:
- [ ] Support curved detectors
- [ ] Support projector for differentiated Blobs and B-Splines
- [ ] Support forward and backward model of light field microscopy (with a
given point spread function)
- [ ] Support generation of some point spread functions
- [ ] Use elsa's projectors with any tensor providing an array protocol
like interface (e.g. [DLPack](https://dmlc.github.io/alpaca/latest/))

Necessary refactors:
- [ ] Projectors exposed to Python should model the SciPy linear operator
- [ ] Reduce required boilerplate and cognitive load to implement projectors

Possible simplifications:
- [ ] Removal of automatically generated Python bindings
- [ ] Removal of GPU expression templates (to simplify code, might be 
re-added later)

