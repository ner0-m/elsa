************
elsa storage
************

.. contents:: Table of Contents


The storage module of `elsa` is an implementation detail of the `DataContainer` and functionals.
It contains data types for owning and non-owning storage plus a set of algorithms. The storage
module is used as generic implementation module and should be used with some caution. The
implementation is based on the library `thrust` and hence might depend on CUDA internals. Exposing `thrust` to
headers thus might introduce a dependency on the Nvidia compiler. In certain cases, this is not what we
want, hence please be careful when using functionality from this module.

Algorithms
==========

The most important algorithms are reductions and (unary and binary) transformations.

Reductions
----------

.. doxygengroup:: reductions
   :project: elsa

Transforms
----------

.. doxygengroup:: transforms
   :project: elsa
