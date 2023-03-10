****************
elsa functionals
****************

.. contents:: Table of Contents


LinearResidual
==============

.. doxygenclass:: elsa::LinearResidual

Functional
==========

.. doxygenclass:: elsa::Functional

Composition of Functionals
==========================

.. doxygenclass:: elsa::FunctionalSum

.. doxygenclass:: elsa::FunctionalScalarMul


Loss functionals
================

Loss functionals are often used as data fidelity terms. They are specific versions
of certain functionals, but are important enough to receive special attention.

LeastSquares
------------

.. doxygenclass:: elsa::LeastSquares

WeightedLeastSquares
--------------------

.. doxygenclass:: elsa::WeightedLeastSquares

EmissionLogLikelihood
---------------------

.. doxygenclass:: elsa::EmissionLogLikelihood

TransmissionLogLikelihood
-------------------------

.. doxygenclass:: elsa::TransmissionLogLikelihood


Norms
=====

Norms are another subclass of functionals. A norm is a functional with the
additional properties, where :math:`f : X \to \mathbb{R}` some functional,
:math:`x \text{ and } y \in X` elements of the domain, and :math:`s \in
\mathbb{R}` a scalar:
* the triangle inequality holds, i.e. :math:`f(x + y) \leq f(x) + f(y)`.
* :math:`f(sx) = |s| f(x)` for all :math:`x \in X`.
* :math:`f(x) = 0` if and only if :math:`x = 0`.

From these it also holds, that the result of a norm is always non-negative.

L1Norm
======

.. doxygenclass:: elsa::L1Norm

WeightedL1Norm
==============

.. doxygenclass:: elsa::WeightedL1Norm

L2Squared
=========

.. doxygenclass:: elsa::L2Squared

L2Reg
=====

.. doxygenclass:: elsa::L2Reg

LInfNorm
========

.. doxygenclass:: elsa::LInfNorm

L0PseudoNorm
------------

.. doxygenclass:: elsa::L0PseudoNorm


Other functionals
=================

Huber
-----

.. doxygenclass:: elsa::Huber


PseudoHuber
-----------

.. doxygenclass:: elsa::PseudoHuber

Quadric
-------

.. doxygenclass:: elsa::Quadric

ConstantFunctional
------------------

.. doxygenclass:: elsa::ConstantFunctional

.. doxygenclass:: elsa::ZeroFunctional

Indicator Functionals
---------------------

.. doxygenclass:: elsa::IndicatorBox

.. doxygenclass:: elsa::IndicatorNonNegativity

