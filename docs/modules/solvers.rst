************
elsa solvers
************

.. contents:: Table of Contents

Solver Interface
================

.. doxygenclass:: elsa::Solver

Iterative
=========

Iterative solvers to solve :math:`A x = b`.

CGLS
####

.. doxygenclass:: elsa::CGLS

CGNE
####

.. doxygenclass:: elsa::CGLS

Landweber iteration
###################

.. doxygenclass:: elsa::LandweberIteration

Landweber
+++++++++

.. doxygenclass:: elsa::Landweber

SIRT
++++

.. doxygenclass:: elsa::SIRT

GMRES
#####

AB_GMRES
++++++++

.. doxygenclass:: elsa::AB_GMRES

BA_GMRES
++++++++

.. doxygenclass:: elsa::BA_GMRES

Smooth
======

Optimization algorithms for smooth problem formulations.

First-order optimization algorithms
###################################

First-order algorithms solve problems of the form

.. math::

   \min_{x \in \mathbb{R}^d} f(x)

with two assumptions:

- :math:`f: \mathbb{R}^d \to \mathbb{R}` is a convex continuously differentiable function
  with Lipschitz continuous gradient, i.e. :math:`f \in C_{L}^{1, 1}(\mathbb{R}^d)` (with
  :math:`L > 0` is the Lipschitz constant)
- The problem is solvable, i.e. there exists an optimal :math:`x^{*}`

Intuition for Momentum
++++++++++++++++++++++

A nice analogy, is a ball in hilly terrain. The ball is at a random position,
with zero initial velocity. The algorithm determines the gradient of potential
energy, which is the force acting on the ball. Which in our case, is exactly
the (negative) gradient of \f$f\f$. Then the algorithm updates the velocity,
which in turn updates the position of the ball. Compared to a vanilla gradient
descent, where the position is directly integrated instead of the velocity.

Phrased differently, the velocity is a look ahead position, from where the
gradient of the current solution is applied to. Nesterov's algorithm improves
on that, by computing the gradient at the look ahead position, instead of at
the current solutions position.

GradientDescent
+++++++++++++++

.. doxygenclass:: elsa::GradientDescent

Nesterov's Fast Gradient Method
+++++++++++++++++++++++++++++++

.. doxygenclass:: elsa::FGM

Optimized Gradient Method
+++++++++++++++++++++++++

.. doxygenclass:: elsa::OGM

SQS Ordered Subsets
+++++++++++++++++++

.. doxygenclass:: elsa::SQS

Non-Smooth
==========

Optimization algorithms for non-smooth problem formulations. These problem
formulations usually contain at least one non-differentiable term, such as the
L1-Norm.

Proximal Gradient Methods
#########################

Proximal gradient methods solves problems of the form

.. math::

   \min_{x \in \mathbb{R}^d} g(x) + h(x)

where :math:`g` is a smooth function, and :math:`h` is *simple* (meaning the
proximal operator is easy to compute).


Proximal Gradient Descent
+++++++++++++++++++++++++

.. doxygenclass:: elsa::PGD

Accelerated Proximal Gradient Descent
+++++++++++++++++++++++++++++++++++++

.. doxygenclass:: elsa::APGD

Alternating Direction Method of Multipliers
###########################################

The Alternating Direction Method of Multipliers (ADMM) solves problems of the form:

.. math::

     \min f(x) + g(z) \\
     \text{s.t. } Ax + Bz = c

With :math:`x \in \mathbb{R}^{n}`, :math:`z \in \mathbb{R}^{m}`, :math:`c \in
\mathbb{R}^{p}`, :math:`A \in \mathbb{R}^{p \times n}` and :math:`B \in
\mathbb{R}^{p \times m}`. Usually, one assumes :math:`f` and :math:`g` to be
convex at least.

This problem in general is quite hard to solve for many interesting applications
such as X-ray CT. However, certain special cases are quite interesting and
documented below.

ADMM with L2 data fidelity term
+++++++++++++++++++++++++++++++

.. doxygenclass:: elsa::ADMML2

Linearized ADMM
+++++++++++++++

.. doxygenclass:: elsa::LinearizedADMM

Orthogonal Matching Pursuit
###########################

.. doxygenclass:: elsa::OrthogonalMatchingPursuit
