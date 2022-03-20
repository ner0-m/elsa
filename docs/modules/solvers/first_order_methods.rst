.. _elsa-first-order-methods-doc:
    
Background
##########

A short description of the pre-conditions for the algorithm will be discussed here.
For a more detailed discussion see the paper _Optimized first-order methods for smooth convex
minimization by Kim and Fessler (link at the bottom).

Intuition
*********
 
Let's first establish a visual intuition of momentum based gradient descent algorithms.
A nice analogy, is a ball in hilly terrain. The ball is at a random position, with zero
initial velocity. The algorithm determines the gradient of potential energy, which is the
force acting on the ball. Which in our case, is exactly the (negative) gradient of \f$f\f$.
Then the algorithm updates the velocity, which in turn updates the position of the ball.
Compared to a vanilla gradient descent, where the position is directly integrated instead of
the velocity.

Phrased differently, the velocity is a look ahead position, from where the gradient of the
current solution is applied to.
Nesterov's algorithm improves on that, by computing the gradient at the look ahead position,
instead of at the current solutions position.

Problem definition
******************

First-order algorithms solve problems of the form

.. math::

   \min_{x \in \mathbb{R}^d} f(x)

with two assumptions:

- :math:`f: \mathbb{R}^d \to \mathbb{R}` is a convex continuously differentiable function
  with Lipschitz continuous gradient, i.e. :math:`f \in C_{L}^{1, 1}(\mathbb{R}^d)` (with
  :math:`L > 0` is the Lipschitz constant)
- The problem is solvable, i.e. there exists an optimal :math:`x^{*}`
 
Solvers
*******

These solvers are currently implemented in elsa:

#. :ref:`Gradient Descent <elsa-solvers-api-gradientdescent>`
#. :ref:`Nesterov's Fast Gradient Method <elsa-solvers-api-fgm>`
#. :ref:`Optimized Gradient Method <elsa-solvers-api-ogm>`

