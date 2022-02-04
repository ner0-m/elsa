.. _elsa-solvers-api:

****************
elsa solvers API
****************

.. contents:: Table of Contents

Solver
======

.. doxygenclass:: elsa::Solver

CG
==

.. doxygenclass:: elsa::CG

Iterative Shrinkage-Thresholding Algorithm
==========================================

.. doxygenclass:: elsa::ISTA

Fast Iterative Shrinkage-Thresholding Algorithm
===============================================

.. doxygenclass:: elsa::FISTA


Alternating Direction Method of Multipliers
===========================================

.. doxygenclass:: elsa::ADMM

SQS Ordered Subsets
===================

.. doxygenclass:: elsa::SQS

Orthogonal Matching Pursuit
===========================

.. doxygenclass:: elsa::OrthogonalMatchingPursuit
 

.. _elsa-solvers-api-first-order-methods:
    
First-order optimization algorithms
===================================

.. include:: first_order_methods.rst

.. _elsa-solvers-api-gradientdescent:
    
GradientDescent
###############
    
.. doxygenclass:: elsa::GradientDescent

.. _elsa-solvers-api-fgm:

Nesterov's Fast Gradient Method
###############################

.. doxygenclass:: elsa::FGM

.. _elsa-solvers-api-ogm:
    
Optimized Gradient Method
#########################

.. doxygenclass:: elsa::OGM
