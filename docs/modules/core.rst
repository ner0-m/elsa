*********
elsa core
*********

.. contents:: Table of Contents

Core Type Declarations
======================

.. doxygenfile:: elsa.h

DataDescriptor
==============

.. doxygenclass:: elsa::DataDescriptor

BlockDescriptor
===============

.. doxygenclass:: elsa::BlockDescriptor

IdenticalBlocksDescriptor
=========================

.. doxygenclass:: elsa::IdenticalBlocksDescriptor

PartitionDescriptor
===================

.. doxygenclass:: elsa::PartitionDescriptor

RandomBlocksDescriptor
======================

.. doxygenclass:: elsa::RandomBlocksDescriptor

DataContainer
=============

.. doxygenclass:: elsa::DataContainer
 
DetectorDescriptor
=============

.. doxygenclass:: elsa::DetectorDescriptor

PlanarDetectorDescriptor
=============

.. doxygenclass:: elsa::PlanarDetectorDescriptor

CurvedDetectorDescriptor
=============

.. doxygenclass:: elsa::CurvedDetectorDescriptor
 
Geometry
=============

.. doxygenclass:: elsa::Geometry
 

LinearOperator
==============

.. doxygenclass:: elsa::LinearOperator


Implementation Details
======================

Cloneable
---------

.. doxygenclass:: elsa::Cloneable

DataHandler
-----------

.. doxygenclass:: elsa::DataHandler

DataHandlerCPU
--------------

.. doxygenclass:: elsa::DataHandlerCPU

DataHandlerGPU
--------------

.. mdinclude:: data_handler_GPU.md
.. doxygenclass:: elsa::DataHandlerGPU

Expression
----------
.. mdinclude:: expression_templates.md

.. doxygenclass:: elsa::Expression
