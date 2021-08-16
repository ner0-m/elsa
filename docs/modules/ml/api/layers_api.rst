Layers
======

Core layers
-----------

.. doxygenclass:: elsa::ml::Input
  :members:

.. doxygenclass:: elsa::ml::Dense
  :members:

Activation layers
-----------------

Use these layers if you want to specify an activation function decoupled from
a dense or convolutional layer. Otherwise see the activation parameter of these
layers.

.. doxygenstruct:: elsa::ml::Sigmoid
  :members:

.. doxygenstruct:: elsa::ml::Relu
  :members:

.. doxygenstruct:: elsa::ml::Tanh
  :members:

.. doxygenstruct:: elsa::ml::ClippedRelu
  :members:

.. doxygenstruct:: elsa::ml::Elu
  :members:

.. doxygenstruct:: elsa::ml::Identity
  :members:

.. doxygenclass:: elsa::ml::Softmax
  :members:

.. doxygenenum:: elsa::ml::Activation

Initializer
-----------

.. doxygenenum:: elsa::ml::Initializer

Convolutional layers
--------------------

.. doxygenenum:: elsa::ml::Padding

.. doxygenstruct:: elsa::ml::Conv1D
  :members:

.. doxygenstruct:: elsa::ml::Conv2D
  :members:

.. doxygenstruct:: elsa::ml::Conv3D
  :members:

.. doxygenstruct:: elsa::ml::Conv2DTranspose
  :members:

Merging layers
--------------

.. doxygenclass:: elsa::ml::Sum
  :members:

.. doxygenclass:: elsa::ml::Concatenate
  :members:

Reshaping layers
----------------

.. doxygenclass:: elsa::ml::Reshape
  :members:

.. doxygenclass:: elsa::ml::Flatten
  :members:

.. doxygenenum:: elsa::ml::Interpolation

.. doxygenstruct:: elsa::ml::UpSampling1D
  :members:

.. doxygenstruct:: elsa::ml::UpSampling2D
  :members:

.. doxygenstruct:: elsa::ml::UpSampling3D
  :members:
