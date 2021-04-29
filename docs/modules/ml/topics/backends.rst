.. _elsa-ml-backends:

******************
Choosing a Backend
******************

elsa's ml module uses highly optimized Deep Learning libaries under the hood. 
Currently two backends are available:

#. `oneDNN <https://github.com/oneapi-src/oneDNN>`_, which is highly optimized for CPU workloads.
#. `cuDNN <https://developer.nvidia.com/cudnn>`_ which is highly optimized for workloads on Nvidia GPUs.

During the construction of a Model a backend can be chosen by specifying a
template parameter:

.. code-block:: cpp

  // Construct a model using CuDNN
  auto model = ml::Model<real_t, MlBackend::Cudnn>(...);

  // Construct a model using oneDNN
  auto model = ml::Model<real_t, MlBackend::Dnnl>(...);

The default template parameter of the Model class is ``MlBackend::Dnnl``.
