.. _elsa-ml-complex-graphs:

***********************************
Constructing complex graph tologies
***********************************

In our :ref:`first example <elsa-ml-first-example>` we constructed a model consisting
of a sequential stack of layers. We now want to learn how complex network 
topologies can be maintained using elsa's ml module.

As an example we will try to model the following network:

.. image:: complex_model.png
  :width: 300
  :align: center
  :alt: ComplexGraph

It's clear that the above model is not sequential anymore and we thus have to 
model the network explicitly.

We start by constructing an input layer followed by a convolutional one:

.. code-block:: cpp

  VolumeDescriptor inputDesc({28, 28, 1});
  auto input = ml::Input(inputDesc, /* batch size */ 128);

  auto conv = ml::Conv2D(
    /* number of kernels */ 32, 
    /* shape of each kernel */ {3, 3, 1},
    /* activation */ ml::Activation::Relu,
    /* strides */ 1,
    /* input padding */ ml::Padding::Valid,
    /* use bias? */ true);
  conv.setInput(&input);

We continue by adding the stack ``MaxPooling2D -> Conv2D -> UpSamling2D``:

.. code-block:: cpp

  // Pooling layer
  auto pooling = ml::MaxPooling2D();
  pooling.setInput(&conv);

  // Second conv layer
  auto conv2 = ml::Conv2D(32, {3, 3, 32}, ml::Activation::Relu, 1, ml::Padding::Same, true);
  conv2.setInput(&pooling);

  // Upsamling layer
  auto upsample = ml::UpSampling2D({2, 2});
  upsample.setInput(&conv2);
       
Now comes the interesting part: At the node label as ``Sum`` both, the previously
defined ``upsample`` and the first convolutional layer ``conv`` merge their 
paths in the graph. Thus we call such a node a `Merging layer`. 

The one used here just sums up its inputs and produces a single output. Declaring
it is straight forward:

.. code-block:: cpp

  auto sum = ml::Sum({
    /* first input */ &upsample,
    /* second input */ &conv
    /* can be more ... */
  });

After this point we continue as usual by defining the other layers and setting
respective inputs:

.. code-block:: cpp

    // third conv layer
    auto conv3 = ml::Conv2D(32, {5, 5, 32}, ml::Activation::Relu, 1, ml::Padding::Valid, true);

    // sum is input
    conv3.setInput(&sum);

    // second pooling layer
    auto pooling2 = ml::MaxPooling2D();
    pooling2.setInput(&conv3);

    // flatten layer
    auto flatten = ml::Flatten();
    flatten.setInput(&pooling2);

    // dense layer
    auto dense = ml::Dense(128, ml::Activation::Relu);
    dense.setInput(&flatten);

    // Dense/Softmax output
    auto dense2 = ml::Dense(10, ml::Activation::Identity);
    dense2.setInput(&dense);
    auto softmax = ml::Softmax();
    softmax.setInput(&dense2);

As usual we can now construct and pretty-print our model:

.. code-block:: cpp

  auto model = ml::Model(&input, &softmax);
  std::cout << model << "\n";

This produces the output

.. code-block:: none

  Model: model
  ________________________________________________________________________________
  Layer (type)                       Output Shape        Param #   Connected to
  ================================================================================
  input_0 (Input)                    (28,  28,   1)      0         conv2d_1
  ________________________________________________________________________________
  conv2d_1 (Conv2D)                  (26,  26,  32)      320       maxpooling2d_3
                                                                   sum_7
  ________________________________________________________________________________
  maxpooling2d_3 (MaxPooling2D)      (13,  13,  32)      0         conv2d_4
  ________________________________________________________________________________
  conv2d_4 (Conv2D)                  (13,  13,  32)      9248      upsampling2d_6
  ________________________________________________________________________________
  upsampling2d_6 (UpSampling2D)      (26,  26,  32)      0         sum_7
  ________________________________________________________________________________
  sum_7 (Sum)                        (26,  26,  32)      0         conv2d_8
  ________________________________________________________________________________
  conv2d_8 (Conv2D)                  (22,  22,  32)      25632     maxpooling2d_10
  ________________________________________________________________________________
  maxpooling2d_10 (MaxPooling2D)     (11,  11,  32)      0         flatten_11
  ________________________________________________________________________________
  flatten_11 (Flatten)               (3872)              0         dense_12
  ________________________________________________________________________________
  dense_12 (Dense)                   (128)               495744    dense_14
  ________________________________________________________________________________
  dense_14 (Dense)                   (10)                1290      softmax_16
  ________________________________________________________________________________
  softmax_16 (Softmax)               (10)                0
  ================================================================================
  Total trainable params: 532234
  ________________________________________________________________________________

