# Architecture
- Describe here the architecture of the model to be used
- The DL model should be easily swapped in/out (without modifications in the shearlets)?
- Transformer models have recently started to outperform Convolutional models, consider
using one here e.g. TransUNet (within time limitations)

# Training
- TODO Do we need to simulate the effect of limited angle tomography onto the images so they
can be used as training data?
- TODO Do I need to save a good performing model in the upcoming elsa library? This model might 
  then be retrieved similarly to the lines of `myModel = models.myModel(preTrained=True)`
- Initially overfit the model to a few samples, can expose some common pitfalls 
  
# Data
- What should be the size of training/test data, (what about validation)?
- Consider data augmentation



This package should not even make it to an MR.
