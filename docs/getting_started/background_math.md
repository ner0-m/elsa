Tomographic reconstruction is a type of inverse problem. A typical notation for this is:

>  :math:`\mathbf A(x) = b`

where :math:`b` is the observed or measured data, :math:`\mathbf A` is an operator and :math:`x` is the solution we seek to find.

The measured or observed data could be a noisy and/or blurred image, or the measured sinogram in the case of X-ray CT. :math:`x` is the original, not transformed data. In the case of image reconstruction, it would be the non noisy or non blurred image. In the case of a X-ray CT, it would the a the scanned object.

From a mathematical standpoint, those two problems do not differ in their high level description, only in their operators. For image deconvolution a different operator :math:`\mathbf A` is necessary compared to X-ray CT.

Now that we got the formulation, we have many possibilities to solve this equation. For X-ray CT, the classical approach is the filtered backprojection (FBP). The foundation of it is the [Radon transform](https://en.wikipedia.org/wiki/Radon_transform). 

For more complex and advanced problems, usually a little different formulation is needed. We'll usually use least squares of the form:

>  :math:`\underset{x}{\operatorname{argmin}} \frac{1}{2} \| \mathbf A x - b \|_2^2`

This is a convenient way of writing our problem. From this point, we can use many different solving algorithms and regularization techniques (e.g. [Tikonov](https://en.wikipedia.org/wiki/Tikhonov_regularization))to get better results. In the :ref:`example above <2D Example>` we used exactly this problem formulation. In elsa, we call these ways of formulation :ref:`problems <elsaproblems>`

Our framework currently supports [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) (GD) and [Conjugate gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method) (CG) to solve the above equation. GD in the current form is most likely more known in the context of machine learning and not typically used for X-ray CT. CG on the other hand has a couple of constraints. It needs a very specific problem formulation, elsa will convert a problem to the desired form if possible. This corresponds to our :ref:`solvers <elsasolvers>`