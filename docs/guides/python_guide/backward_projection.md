Backward projection
-------------------

From the previous section, we created an artifical sinogram through simulations. If you would have
experimental data from an X-ray scanner, you would load that sinogram and start from there. Now we 
want a reconstruction of the original object of interest. There are many different ways to do this.

The first and most simple one is to perform the adjoint operation of the forward projection. From
the previous example, simply run the following code:

```python
backprojection = projector.applyAdjoint(sinogram)
plt.imshow(np.array(backprojection))
plt.show()
```

Uh, that kind of looks like the original phantom, but it is quite blurry. What can we do about
that? From an engineering point of view, this image needs deblurring, which can be achieved in
many different ways, such as filtering. However, let us step back again and look at the bigger
picture.

### Towards a successful reconstruction

In this guide, we have not talked about math a lot. However, tomographic reconstruction boils down
to solving an inverse problem. Broadly speaking, there are two very different approaches to solve an
inverse problem:
1. Analytical reconstruction
2. Iterative reconstruction

Both approaches mainly differ in the timepoint of discretization - in analytical reconstruction, you
discretize at the end, for iterative reconstruction you discretize straight away.

In the next section, we will show an analytical reconstruction method, the so-called _filtered
backprojection_. It is a very popular method in the field of X-ray computed tomography.

However, for all interesting problems, the inverse problem is ill-conditioned or ill-posed. A
problem is well-posed if all of the following conditions are met:
1. A solution exits
2. The solution is unique
3. The solution is stable

If at least one is not met, the problem is ill-posed. A solution is considered stable, if small
changes to the measurements only induce small changes in the solution. Mathematically speaking,
the solution depends continuously on the measurements. Hence, for an unstable problem, a small
change in the measurements may lead to an arbitrarily large jump in the solution.

Iterative algorithms have been developed and investigated to deal with ill-posed problems.
Importantly, they are typically able to handle such situations better than analytical 
reconstruction methods, usually at the cost of runtime efficiency. Still, iterative algorithms 
are very important and interesting, hence we will devote a couple of sections to different algorithms.


_(to be continued)_