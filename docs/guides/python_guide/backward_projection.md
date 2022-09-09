Backward projection
-------------------

Now, you have this sinogram, but so far we only dealt with creating some measurements. If you'd have
experimental data from a X-ray scanner, you'd load the sinogram and start from here. So now we want
to reconstruction the the original object of interest. There are many different ways to solve this.

The first and simplest, is to perform the adjoint operations of the forward projection. Simply from
the previous example run the following code:

```python
backprojection = projector.applyAdjoint(sinogram)
plt.imshow(np.array(backprojection))
plt.show()
```

Uh, that kind of looks like the original phantom, but that is quite blurry. What can we do about
that? From an engineering point of view, this image needs deblurring, which can be achieved through
many different ways, such as filtering. However, let us step back again and look at the bigger
picture.

### Towards a successful reconstruction

In this guide, we have not talked about math a lot. However, tomographic reconstruction boils down
to solving an inverse problem. Broadly speaking, there are two very different approaches to solve an
inverse problem:
1. Analytical reconstruction
2. Iterative reconstruction

In the next chapter, we will show the analytical reconstruction, the so-called _filtered
backprojection_. It is a very popular method in the field of X-ray tomography.

However, for all interesting problems, the inverse problem is ill-conditioned or ill-posed. A
problem is well-posed if all of the following conditions are met:
1. A solution exits
2. The solution is unique
3. The solution is stable
If at least one is not met, the problem is ill-posed. A solution is considered stable, if small
changes to the measurements, only induces a small change in the solution. Mathemathically speaking,
the solution depends continuously on the measurements. Hence, an for an unstable problem, a small
change in the measurements may lead to an arbitrary jump in the solution.

Iterative algorithms have been developed and investigated to deal with ill-posed problems.
Importantly, they are able to handle such situations better then the analytical reconstruction.
Usually at the cost of runtime efficiency. Still, iterative algorithms are tremendously important
and interesting, hence we will devote a couple of chapters to different algorithms.
