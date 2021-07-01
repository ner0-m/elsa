#!/usr/bin/env python3
import sys
sys.path.append("/home/buerger/Documents/master_thesis/elsa/build")

import pyelsa as elsa
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.misc

def im2patches(image, blocksize, stride):
    print("Image shape:", image.shape)
    nrows = image.shape[0]
    ncols = image.shape[1]

    npatches_x = math.floor(1 + ((nrows - blocksize) / stride))
    npatches_y = math.floor(1 + ((ncols - blocksize) / stride))
    patches = np.zeros([npatches_x*npatches_y, blocksize*blocksize])
    print("Patches shape:", patches.shape)

    patch_idx = 0

    for x_base_idx in range(0, npatches_x): 
        for y_base_idx in range(0, npatches_y): 
            for block_x in range(blocksize):
                for block_y in range(blocksize):
                    patches[patch_idx][block_y*blocksize+block_x] = image[stride*x_base_idx+block_x][stride*y_base_idx+block_y]
                    
            patch_idx += 1
    return patches 

def patches2im(patches, imageshape, blocksize, stride):
    image = np.zeros(imageshape)
    last_patch_x = math.floor((imageshape[0] - blocksize) / stride)
    last_patch_y = math.floor((imageshape[1] - blocksize) / stride)

    x = 0
    y = 0
    patchnr = 0
    for i in range(patches.shape[0]):
        patch = patches[i,:].reshape(blocksize, blocksize, order='F')
        try:
            image[x:x+blocksize, y:y+blocksize] += patch
            #print(f"Writing patch {patchnr} to block {x},{y} - {x+blocksize},{y+blocksize}")
        except:
            print(x,y)
            sys.exit(1)
        if(y < last_patch_y):
            y += 1
        else:
            y = 0
            x += 1
        patchnr +=1

    image /= blocksize*blocksize

    return image

raccoon = scipy.misc.face(gray=True)
raccoon = raccoon / 255.0 #scale between 0 and 1
#downsample size by 4, magic oneliner I found on stackoverflow
raccoon = raccoon[::4, ::4] + raccoon[1::4, ::4] + raccoon[::4, 1::4] + raccoon[1::4, 1::4]
raccoon /= 4.0

noise = np.random.normal(0,0.1,raccoon.shape)
noisy_raccoon = raccoon + noise 

patched_raccoon = patches2im(im2patches(raccoon, 10, 1), raccoon.shape, 10, 1)

plt.subplot(131)
plt.imshow(raccoon, cmap='gray')
plt.subplot(132)
plt.imshow(noisy_raccoon, cmap='gray')
plt.subplot(133)
plt.imshow(patched_raccoon, cmap='gray')
plt.show()

def old_stuff():
    size = np.array([128, 128])
    phantom = elsa.PhantomGenerator.createModifiedSheppLogan(size)
    volume_descriptor = phantom.getDataDescriptor()
    
    # generate circular trajectory
    num_angles = 180
    arc = 360

    sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        num_angles, phantom.getDataDescriptor(), arc, size[0] * 100, size[0])
        
    # setup operator for 2d X-ray transform
    projector = elsa.SiddonsMethod(volume_descriptor, sino_descriptor)

    # simulate the sinogram
    sinogram = projector.apply(phantom)

    # setup reconstruction problem
    problem = elsa.WLSProblem(projector, sinogram)

    # solve the problem
    solver = elsa.CG(problem)
    n_iterations = 20
    reconstruction = solver.solve(n_iterations)

    # plot the reconstruction
    plt.imshow(np.array(reconstruction), '2D Reconstruction')
    plt.show()

