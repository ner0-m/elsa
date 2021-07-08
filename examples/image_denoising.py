#!/usr/bin/env python3
from datetime import datetime
import os
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
    last_patch_x = math.floor(imageshape[0] - blocksize)
    last_patch_y = math.floor(imageshape[1] - blocksize)

    x = 0
    y = 0
    for i in range(patches.shape[0]):
        patch = patches[i,:].reshape(blocksize, blocksize, order='F')
        image[x:x+blocksize, y:y+blocksize] += patch

        if(y < last_patch_y):
            y += stride
        else:
            y = 0
            x += stride

    image /= (blocksize*blocksize/stride)

    return image

def plot_and_save(original, noisy, reconstruction):
    filename = "imagedenoising"
    if not os.path.exists(filename):
        os.makedirs(filename)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    filename += "/" + timestamp + ".png"

    plt.subplot(131)
    plt.imshow(raccoon, cmap='gray')
    plt.subplot(132)
    plt.imshow(noisy_raccoon, cmap='gray')
    plt.subplot(133)
    plt.imshow(reconstructed_raccoon, cmap='gray')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

#############MAIN######################

raccoon = scipy.misc.face(gray=True)
raccoon = raccoon / 255.0 #scale between 0 and 1
#downsample size by 4, magic oneliner I found on stackoverflow
raccoon = raccoon[::4, ::4] + raccoon[1::4, ::4] + raccoon[::4, 1::4] + raccoon[1::4, 1::4]
raccoon /= 4.0

noise = np.random.normal(0,0.1,raccoon.shape)
noisy_raccoon = raccoon + noise 

patched_raccoon = im2patches(noisy_raccoon, 12, 4)
n_patches = patched_raccoon.shape[0]
patchlength = patched_raccoon.shape[1]
patch_descriptor = elsa.pyelsa_core.VolumeDescriptor([patchlength])
raccoon_descriptor = elsa.pyelsa_core.IdenticalBlocksDescriptor(n_patches, patch_descriptor)
raccoon_dc = elsa.pyelsa_core.DataContainer(raccoon_descriptor, patched_raccoon.flatten())

problem = elsa.pyelsa_problems.DictionaryLearningProblem(raccoon_dc, 128)
solver = elsa.pyelsa_solvers.KSVD(problem, 16)
representations = solver.solve(50)
dictionary = solver.getLearnedDictionary()

reconstructed_raccoon_patches = np.zeros(patched_raccoon.shape)
for i in range(n_patches):
    reconstructed_raccoon_patches[i] = np.array(dictionary.apply(representations.getBlock(i)))

reconstructed_raccoon = patches2im(reconstructed_raccoon_patches, raccoon.shape, 12, 4)

plot_and_save(raccoon, noisy_raccoon, reconstructed_raccoon)
