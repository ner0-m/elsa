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
    coeffs = np.zeros(imageshape) # keep track of how many patches have been used to create a pixel
    last_patch_x = math.floor(imageshape[0] - blocksize)
    last_patch_y = math.floor(imageshape[1] - blocksize)

    x = 0
    y = 0
    for i in range(patches.shape[0]):
        patch = patches[i,:].reshape(blocksize, blocksize, order='F')
        image[x:x+blocksize, y:y+blocksize] += patch
        coeffs[x:x+blocksize, y:y+blocksize] += np.ones(patch.shape)

        if(y + stride <= last_patch_y):
            y += stride
        else:
            y = 0
            x += stride

    coeffs[coeffs == 0] = 1 # avoid dividing by zero for pixels that don't appear in any patch
    image /= coeffs

    return image

def save_results(original, noisy, reconstruction):
    filename = "imagedenoising"
    if not os.path.exists(filename):
        os.makedirs(filename)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    filename += "/" + timestamp

    with open(filename + "_params.txt", 'w') as f:
        f.write(f"Gaussian noise sigma: {sigma}\n")
        f.write(f"Blocksize {blocksize}, Stride {stride}\n")
        f.write(f"Number of atoms {nAtoms}, Sparsity level {sparsityLevel}\n")
        f.write(f"KSVD iterations: {nIterations}\n")

    plt.imsave(filename + "_original.png", original, cmap='gray')
    plt.imsave(filename + "_noisy.png", noisy, cmap='gray')
    plt.imsave(filename + "_reconstruction.png", reconstruction, cmap='gray')
    plt.show()


############PARAMS#####################
sigma = 0.05
blocksize = 5
stride = 4
nAtoms = 128
sparsityLevel = 6
nIterations = 15
#############MAIN######################
elsa.Logger.setLevel(elsa.LogLevel.OFF)

print("Image Denoising: Preprocessing image...")
raccoon = scipy.misc.face(gray=True)
raccoon = raccoon / 255.0 #scale between 0 and 1
#downsample size by 2, magic oneliner I found on stackoverflow
raccoon = raccoon[::2, ::2] + raccoon[1::2, ::2] + raccoon[::2, 1::2] + raccoon[1::2, 1::2]
raccoon /= 2.0

noise = np.random.normal(0,sigma,raccoon.shape)
noisy_raccoon = raccoon + noise 

image_descriptor = elsa.pyelsa_core.VolumeDescriptor(noisy_raccoon.shape);
image_dc = elsa.pyelsa_core.DataContainer(image_descriptor, noisy_raccoon.flatten())

denoising_task = elsa.pyelsa_tasks.ImageDenoisingTask(blocksize, stride, sparsityLevel, nAtoms, nIterations)

reconstructed_raccoon = denoising_task.train(image_dc)
reconstructed_raccoon = np.array(reconstructed_raccoon)
print("Non-zeros:")
print(np.count_nonzero(reconstructed_raccoon))
print("Reconstruciton shape:")
print(reconstructed_raccoon.shape)

save_results(raccoon, noisy_raccoon, reconstructed_raccoon)
# OLD STUFF:

#patched_raccoon = im2patches(noisy_raccoon, blocksize, stride)
#n_patches = patched_raccoon.shape[0]
#patchlength = patched_raccoon.shape[1]
#patch_descriptor = elsa.pyelsa_core.VolumeDescriptor([patchlength])
#raccoon_descriptor = elsa.pyelsa_core.IdenticalBlocksDescriptor(n_patches, patch_descriptor)
#raccoon_dc = elsa.pyelsa_core.DataContainer(raccoon_descriptor, patched_raccoon.flatten())

#print("Image Denoising: Learning dictionary and representations...")
#problem = elsa.pyelsa_problems.DictionaryLearningProblem(raccoon_dc, nAtoms)
#solver = elsa.pyelsa_solvers.KSVD(problem, sparsityLevel)
#representations = solver.solve(nIterations)
#dictionary = solver.getLearnedDictionary()

#reconstructed_raccoon_patches = np.zeros(patched_raccoon.shape)
#for i in range(n_patches):
    #reconstructed_raccoon_patches[i] = np.array(dictionary.apply(representations.getBlock(i)))

#print("Image Denoising: Reconstructing image from resulting patches")
#reconstructed_raccoon = patches2im(reconstructed_raccoon_patches, raccoon.shape, blocksize, stride)

#save_results(raccoon, noisy_raccoon, reconstructed_raccoon)
