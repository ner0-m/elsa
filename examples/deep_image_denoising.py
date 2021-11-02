#!/usr/bin/env python3
from datetime import datetime
import os
import sys

import pyelsa as elsa
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.misc

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

############PARAMS#####################
sigma = 0.05
blocksize = 8
stride = 4
nAtoms = [128,64,32]
#activations = [elsa.pyelsa_core]
sparsityLevel = 10
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

image_dc = elsa.pyelsa_core.DataContainer(noisy_raccoon)

denoising_task = elsa.pyelsa_tasks.ImageDenoisingTask(blocksize, stride, sparsityLevel, nIterations, nAtoms)

print("Image Denoising: Training dictionary...")
reconstructed_raccoon = denoising_task.train(image_dc)
reconstructed_raccoon = np.array(reconstructed_raccoon)

save_results(raccoon, noisy_raccoon, reconstructed_raccoon)
