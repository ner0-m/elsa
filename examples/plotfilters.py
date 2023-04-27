import numpy as np
import matplotlib.pyplot as plt
from skimage.transform.radon_transform import _get_fourier_filter

filters = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]

for ix, f in enumerate(filters):
    response = _get_fourier_filter(2000, f)
    plt.plot(response, label=f)


plt.xlim([0, 1000])
plt.xlabel("frequency")
plt.legend()
plt.show()
