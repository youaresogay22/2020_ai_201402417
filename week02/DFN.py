import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join

datax = np.load(join('data', 'MNIST.npy'))
datay = np.load(join('data', 'Label.npy'))

print(datax.shape)
print(datay.shape)

img = plt.imshow(datax[1])