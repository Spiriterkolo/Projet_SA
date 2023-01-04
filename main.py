import numpy as np
import matplotlib.pyplot as plt

#define the mask for the corresponding part where size is the image dimension (same for all images)
#and part is the area of the mask that will contains ones
def mask_maker(size, indice):
    mask = np.zeros(size)
    indice_i = indice // 4
    indice_j = indice % 4
    start_i = size[0] * int(indice_i / 4)
    start_j = size[1] * int(indice_j / 4)
    offset = int(size[0]/4)
    mask[start_i:start_i+offset, start_j:start_j+offset] = np.ones((offset, offset))
    return mask





