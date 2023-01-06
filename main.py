import numpy as np
import matplotlib.pyplot as plt

#define the mask for the corresponding part where size is the image dimension (same for all images)
#and part is the area of the mask that will contains ones
#for 16 masks
def mask_maker(size, indice):
    if indice > indice*indice-1:
      print("Mask index out of range")
      return np.zeros(size)
    mask = np.zeros(size)
    indice_i = indice // 4
    indice_j = indice % 4
    offset = int(size[0]/4)
    start_i = offset * int(indice_i)
    start_j = offset * int(indice_j)
    mask[start_i:start_i+offset, start_j:start_j+offset] = np.ones((offset, offset))
    return mask





