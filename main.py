import numpy as np
import matplotlib.pyplot as plt


def sd(face, softui, maski):
    sdf = maski * softui + (1 - maski) * face
    return sdf

def softdcr(face, softu, mask):
    softufs = []
    for i in range(len(softu)):
        facet = sd(face, softu[i], mask[i])
        softufs.append(facet)
    return softufs


