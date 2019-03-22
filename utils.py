import numpy as np
import cmath

def cartesianToPolar(input):
    magnitude = np.vectorize(np.linalg.norm)
    phase = np.vectorize(np.angle)

    return np.dstack([magnitude, phase])

def polarToCartesian(input):
    output = np.zeros((input.shape[0], input.shape[1]), dtype=np.complex64)

    it = np.nditer(output, flags=['multi_index'])
    while not it.finished:
        output[it.multi_index] = cmath.rect(input[it.multi_index][0], input[it.multi_index][1])
        temp = it.iternext()

    return output