'''
originally writtin by E. Najarro
https://github.com/enajx/HebbianMetaLearning
'''

import numpy as np
from numba import njit
import torch
import torch.nn as nn

@njit
def hebbian_update(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3, inds, r):

        heb_offset = 0
        ## Layer 1         
        for i in range(weights1_2.shape[1]):
            for j in range(weights1_2.shape[0]):
                idx = (weights1_2.shape[0]-1)*i + i + j
                change  = heb_coeffs[inds[idx]][0] * ( 
                          heb_coeffs[inds[idx]][1] * o0[i] * o1[j]
                        + heb_coeffs[inds[idx]][2] * o0[i]
                        + heb_coeffs[inds[idx]][3] * o1[j]  
                        + heb_coeffs[inds[idx]][4] * r  
                        + heb_coeffs[inds[idx]][5])

                weights1_2[:,i][j] += change

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]):
            for j in range(weights2_3.shape[0]):
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                change  = heb_coeffs[inds[idx]][0] * ( 
                          heb_coeffs[inds[idx]][1] * o1[i] * o2[j]
                        + heb_coeffs[inds[idx]][2] * o1[i]
                        + heb_coeffs[inds[idx]][3] * o2[j]
                        + heb_coeffs[inds[idx]][4] * r
                        + heb_coeffs[inds[idx]][5])
                weights2_3[:,i][j] += change

        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]):
            for j in range(weights3_4.shape[0]):
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                change  = heb_coeffs[inds[idx]][0] * (
                          heb_coeffs[inds[idx]][1] * o2[i] * o3[j]
                        + heb_coeffs[inds[idx]][2] * o2[i]
                        + heb_coeffs[inds[idx]][3] * o3[j]  
                        + heb_coeffs[inds[idx]][4] * r
                        + heb_coeffs[inds[idx]][5])

                weights3_4[:,i][j] += change

        return weights1_2, weights2_3, weights3_4

