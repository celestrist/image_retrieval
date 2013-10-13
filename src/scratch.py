"""
input :
"""
import numpy as np
import skimage.io as io

def patchize(img,size):
    sz = img.itemsize
    h,w = img.size
    bh,bw = size
    shape = (h/bh, w/bw, bh, bw)
    strides = sz*np.array([w*bh,bw,w,1])
    blocks=np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return blocks
