from random import shuffle
from itertools import groupby
from numpy import array,prod
import numpy as np
import skimage.io as io

def map2D(f,arr,level=2):
    s = arr.shape
    ns = [prod(s[:level])]
    ns.extend(s[level:])
    tmp = arr.reshape(ns)
    return array(map(f, tmp)).reshape(s[:level])

def functionize(arr,level=2):
    return map2D(lambda x: lambda :x,arr,level)

def patchize(img,size):
    sz = img.itemsize
    h,w = img.shape
    bh,bw = size
    shape = (h/bh, w/bw, bh, bw)
    strides = sz*np.array([w*bh,bw,w,1])
    blocks=np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return functionize(blocks,2)

def max_pool(arr,agg):
    h,w = arr.shape
    fQ = arr[:h/2,:w/2]
    sQ = arr[:h/2,w/2:]
    tQ = arr[h/2:,:w/2]
    Q = arr[h/2:,w/2:]
    return agg([fQ.sum(),sQ.sum(),tQ.sum(),Q.sum()])
            
def split_randomly(arr,split_at=30):
    shuffle(arr)
    return arr[:split_at],arr[split_at:]

def append_files(list_file_handle,dst_file):
    with open(dst_file,"a") as f:
        for files in list_file_handle :
            f.write(files.read())
            files.close()

def path_to_file(list_file_path):
	"""
	input  :  [file_path]
	output :  generator(file_handler)
	"""
	return (open(x) for x in list_file_path)

def make_train_test_index(arr,key) :
	grouped = groupby(arr,key)
	train_index = []
	test_index = []
	for key,gen in grouped: 
		train,test = split_randomly(list(gen))
		train_index.extend(train)
		test_index.extend(test)
	return train_index,test_index	
