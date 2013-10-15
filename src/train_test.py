"""
Usage:
    train_test.py <dir>

"""

import os
from utilities import append_files,make_train_test_index, path_to_file, patchize,map2D,max_pool
from scipy.sparse import hstack
from docopt import docopt

args = docopt(__doc__)
img_dir = args["<dir>"]

def make_train_test_set(img_dir): 
    os.chdir(img_dir)
    all_images = [x for x in os.listdir(".") if x[-2:] == "vw" ]
    all_images = sorted(all_images,key=lambda x: x.lower())
    train_index, test_index = make_train_test_index(all_images,lambda x:x[:-15])
    train_files = path_to_file(train_index)
    test_files = path_to_file(test_index)
    append_files(train_files,"trainingSet.vw")
    append_files(test_files,"testingSet.vw")

make_train_test_set(img_dir)
