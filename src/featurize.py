"""
Usage:
    featurize.py <img_dir>

"""

from __future__ import print_function
import os
from utilities import make_train_test_index, path_to_file, patchize,map2D,max_pool
from scipy.sparse import hstack
from docopt import docopt
import cPickle
import skimage.io as io
import json

arguments = docopt(__doc__)
img_dir = arguments["<img_dir>"]


def featurize(img,mapper): 
    patches = patchize(img,(5,5))
    t_patches = map2D(mapper,patches)
    feature = max_pool(t_patches,hstack)
    return feature

def make_print_unique():
    saved = [0]
    def f(x):
        if saved[0] % 100 == 0:
            print(x)
            saved[0]+=1
        else:
            saved[0]+=1
    return f        
print_unique = make_print_unique()

with open("data/mapper.pkl","r") as f:
    cls = cPickle.load(f)
with open("data/enum_class.json") as f:
    class_label_dict = json.load(f)
mapper = lambda x : cls.transform(x().ravel())
os.chdir(img_dir)
for img_file in [x for x in os.listdir(".") if x[-3:]=="jpg"]:
    img = io.imread(img_file)
    c = img_file[:-15]
    print_unique(img_file)
    out_file = img_file.replace("jpg","vw")
    fea = featurize(img,mapper)
    fea = fea.tocsr()
    f = open("feature/{}".format(out_file),"w")
    output = map(lambda x:"{}:{}".format(x[0],x[1]),zip(fea.indices,fea.data))
    print("{} |".format(class_label_dict[c]),*output,file=f)
    f.close()
