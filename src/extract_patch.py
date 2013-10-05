"""
Input  : A directory with images, directory of the output csv
Output : A csv file with (n_images*n_patches_per_image) lines and (size_patches) columns

Usage:
    extract_patch.py <input_directory> <output_file> [options]

Options:
    --n_patches=<n>   Total patches to be extracted in an image [default: 100].
    --size=<px>       Length or width of patches in pixels [default: 6].
"""
from __future__ import print_function
from sklearn.feature_extraction.image import extract_patches_2d
import skimage.io as io
import os
from sklearn import preprocessing
import numpy as np
from docopt import docopt


arguments = docopt(__doc__)
input_directory = arguments["<input_directory>"]
output_file = arguments["<output_file>"]
n_patches = int(arguments["--n_patches"])
s = int(arguments["--size"])
size = (s,s)

# Types
# ===============================================
# image                 :: (n,n)[float]
# patch                 :: (6,6)[float]
# patches               :: (100)[patch]




# extract :: image -> patches
def extract(image):
    patch = extract_patches_2d(image,\
                               size,\
                               max_patches=n_patches)
    return preprocessing.scale(np.array(patch,dtype=np.float))

# linearlize :: patch -> (36)[float] := l_patch
def linearlize(patch):
    return patch.ravel()

# tag :: l_patch -> [string:l_patch] := taged_l_patch
def tag(l_patch,str):
    l_patch = l_patch.astype(np.object)
    return np.insert(l_patch,0,str)

# log_console :: string -> IO()
def log_console(msg):
    length = len(msg)
    print(msg)
    print("="*length)

# make_print_unique ::  -> string -> Maybe(IO())
def make_print_unique():
    saved = []
    def f(x):
        if x in saved:
            pass
        else:
            print(x)
            saved.append(x)
    return f        

# proccess_patch        = linearlize > scale > tag
#                       :: patch string -> taged_l_patch
def process_patch(patch,id):
    return tag(linearlize(patch),id)

# print_processed_patch :: taged_l_patch file -> IO()
def print_processed_patch(t_patch,file):
    print(*t_patch,file=file,sep=",")

# process_print_patch   :: patch string file -> IO()    
def process_print_patch(patch,id,file):
    print_processed_patch(process_patch(patch,id),file)

# process_image :: img_name:=string -> IO(patches)   
def process_image(img_name,img_directory,log_function):
    img_class = img_name[:-15]
    img_path = os.path.join(img_directory,img_name)
    log_function(img_class)
    
    return extract(io.imread(img_path))
    
    
with open(output_file,"w") as f:
    
    log_console("Extracting Patches")
    print_unique = make_print_unique()
    img_list = [x for x in os.listdir(input_directory) if x[-3:] == "jpg"]
    img_list = sorted(img_list,key=lambda x : x.lower())
    assert len(img_list) == 9144

    for img in img_list:
        for patch in process_image(img,input_directory,print_unique):
            process_print_patch(patch,img,f)
