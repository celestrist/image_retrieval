import os
from utilities import make_train_test_index, path_to_file, patchize,map2D,max_pool
from scipy.sparse import hstack

def train(classsifier,dataset):
    classifier.fit(dataset)
    return classifier

def make_train_test_set(img_dir): 
    home = os.path.expanduser("~")
    d = os.path.join(home,img_dir)        
    os.chdir(d)
    all_images = [x for x in os.listdir(".") if x[-3:] == "jpg" ]
    all_images = sorted(all_images,key=lambda x: x.lower())
    train_index, test_index = make_train_test_index(all_images,lambda x:x[:-15])
    train_files = path_to_file(train_index)
    test_files = path_to_file(test_index)
    append_files(train_files,"trainingSet.vw")
    append_files(test_files,"testingSet.vw")
        
def feauturize(img,mapper): 
    patches = patchize(img,(5,5))
    t_patches = map2D(mapper,patches)
    
    feature = max_pool(t_patches,hstack)
    return feature
