"""
Usage:
    make_mapper [options] <training_set> <mapper_path>

Options:
    --n_estimators=<n>    Number of trees in the forest [default:10]
"""


import pandas as pd
import sys
import numpy as np
import cPickle
from sklearn.ensemble import RandomTreesEmbedding
from docopt import docopt

arguments = docopt(__doc__)
input_path = arguments["<training_set>"]
n = arguments["--n_estimators"]
output_path = arguments["<mapper_path>"]

print "Reading Data"
data = pd.read_csv(input_path,header=None).values[:,1:]


print "Constructing Mapper"
mapper = RandomTreesEmbedding(n_estimators=n)
mapper.fit(data)

print "Saving Mapper to {}".format(output_path)
with open(output_path,"w") as f:
    cPickle.dump(mapper,f)

    

