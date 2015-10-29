import os, sys
import scipy.io
import numpy as np
import cPickle as pickle

MODEL_FILE=['./mdl/weights_srnet_x2_52.p','./mdl/weights_srnet_x2_310.p']
mdls = []
for f in MODEL_FILE:
    mdls += [pickle.load(open(f, 'rb'))]

scipy.io.savemat("weights_srnet_x2_52.mat",mdls[0])
scipy.io.savemat("weights_srnet_x2_310.mat",mdls[1])