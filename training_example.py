#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:46:07 2024

@author: dourya
"""

import sys
import numpy as np
import xarray as xr
sys.path.append('/cnrm/mosca/USERS/dourya/scripts/python/NN/GITHUB')
from emu_funs import Predictors,Pred,Target,wrapModel,Domain  
from emu_funs import launch_gpu
from collections import UserDict

# ----------------- Namelist ------------------------
# Please modify the following namelist with the values corresponding to 
# your emulator.  

launch_gpu(2)


var0_nosfc1 = ['zg850','zg700','zg500',
           'ta850','ta700','ta500',
           'hus850','hus700','hus500',
           'ua850','ua700','ua500',
           'va850','va700','va500']

## The idea is to create 1 object predictor for each simulation to put in the training set

namelist_in_1 = UserDict({ 
'target_var':'tas',
'domain':'ALP', 
'domain_size':(20,16), 
'filepath_in': 'path to inputs', 
'filepath_ref': 'path to reference file, should be similar to input file, must be the same for any predictor set',
'aero_ext':True,'aero_stdz':False,'aero_var':'aero',
'filepath_aero':'path to corresponding aerosol file (must be daily)',
'var_list' : var0_nosfc1, 
'opt_ghg' : 'ONE', 
'filepath_forc' : 'path to corresponding .csv including ghg, see example file',
'filepath_grid' : 'path to a .nc with output grid information, can be extracted from target file',
'filepath_model' : 'path to save the model, format can be .h5 or .keras',
'filepath_target': 'path to target file (high resolution, inluding target domain and target variable)'
})
for key in namelist_in_1: 
    setattr(namelist_in_1,key,namelist_in_1[key]) 

input_1 = Predictors(namelist_in_1.domain,
                     namelist_in_1.domain_size,
                    filepath=namelist_in_1.filepath_in,
                    filepath_ref=namelist_in_1.filepath_ref,
                    var_list=namelist_in_1.var_list,
                    filepath_aero=namelist_in_1.filepath_aero,
                    filepath_forc=namelist_in_1.filepath_forc,
                    opt_ghg=namelist_in_1.opt_ghg,
                    aero_ext=namelist_in_1.aero_ext,
                     aero_var=namelist_in_1.aero_var,
                     aero_stdz=namelist_in_1.aero_stdz)


namelist_in_2 = UserDict({ 
'same as namelist_in_1 with a different input file ex: other simulation. All file must be updated
})
for key in namelist_in_2: 
    setattr(namelist_in_2,key,namelist_in_2[key]) 

input_2 = Predictors(namelist_in_2.domain,
                     namelist_in_2.domain_size,
                    filepath=namelist_in_2.filepath_in,
                    filepath_ref=namelist_in_2.filepath_ref,
                    var_list=namelist_in_2.var_list,
                    filepath_aero=namelist_in_2.filepath_aero,
                    filepath_forc=namelist_in_2.filepath_forc,
                    opt_ghg=namelist_in_2.opt_ghg,
                    aero_ext=namelist_in_2.aero_ext,
                      aero_var=namelist_in_2.aero_var,
                      aero_stdz=namelist_in_2.aero_stdz)

listin = []

listin.append(input_1)
listin.append(input_2)


targets = []
targets.append(Target(namelist_in_1.target_var, filepath=namelist_in_1.filepath_target,filepath_grid=namelist_in_1.filepath_grid).target.values-273.16)  
targets.append(Target(namelist_in_2.target_var, filepath=namelist_in_2.filepath_target,filepath_grid=namelist_in_2.filepath_grid).target.values-273.16)  

amodel = wrapModel(inputIn=listin,
                  targetIn=targets,
                  filepath_model = namelist_in_1.filepath_model,
                  filepath_grid=namelist_in_1.filepath_grid,
                  LR=0.0005) 
