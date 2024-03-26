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

namelist_in_1 = UserDict({ 
'target_var':'tas',
'domain':'FRA', 
'domain_size':(20,16), 
'filepath_in': '/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/input/X_UP_RCM/APPLICATION/HIST/smoothed/X_EUC12_fullvar_smth3_aero.nc', 
'filepath_ref': '/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/input/X_UP_RCM/APPLICATION/HIST/smoothed/X_EUC12_fullvar_smth3_aero.nc',
'aero_ext':True,'aero_stdz':False,'aero_var':'aero',
'filepath_aero':'/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/AEROSOLS/HIST/EUC12/aero_szopa_ant_hist_EUC12_uprcm_day_1950-2005.nc',
'var_list' : var0_nosfc1, 
'opt_ghg' : 'ONE', 
'filepath_forc' : '/cnrm/mosca/USERS/dourya/NO_SAVE/DATA/CNRM-CM5/GHG/GHG_RCP85_withONE.csv',
'filepath_grid' : '/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/target/2D/FRA_BOX1/grid_frabox.nc',
'filepath_model' : '//cnrm/mosca/USERS/dourya/NO_SAVE/NN_models/EMULALD_tas_UNETv1_CNRM-CM5-H85.h5',
'filepath_target':'/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/target/2D/FRA_BOX1/HIST/tas_aldcnrm_frabx1_hist_1951-2005.nc'
})
for key in namelist_in_1: 
    setattr(namelist_in_1,key,namelist_in_1[key]) 

namelist_in_2 = UserDict({ 
'target_var':'tas',
'domain':'FRA', 
'domain_size':(20,16), 
'filepath_in': '/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/input/X_UP_RCM/APPLICATION/RCP85/smoothed/X_EUC12_fullvar_smth3_aero.nc', 
'filepath_ref': '/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/input/X_UP_RCM/APPLICATION/HIST/smoothed/X_EUC12_fullvar_smth3_aero.nc',
'aero_ext':True,'aero_stdz':False,'aero_var':'aero',
'filepath_aero':'/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/AEROSOLS/RCP85/EUC12/aero_szopa_ant_rcp85_EUC12_uprcm_day_2006-2100.nc',
'var_list' : var0_nosfc1, 
'opt_ghg' : 'ONE', 
'filepath_forc' : '/cnrm/mosca/USERS/dourya/NO_SAVE/DATA/CNRM-CM5/GHG/GHG_RCP85_withONE.csv',
'filepath_grid' : '/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/target/2D/FRA_BOX1/grid_frabox.nc',
'filepath_model' : '//cnrm/mosca/USERS/dourya/NO_SAVE/NN_models/EMULALD_tas_UNETv1_CNRM-CM5-H85.h5',
'filepath_target':'/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/target/2D/FRA_BOX1/RCP85/tas_aldcnrm_frabx1_rcp85_2006-2100.nc'
})
for key in namelist_in_2: 
    setattr(namelist_in_2,key,namelist_in_2[key]) 


listin = []

input_aldcnrm_hist = Predictors(namelist_in_1.domain,namelist_in_1.domain_size,
                    filepath=namelist_in_1.filepath_in,
                    filepath_ref=namelist_in_1.filepath_ref,
                    var_list=namelist_in_1.var_list,
                    filepath_aero=namelist_in_1.filepath_aero,
                    filepath_forc=namelist_in_1.filepath_forc,
                    opt_ghg=namelist_in_1.opt_ghg,
                    aero_ext=namelist_in_1.aero_ext,aero_var=namelist_in_1.aero_var,aero_stdz=namelist_in_1.aero_stdz)


input_aldcnrm_rcp85 = Predictors(namelist_in_2.domain,namelist_in_2.domain_size,
                                 filepath=namelist_in_2.filepath_in,
                                 filepath_ref=namelist_in_2.filepath_ref,
                                 var_list=namelist_in_2.var_list,
                                 filepath_aero=namelist_in_2.filepath_aero,
                                 filepath_forc=namelist_in_2.filepath_forc,
                                 opt_ghg=namelist_in_2.opt_ghg,
                                 aero_ext=namelist_in_2.aero_ext,aero_var=namelist_in_2.aero_var,aero_stdz=namelist_in_2.aero_stdz)
    
listin.append(input_aldcnrm_hist)
listin.append(input_aldcnrm_rcp85)


targets = []
targets.append(Target(namelist_in_1.target_var, filepath=namelist_in_1.filepath_target,filepath_grid=namelist_in_1.filepath_grid).target.values-273.16)  
targets.append(Target(namelist_in_2.target_var, filepath=namelist_in_2.filepath_target,filepath_grid=namelist_in_2.filepath_grid).target.values-273.16)  

amodel = wrapModel(inputIn=listin,
                  targetIn=targets,
                  filepath_model = namelist_in_1.filepath_model,
                  filepath_grid=namelist_in_1.filepath_grid,
                  LR=0.0005) 
