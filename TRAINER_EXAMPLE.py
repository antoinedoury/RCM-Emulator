
import sys
import xarray as xr
import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd 
import random as rn
from sklearn.model_selection import train_test_split

#import the defined functions 
from INPUT_MAKER import * 
from make_unet import *


SCENARIO=['HIST' , 'RCP85']
var_list = ['zg850','zg700','zg500',
   'ta850','ta700','ta500',
   'hus850','hus700','hus500',
   'ua850','ua700','ua500',
   'va850','va700','va500',
   'uas','vas','psl'] 

inputs_2D=[]
inputs_1D=[]
target_times=[]
targets=[]
for scen in SCENARIOS:
  
    filepath= "/"  # path to the input data used to train the emulator
    filepath_ref="/" # path to the reference file used to standardize the 1D inputs
    filepath_aero = "/" # path to file containing the aerosols
    filepath_ghg="/" # path to the ghg file (yearly)

    # We use here the input_maker function defined in INPUT_MAKER.py file
    
    i2D , i1D = input_maker(filepath = filepath,   # path where to find the smoothed data
                             var_list = var_list,           # list of all variables except forcings to put in the input, format :['zg850','ta850',...] 
                             scen = scen,                    # scenario to do format: 'RCP85'
                             ghg = ghg,                    # use ghg in the input, format: one,multi or no
                             sol = sol,                     # use solar constant in the input, format: bool
                             ozone =ozone,                    # use ozone (chlore) in the input, format: bool
                             aero = aero,                   # use aerosols in the input, format: bool
                             seas = seas,                   # put a cos,sin vector to control the season, format : bool
                             means = means,                    # add the mean of the variables raw or stdz, format : r,s,n
                             stds = stds,                     # add the std of the variables raw or stdz, format : r,s,n
                             domain = domain,                 # select the TARGET domain,
                             size_input_domain =size_input_domain,          # size of domain, format: 8,16,32
                             filepath_ref=filepath_ref,      # path for the reference file used for sdtz
                             ref_period=ref_period,         # period for stdz must be an array of len 2
                             filepath_aero = filepath_aero,
                             filepath_ghg=filepath_ghg)  # path to aerosols files

                
    inputs_1D.append(i1D)
    inputs_2D.append(i2D)

    filepath_target= '/' # path to target file
    target_dataset = xr.open_dataset(filepath_target)
    targets.append(target_tas.values-273.16)
    target_times.append(target_dataset.time.values)
    
    
full_input=[np.concatenate(inputs_2D,axis=0),np.concatenate(inputs_1D,axis=0)]
full_target=np.concatenate(targets,axis=0)
target_time=np.concatenate(target_times,axis=0)
target_lon=target_dataset['lon']
target_lat=target_dataset['lat']


## In Doury et al (2022) we chose to mask the over seas values as we are not interested in them, we did so by setting them always to 0.

sftlf_path='/cnrm/mosca/USERS/dourya/NO_SAVE/DATA/ALADIN/sftlf_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_fx.nc'
sftlf_file = xr.open_dataset(sftlf_path)
sftlf_lon=sftlf_file['lon'][:]
sftlf_lat=sftlf_file['lat'][:]
x,y=[ int(a) for a in np.where(sftlf_lon.values.round(8)==target_lon[0,0].values.round(8))]
x1,y1=np.where(sftlf_lat.values.round(8)==target_lat[0,0].values.round(8))
if x not in x1 or y not in y1 :
    print('error')
sftlf= sftlf_file['sftlf'].values[x:(x+64),y:(y+64)]
sftlf[sftlf>0.50]=1
sftlf[sftlf<0.50]=0
full_target = full_target*sftlf

# We use here the unet_maker function defined in make_unet
unet=unet_maker( nb_inputs=len(full_input),
                size_target_domain=full_target.shape[1],
                shape_inputs=[A.shape[1:] for A in full_input],
                filters = 64 )
LR, batch_size, epochs = 0.005, 32, 100
unet.compile(optimizer=Adam(lr=LR), loss='mse', metrics=[rmse_k])  
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
             EarlyStopping(monitor='val_loss', patience=15, verbose=1),                ## Stops the fitting if val_loss does not improve after 15 iterations
             ModelCheckpoint(path_model, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model

#We set here the training sample and the validation one
rn.seed(123)
idx_train=rn.sample(range(full_target.shape[0]), int(0.8*full_target.shape[0]))
full_input_train=[full_input[k][idx_train,:,:,:] for k in range(len(full_input))]
full_input_test=[np.delete(full_input[k],idx_train,axis=0) for k in range(len(full_input))]
full_target_train=full_target[idx_train,:,:]
full_target_test=np.delete(full_target,idx_train,axis=0)

## FIt of the EMUL-UNET
unet.fit(full_input_train, full_target_train[:,:,:,None] , 
         batch_size=batch_size, epochs=epochs,  
         validation_data=(full_input_test,full_target_test[:,:,:,None]), 
         callbacks = callbacks)

