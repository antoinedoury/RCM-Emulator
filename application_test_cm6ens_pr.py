#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:28:58 2024

@author: dourya
"""

import sys
import numpy as np
import xarray as xr
sys.path.append('/cnrm/mosca/USERS/dourya/scripts/python/NN/GITHUB')
from emu_funs import Predictors,Pred,Target,wrapModel,Domain  
from emu_funs import launch_gpu
from collections import UserDict
from datetime import date

# ----------------- Namelist ------------------------
# Please modify the following namelist with the values corresponding to 
# your emulator.  

launch_gpu(0)


dir_cm6='/cnrm/mosca/USERS/dourya/NO_SAVE/DATA/CNRM-CM6/'

var0_nosfc1 = ['zg850','zg700','zg500',
           'ta850','ta700','ta500',
           'hus850','hus700','hus500',
           'ua850','ua700','ua500',
           'va850','va700','va500']

filename='tas_ALPX-12_CNRM-CM6_historical_r15i1p1f2_CNRM_CNRM-ALADIN63-emul-CNRM-UNET11-tP1_v1-r1_day_19500101-20141231.nc'


namelist_out = UserDict({ 
    'target_var':'pr',
    'domain':'FRA', 
    'domain_size':(22,16), 
    'filepath_in':dir_cm6+'cassou_ens/CNRM-CM6-1_historical_r15i1p1f2/X_fullvar_smth3_historical_r15i1p1f_1950-2014.nc', 
    'filepath_ref':'/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/input/X_UP_RCM/APPLICATION/HIST/smoothed/X_EUC12_fullvar_smth3_aero.nc', 
    'var_list' : var0_nosfc1, 
    'opt_ghg' : 'ONE', 
    'filepath_forc' : dir_cm6+'GHG/GHG_CMIP6ssp245.csv',
    'filepath_grid' : '/cnrm/mosca/USERS/dourya/NO_SAVE/data_NN/target/2D/FRA_BOX1/grid_frabox.nc',
    'filepath_model' : '/cnrm/mosca/USERS/dourya/NO_SAVE/NN_models/development/FRA/emul_pr_vpaper_fra_nosfc_cnrm_2216.h5'   ,
    'filepath_aero':'/cnrm/mosca/USERS/dourya/NO_SAVE/DATA/CNRM-CM6/AEROSOLS/od550aer_AERday_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc',
    'aero_ext':True,'aero_stdz':True,'aero_var':'od550aer',
    'filepath_out':dir_cm6+'cassou_ens/CNRM-CM6-1_historical_r15i1p1f2/'+filename,
    })

attributes={
    "Conventions" : 'CF-1.10',
        "activity_id": 'emulation',
        "contact": 'contact.aladin-cordex@meteo.fr',
        "domain_id": 'ALPX-12',
        "domain": 'EUR-11 CORDEX domain cropped to a domain centered on Alps.',
        "driving_experiment":'Historical run with GCM forcing',
        "driving_experiment_id":'historical',
        "driving_institution_id":'CNRM-CERFACS',
        "driving_source_id":'CNRM-CM6',
        "driving_variant_label":'r15i1p1f',
        "emulator":  'CNRM_UNET11, introduced in Doury et al, 2022, is based fully'\
                        'convolutional neural network shaped from the UNeT base (Ronnenberg et al, 2015).'\
                            'The network is  minimizing the mean squared error (mse) loss function.',
        "emulator_id":'CNRM-UNET11',
        "frequency": 'day',
        "further_info_url":'',
        "institution":'Centre National de Recherches Meteorologiques,CNRM, Toulouse, France',
        "institution_id":"CNRM",
        'mip_era':"CMIP6",
        "native_resolution" : "0.11°",
        "product":"emulator_output",
        "project_id":"I4C",
        "realm":"",
        "source": "CNRM-UNET11 is trained here for the near surface temperature of CNRM-ALADIN63 RCM ",
        "source_id":'ALADIN63-emul-CNRM-UNET11',
        "source_type":'RCM_emulator',
        "version_realization":'v1-r1',
        "target_institution_id":'CNRM',
        "target_source_id":'CNRM-ALADIN63',
        "target_version_realization":'v1',
        "tracking_id":'',
        "training":'Trained using PP predictors from the CNRM ALADIN63 RCM nested into (CMI5) CNRM-CM5 for the period 1950-2100'\
                    'combining the historical and rcp85 simulations. The emulator is trained in perfect model framework, implying'\
                    'that predictors and predictands come from the same RCM simulation, predictors are coarsened to the GCM resolution'\
                    '(150km) and a spatial smoothing is applied, following the protocol described in Doury et al, 2022. The preditors'\
                    'include the geopotential altitude, the temperature, the specific humidity and the eastern and northern wind components'\
                    'at 3 pressure levels (500, 700, 850 hpa). External forcings such as the GHG concentration, and the solar and ozone values'\
                    'are also inputs.The predictor list and pre-processing is identical from the one described in Doury et al 2022,'\
                        'except that we remove here the uas,vas and psl variables from predictors list.',
        "training_id": 'tP1',
        "variable_id":'tas',
        "version_realization_info":'this is the 1st realization of the emulator CNRM-ALADIN63-emul-CNRM-UNET11-TP1 over ALPX-12 domain.',
        "license":'',
        "reference":"""Doury, A., Somot, S., Gadat, S. et al. Regional climate model emulator based on deep learning: 
            concept and first evaluation of a novel hybrid downscaling approach. Clim Dyn 60, 1751–1779 (2023).
            https://doi.org/10.1007/s00382-022-06343-9""",
        "creation_date":date.today().strftime("%d/%m/%Y")
        }


for key in namelist_out: 
    setattr(namelist_out,key,namelist_out[key]) 
    

listin_pred = []

aninput_pred = Predictors(namelist_out.domain,namelist_out.domain_size,
                filepath=namelist_out.filepath_in,
                filepath_ref=namelist_out.filepath_ref,
                var_list=namelist_out.var_list,
                filepath_forc=namelist_out.filepath_forc,
                filepath_aero=namelist_out.filepath_aero,
                opt_ghg=namelist_out.opt_ghg,
                aero_ext=namelist_out.aero_ext,aero_var=namelist_out.aero_var,aero_stdz=namelist_out.aero_stdz)

listin_pred.append(aninput_pred)

apred = Pred(namelist_out.domain, namelist_out.domain_size,
              inputIn = listin_pred,
              filepath_grid =namelist_out.filepath_grid,
              filepath_out=namelist_out.filepath_out,
              filepath_model=namelist_out.filepath_model,
              attributes=attributes)

print(f"apred is {apred.ds}")     