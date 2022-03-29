#################################################
######            INPUT MAKER         ###########
#################################################

## This file includes a function to prepare the inputs for the emulator 
## It contains several options to include or not different kind of input.
## The function returns an array containing the 2D inputs on the shape [time,lon,lat, variables]
## If asked it also returns a 1D input with the selected options. Its shape is [time,1,1,variables]
## it also propose 2 ways of standardizing/normalizing the 2D inputs: 
## - the first one is the one used in Doury et al (2022) : it normalize each 2D input daily ( removing the daily mean and std of each map)
## with this normalization it is highly recomended to include a vector containing at least the daily mean of each map.  
## It also possible to standardize these means according to a chosen reference file/period 
## - the second normalization option standardize the 2D inputs according to a reference file/period .
##
##
##
## Some file need to be prepared in advance : 
## - A netcdf file including at least all the variables to be included in the input. 
## In Doury et al, we smooth by a moving average filter the data, this need to done before, this function does not do it. 
## - A netcdf file including the aerosols, at the same timestep of the input file, at least covering the period of the input file
## - A file containing the external forcing (Greenhouse gases, etc. ) to the input simulation. This function is made to read this file as a ascii file, this might need to be adapted 
## - the reference file to standardize the data (1D, or 2D) if needed. It needs to be similar as the main input file. 
## 
## This function defines the input domain according to the target domain and the input domain size. This is not done automatically and is coded in hard in the function. 
## It needs to be adapted when one wants to use it on a different domain. 
## There are 2 target domains included in this function : 'MBP', a 64*64 grid point used in Doury et al. (2022) and ALP a 128*128 grid point domain centered over the ALPS.
## These 2 target domains are subsamples of the ALADINv63 domain, as the emulator was designed to emulate this RCM. 
## For each of these two domains, the corners of the input domain are define according to the size of the input.
## They are define on a subsample of the CNRM-CM5 grid (the inputs grid for Doury et al (2022)), the description of the grid is available in the repository as 'large_input_grid'.




def standardize(data ):
    import numpy as np
    mean =  np.nanmean(data,axis=(1,2), keepdims=True)
    sd   =  np.nanstd(data,axis=(1,2), keepdims=True)
    ndata = (data - mean)/sd
    return (ndata)


def standardize2(data,ref ):
    import numpy as np
    mean =  np.nanmean(ref,axis=(0,1,2), keepdims=True)
    sd   =  np.nanstd(data,axis=(0,1,2), keepdims=True)
    ndata = (data - mean)/sd
    return (ndata)


def mon2day(data):
    import numpy as np
    nbyrs=int(data.shape[0]/12)
    months=np.tile([31,28,31,30,31,30,31,31,30,31,30,31],nbyrs)
    daily = np.concatenate([np.transpose( np.tile(data[i,:,:,:],months[i]),[2,0,1]) for i in range(len(months)) ])
    return daily
    

def input_maker(filepath,                       # path where to find the data
                var_list,                       # list of all variables except forcings to put in the input, format :['zg850','ta850',...] 
                scen,                           # scenario to do format: 'RCP85'
                ghg = 'no',                     # use ghg in the input, format: one,multi or no
                sol = False,                    # use solar constant in the input, format: bool
                ozone =False,                   # use ozone (chlore) in the input, format: bool
                aero = False,                   # use aerosols in the input, format: bool
                seas = False,                   # put a cos,sin vector to control the season, format : bool
                means = 'n',                    # add the mean of the variables raw or stdz, format : r,s,n
                stds = 'n',                     # add the std of the variables raw or stdz, format : r,s,n
                domain = 'MBP',                 # select the TARGET domain, must be define in advance
                size_input_domain =16,          # size of domain, format: 8,16,32, must be define in advance and documented in the function see below
                filepath_ref='',                # path for the reference file used for sdtz
                ref_period=[],                  # period for stdz must be an array of len 2
                filepath_aero = '',             # path to aerosols files
                stand=1):                       # choice of the standardization option 
                
                
    
    
    '''
    PAKCAGES
    '''
    import xarray as xr
    import numpy as np
    import os
    from math import cos,sin,pi
    from pandas import to_datetime
    
    from astropy.io import ascii

    
    '''
    VERIFICATIONS
    '''
    try: 
        var_list
    except NameError:
        print('please choose a varlist')
        
    try: 
        scen
    except NameError:
        print('please choose a scenario')

    possible_ghg = ['one','multi','no']
    if ghg not in possible_ghg :
        raise ValueError('ghg argument must be in [one,multi,no]')
        
    if not isinstance(aero, bool) :
        raise ValueError('aero must be bool')
        
    if not isinstance(seas, bool) :
        raise ValueError('seas must be bool')
        
    possible_means = ['r','s','n']
    if means not in possible_means :
        raise ValueError('means must be in {}'.format(possible_means))

    possible_means = ['r','s','n']
    if stds not in possible_means :
        raise ValueError('std must be in {}'.format(possible_means))
        
    possible_domains = ['MBP','PYR','ALP']
    if domain not in possible_domains :
        raise ValueError('domain must be in {}'.format(possible_domains))
        

    possible_size_domains = [8,16,22,32,(22,16)]
    if size_input_domain not in possible_size_domains :
        raise ValueError('size input domain must be in {}'.format(possible_size_domains))
    
    if not os.path.isfile(filepath):
        raise ValueError('File does not exist')
    
    if means != 'n' or stds != 'n':
            if not os.path.isfile(filepath_ref):
                raise ValueError('File for stdz means/stds does not exist')
            if len(ref_period) != 2 :
                raise ValueError('ref_period must be an array of size 2 [start year; end year]')
    
    '''
    EXTRACT SMOOTHED FILE BX2
    '''
    DATASET=xr.open_dataset(filepath)
    
    ''' 
    MAKE THE 2D INPUT ARRAY
    SHAPE [nbdays, lat, lon, nb_vars]
    '''
    if domain == 'MBP' : 
        if size_input_domain == 8:
            lon_b,lon_e,lat_b,lat_e = 44,52,12,20
        elif size_input_domain == 16:
            lon_b,lon_e,lat_b,lat_e = 39,55,9,25
        elif size_input_domain == 32:
            lon_b,lon_e,lat_b,lat_e = 34,66,4,36
    if domain=='ALP' : 
        if size_input_domain == 16:
            lon_b,lon_e,lat_b,lat_e = 44,60,10,26
        elif size_input_domain == (22,16):
            lon_b,lon_e,lat_b,lat_e = 40,62,11,27
        elif size_input_domain == 22:
            lon_b,lon_e,lat_b,lat_e = 40,62,9,31
        elif size_input_domain == 32:
            lon_b,lon_e,lat_b,lat_e = 35,67,5,37
            
            
    var_list2=var_list[:]
    if 'GCM' in filepath:
        for i in range(len(var_list)):
            if '200' in var_list[i]:
                var_list2[i]=var_list[i][:-3]+'250'

                
    
    if 'sst' in var_list:
        DATASET['sst'].values[np.where(np.isnan(DATASET['sst']))]=0
                
    INPUT_2D=np.transpose(np.asarray([DATASET[i].values[:,lat_b:lat_e,lon_b:lon_e] for i in var_list2]),[1,2,3,0])
    if stand==1:
       INPUT_2D_SDTZ = standardize(INPUT_2D)
    elif stand==2:
       DATASET_ref=xr.open_dataset(filepath_ref).sel(time=slice(str(ref_period[0]),str(ref_period[1])))
       REF=np.transpose(np.asarray([DATASET_ref[i].values[:,lat_b:lat_e,lon_b:lon_e] for i in var_list2]),[1,2,3,0])
       INPUT_2D_ARRAY = standardize2(INPUT_2D,REF)
    
    if aero:
        start=0
        if 'HIST' in filepath and 'RCM' in filepath:
            start=12
        aero_dataset= xr.open_dataset(filepath_aero).sel(time=DATASET.time,method='pad').sel(lon=DATASET.lon[lon_b:lon_e],lat=DATASET.lat[lat_b:lat_e],method='nearest').aero
        INPUT_2D_ARRAY=np.concatenate([INPUT_2D_SDTZ,aero_dataset.values[:,:,:,None]],axis=3)
    else:
        INPUT_2D_ARRAY=INPUT_2D_SDTZ
    
    '''
    MAKE THE 1D INPUT ARRAY
    CONTAINS MEANS, STD, GHG, SEASON IF ASKED
    '''
    INPUT_1D=[]
    if ghg=='one' : 
        if scen.lower() == 'hist':
            s='rcp85'
        else :
            s=scen.lower()
            
        yr_b=to_datetime(DATASET['time'].values[0]).year
        yr_e=to_datetime(DATASET['time'].values[-1]).year
        forcings = ascii.read('/home/dourya/WORK/RAWDATA/GHG/ONE/'+s.lower()+'.asc')
        vect_ghg =np.reshape(np.repeat(forcings['GHG'][np.where(forcings['YEARS']==yr_b)[0][0]:np.where(forcings['YEARS']==(yr_e+1))[0][0]],365,axis=0),(INPUT_2D.shape[0],1,1,1))
        INPUT_1D.append(vect_ghg)
        
    elif ghg=='multi':
        if scen.lower() == 'hist':
            s='rcp85'
        else :
            s=scen.lower()
            
        yr_b=to_datetime(DATASET['time'].values[0]).year
        yr_e=to_datetime(DATASET['time'].values[-1]).year
        forcings = np.loadtxt('/home/dourya/WORK/RAWDATA/GHG/MULTI/GHG_'+s.upper()+'.dat')
        
        
        ghg_ref=forcings[np.where(forcings[:,0]==ref_period[0])[0][0]:np.where(forcings[:,0]==ref_period[1]+1)[0][0],1:6]
        ghg_ref_mean=ghg_ref.mean(axis=0)
        ghg_ref_std=ghg_ref.std(axis=0)
        
        vect_ghg=np.reshape(
                np.repeat(
                        (forcings[np.where(forcings[:,0]==yr_b)[0][0]:np.where(forcings[:,0]==yr_e+1)[0][0],1:6] - ghg_ref_mean)/ghg_ref_std,
                        365,axis=0),
                        (INPUT_2D.shape[0],1,1,5))
        INPUT_1D.append(vect_ghg)
        
    if sol :
        if scen.lower() == 'hist':
            s='rcp85'
        else :
            s=scen.lower()
            
        yr_b=to_datetime(DATASET['time'].values[0]).year
        yr_e=to_datetime(DATASET['time'].values[-1]).year
        forcings = np.loadtxt('/home/dourya/WORK/RAWDATA/GHG/MULTI/GHG_'+s.upper()+'.dat')
        csol_ref=forcings[np.where(forcings[:,0]==ref_period[0])[0][0]:np.where(forcings[:,0]==ref_period[1])[0][0],6]
        csol_ref_mean=csol_ref.mean(axis=0)
        csol_ref_std=csol_ref.std(axis=0)
        csol=(forcings[np.where(forcings[:,0]==yr_b)[0][0]:np.where(forcings[:,0]==yr_e+1)[0][0],6] - csol_ref_mean)/csol_ref_std
        
        INPUT_1D.append(csol.repeat(365).reshape(INPUT_2D.shape[0],1,1,1))
        
    if ozone :
        if scen.lower() == 'hist':
            s='rcp85'
        else :
            s=scen.lower()
            
        yr_b=to_datetime(DATASET['time'].values[0]).year
        yr_e=to_datetime(DATASET['time'].values[-1]).year
        forcings = np.loadtxt('/home/dourya/WORK/RAWDATA/GHG/MULTI/GHG_'+s.upper()+'.dat')
        oz_ref=forcings[np.where(forcings[:,0]==ref_period[0])[0][0]:np.where(forcings[:,0]==ref_period[1])[0][0],7]
        oz_ref_mean=oz_ref.mean(axis=0)
        oz_ref_std=oz_ref.std(axis=0)
        oz=(forcings[np.where(forcings[:,0]==yr_b)[0][0]:np.where(forcings[:,0]==yr_e+1)[0][0],7] - oz_ref_mean)/oz_ref_std    
        
        INPUT_1D.append(oz.repeat(365).reshape(INPUT_2D.shape[0],1,1,1))
    if means != 'n' : 
        vect_means = INPUT_2D.mean(axis=(1,2))
    if stds != 'n' : 
        vect_std = INPUT_2D.std(axis=(1,2))   
    
    if means == 's' or stds == 's' :        
        '''
        EXTRACT REFERENCES FILE USED FOR SDTZ MEANS
        &
        COMPUTE MEANS
        '''
        DATASET_ref=xr.open_dataset(filepath_ref)
        
      
        years=np.asarray([y for y in to_datetime(DATASET_ref['time'].values).year])
        b=np.where(years==ref_period[0])
        e=np.where(years==ref_period[1])
        REF_ARRAY=np.transpose(np.asarray([DATASET_ref[i].values[b[0][0]:e[0][0],lat_b:lat_e,lon_b:lon_e] for i in var_list]),[1,2,3,0])
        DATASET_ref.close()
    if means == 's':
        ref_means=REF_ARRAY.mean(axis=(1,2))  
        ref_means_mean=ref_means.mean(axis=0)
        ref_means_std=ref_means.std(axis=0)
        vect_means_stdz = (vect_means - ref_means_mean)/ref_means_std
        INPUT_1D.append(vect_means_stdz.reshape(INPUT_2D.shape[0],1,1,INPUT_2D.shape[3]))
    elif means=='r':
        INPUT_1D.append(vect_means.reshape(INPUT_2D.shape[0],1,1,INPUT_2D.shape[3]))

            
    if stds == 's':
        ref_stds=REF_ARRAY.std(axis=(1,2))  
        ref_stds_mean=ref_stds.mean(axis=0)
        ref_stds_std=ref_stds.std(axis=0)
        vect_std_stdz = (vect_std -ref_stds_mean)/ref_stds_std
        INPUT_1D.append(vect_std_stdz.reshape(INPUT_2D.shape[0],1,1,INPUT_2D.shape[3]))
    elif stds=='r':
        INPUT_1D.append(vect_std.reshape(INPUT_2D.shape[0],1,1,INPUT_2D.shape[3]))
            
    if seas : 
        cosvect = np.tile([cos(2*i*pi/365)  for i in range(365)],int(INPUT_2D.shape[0]/365)) 
        sinvect = np.tile([sin(2*i*pi/365)  for i in range(365)],int(INPUT_2D.shape[0]/365)) 
        INPUT_1D.append(cosvect.reshape(INPUT_2D.shape[0],1,1,1))
        INPUT_1D.append(sinvect.reshape(INPUT_2D.shape[0],1,1,1))
    INPUT_1D_ARRAY= np.concatenate(INPUT_1D,axis=3)
    DATASET.close()
    
    return INPUT_2D_ARRAY,INPUT_1D_ARRAY
