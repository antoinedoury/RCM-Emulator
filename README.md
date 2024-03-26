# RCM-Emulator


This repository contains the main functions used to build the Emul-UNet used in Doury et al,2022 and 2024 (in review). 

It is organised as follow : \\
emulator_functions.py is including all function necessary to build,train and apply the emulator\\
training_example.py illustrates how to train the emulator. The design of the network is hidden and automatically done by the unet_maker function inside the wrapModel class\\
application.py illustrates how to use a trained emulator.\\

The X_EUC12_fullvar_smth3_aero......nc is an example of the shape of an input file to be passed to the predictor class
The GHG....csv is an example of external forcing file for RCP85. 

Doury, A., Somot, S., Gadat, S. et al. Regional climate model emulator based on deep learning: concept and first evaluation of a novel hybrid downscaling approach. Clim Dyn (2022). https://doi.org/10.1007/s00382-022-06343-9
