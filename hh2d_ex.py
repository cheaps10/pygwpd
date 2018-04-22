#!/usr/bin/env python
import os
import time
import numpy as np

import potential
import single_surface_model as ssm

t_start = time.time()
input_params = {}
#Potential energy model
input_params['model'] = potential.HH2dMCTDH()

#Basis function and basis set parameters
input_params['mass'] = 1.0              #Mass of system
input_params['x0'] = np.array([2.0, 2.0])   #Initial coordinate for each dimension
input_params['k0'] = np.array([0.0, 0.0])    #Initial momentum for each dimension
input_params['basis_function_width'] = np.array([0.5, 0.5])  #alpha - basis function widht parameter
input_params['basis_set_size'] = 150    #Number of basis functions
input_params['add_cs'] = False #Boolean to make last basis function correspond to the initial state
#Called 'cs' for coherent state

#run parameters
input_params['integrator'] = 'RK4' 
input_params['time_step'] = 0.05   
input_params['prop_time'] = 25.0   
input_params['save_wf_times'] = np.array([5.0, 10., 20., 25.])
input_params['save_all_wf'] = False
input_params['store_wf'] = False 
input_params['write_wf'] = True
input_params['job_name'] = 'hh2d_ex' 
input_params['tcf_type'] = 2 


#Solution of QM coefficients
input_params['matrix_type'] = 'pseudospectral' 
# input_params['solver'] = 'bicg' 
input_params['solver'] = 'lstsq'
input_params['svd_threshold'] = 1e-4

#Initialize ODE class
hh2d_ex = ssm.GWPD_SS_Model(**input_params)

hh2d_ex.setup_basis()
hh2d_ex.propagate()

with open(ss_test.outfile, 'a') as f:
    f.write('Run time is : {:5.2f} seconds\n'.format(time.time() - t_start))

