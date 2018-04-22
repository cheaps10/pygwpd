#!/usr/bin/env python
import os
import time
import numpy as np

import potential
import single_surface_model as ssm

t_start = time.time()

input_params = {}

#Potential energy model
input_params['model'] = potential.Morse()

#Basis function and basis set parameters
input_params['mass'] = 1.0              #Mass of system
input_params['x0'] = np.array([9.32])   #Initial coordinate for each dimension
input_params['k0'] = np.array([0.0])    #Initial momentum for each dimension
input_params['basis_function_width'] = np.array([0.5])  #alpha - basis function widht parameter
input_params['basis_set_size'] = 200    #Number of basis functions
input_params['add_cs'] = False #Boolean to make last basis function correspond to the initial state
#Called 'cs' for coherent state

#run parameters
input_params['integrator'] = 'RK4' #Select integrator, RK4 is only current option
input_params['time_step'] = 0.1    #ODE timestep
input_params['prop_time'] = 25.0   #Total propagation time
input_params['job_name'] = 'morse_ex' 
input_params['tcf_type'] = 2 #whether to evaluate trick where TCF 
#can be calculated at 2*t if initial wave packet has zero initial KE
input_params['wf_renorm'] = True



#Options for saving the wave function at various times
#At the selected time, the basis set is projected onto a grid of evenly spaced points
#Only valid for 1 and 2-dimensional cases
input_params['save_wf_times'] = np.array([4.0, 6.0, 8.0]) #Times at which to calculate the wave function
#Whether you store the calculated WFs in memory
input_params['store_wf'] = False 
#Whether they are written to text files
input_params['write_wf'] = True
#A boolean to save all steps.  I added this primarily for animations
input_params['store_all_wf'] = False

#Solution of QM coefficients
#The two options for matrix_type are pseudospectral or galerkin
#only pseudospectral is available for nonadiabatic calculations
input_params['matrix_type'] = 'pseudospectral' #Default

# input_params['solver'] = 'bicg' #The paper uses the lstsq, but I have been testing other methods
input_params['solver'] = 'lstsq'
input_params['svd_threshold'] = 1e-3

#Initialize ODE class
morse_ex = ssm.GWPD_SS_Model(**input_params)

#Generates the initial basis set, available as 'init_basis' attribute which is
#a class instance of Basis_function.  'prop_basis' is a copied instance that 
#is modified at each time
#This needs to be called before just about anything else
morse_ex.setup_basis()

morse_ex.propagate()


with open(ss_test.outfile, 'a') as f:
    f.write('Run time is : {:5.2f} seconds\n'.format(time.time() - t_start))

