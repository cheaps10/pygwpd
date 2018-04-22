import numpy as np
from pygwpd import nonadiabatic_model

tully1_params = {}


tully1_params['model'] = potential.Tully1()

#Basis function and basis set parameters
tully1_params['mass'] = 2000.0          
tully1_params['x0'] = np.array([-5.0])  
tully1_params['k0'] = np.array([20.0])  
tully1_params['basis_function_width'] = np.array([1.0]) 
tully1_params['basis_set_size'] = 200    
tully1_params['add_cs'] = False 

#Nonadiabatic options
tully1_params['basis_velocity'] = 'ehrenfest'
tully1_params['electronic_rep'] = 'adiabatic'
tully1_params['init_surface'] = 0

#run parameters
tully1_params['integrator'] = 'RK4'
tully1_params['time_step'] = 1.0   
tully1_params['prop_time'] = 1000.0   #Total propagation time
tully1_params['save_wf_times'] = None
tully1_params['job_name'] = 'tully1_ex'
tully1_params['tcf_type'] = 1 
tully1_params['store_wf'] = False 
tully1_params['wf_renorm'] =  False

#Solution of QM coefficients
tully1_params['matrix_type'] = 'pseudospectral'
tully1_params['solver'] = 'lstsq'
tully1_params['svd_threshold'] = 5e-3

#Initialize ODE class
tully1_ex = nonadiabatic_model.GWPD_Nonadiabatic_Model(**tully1_params)

tully1_ex.setup_basis()
tully1_ex.propagate()
