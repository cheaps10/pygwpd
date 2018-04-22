#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy as sp
import potential

"""
Created on Thu Apr 19 13:35:16 2018

@author: chad
"""
np.seterr(divide='ignore')
class Sinc_Single_Surface(object):
    """
    Sinc pseudospectral method for 1-D dynamics
    Cannot find my 2-D code and originally wrote it for just 1-D.


    """

    def __init__(self, **kwargs):
        """
        Use a dictionary to (debatably) simplify parameter specification.
        While the defaults are allowed, they are somewhat nonsensical.
        You will likely not want to run a free particle at the origin
        with no kinetic energy.
        """
#        self.mass = params[0]
#        self.xc = params[1]
#        self.k0 = params[2]
#        self.ts = params[3]
#        self.gridsize = params[4]
#        self.times = params[5]
#        self.alpha =  params[6]
#        self.time_step = params[7]
#        self.potential_name = params[8]       
        
        prop_defaults = {
                'model': potential.FreeParticle(),
                'mass':1.0,
                'x0': np.array([0.0]),
                'k0': np.array([0.0]),
                'wavepacket_width': np.array([1.0]),
                'ndim':1,
                'grid_spacing':1e-1,
                'grid_lims':np.array([[-1., 1.]]),
                'grid_points':None,
                'time_step':1.0,
                'prop_time':10.0,
                'save_wf_times':None,
                'job_name':'test',
                'tcf_type':1,
                'integrator':'exp',
                'save_all_wf':False,
                'lanczos_thresh':1e-6,
                'lanczos_dim':25,
                }
       
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default)) 
        
        self.outfile = self.job_name + '.out'
        if os.path.isfile(self.outfile):
            os.remove(self.outfile)

        for (prop, new) in kwargs.items():
            if prop not in prop_defaults.keys():
                with open(self.outfile, 'a') as f:
                    f.write('Warning, adding non-default attribute'
                            ' at initialization\n')
                    f.write('Attribute is {}\n'.format(prop))
                    
                setattr(self, prop, new) 
        self.alpha = self.wavepacket_width
        assert self.ndim == self.grid_lims.shape[0], "Grid limits and dimensions do not match!"    
#        assert self.ndim == 1, "Only supports 1-D right now"

        if self.grid_spacing:
            assert self.grid_points is None, "Conflicting arguments, grid_points and grid_spacing" 
            if self.ndim == 1:
                self.nbasis = 1 + int((self.grid_lims[0,1] - self.grid_lims[0,0]) / self.grid_spacing)
                self.nx = self.nbasis
                self.x_grid = np.linspace(*self.grid_lims[0], self.nbasis).reshape(-1,1)
                self.x_space = self.x_grid[1,0] - self.x_grid[0,0]
                self.y_space = 1.0
            elif self.ndim == 2:
                if hasattr(self.grid_spacing, '__iter__'):
                    self.nx = 1 + int((self.grid_lims[0,1] - self.grid_lims[0,0]) / self.grid_spacing[0])
                    x_temp = np.linspace(*self.grid_lims[0], self.nx)
                    self.ny = 1 + int((self.grid_lims[1,1] - self.grid_lims[1,0]) / self.grid_spacing[1])
                    y_temp = np.linspace(*self.grid_lims[1], self.ny)
                    self.x_space = x_temp[1] - x_temp[0]
                    self.y_space = y_temp[1] - y_temp[0]
                else:
                    self.nx = 1 + int((self.grid_lims[0,1] - self.grid_lims[0,0]) / self.grid_spacing)
                    x_temp = np.linspace(*self.grid_lims[0], self.nx)
                    self.ny = 1+ int((self.grid_lims[1,1] - self.grid_lims[1,0]) / self.grid_spacing)
                    y_temp = np.linspace(*self.grid_lims[1], self.ny)
                    self.x_space = x_temp[1] - x_temp[0]
                    self.y_space = y_temp[1] - y_temp[0]
                xx, yy = np.meshgrid(x_temp, y_temp)
                self.nbasis = xx.size
                self.x_grid = np.c_[xx.reshape(self.nbasis), yy.reshape(self.nbasis)]

        if self.grid_points:
            assert self.grid_spacing is None, "Conflicting arguments, grid_points and grid_spacing" 
            if self.ndim == 1:
                self.nbasis = self.grid_points
                self.x_grid = np.linspace(*self.grid_lims[0], self.nbasis).reshape(-1,1)
                self.x_space = self.x_grid[1,0] - self.x_grid[0,0]
                self.y_space = 1.0
            elif self.ndim == 2:
                if hasattr(self.grid_spacing, '__iter__'):
                    x_temp = np.linspace(*self.grid_lims[0], self.grid_points[0])
                    y_temp = np.linspace(*self.grid_lims[1], self.grid_points[1])
                else:
                    x_temp = np.linspace(*self.grid_lims[0], self.grid_points)
                    y_temp = np.linspace(*self.grid_lims[1], self.grid_points)
                xx, yy = np.meshgrid(x_temp, y_temp)
                self.nbasis = xx.size
                self.x_grid = np.c_[xx.reshape(self.nbasis), yy.reshape(self.nbasis)]
                
        if self.ndim == 2:
            assert self.nx == self.ny, ('The code currently requires an equal number of points in '
                                         'both dimensions, sorry!')

#        self.init_psi = ((np.pi**self.ndim / np.prod(self.alpha))**(1./4.) 
#                        * np.exp((-self.alpha * (self.x_grid - self.x0)**2 
#                          + 1.0j*self.k0*(self.x_grid - self.x0))))
        
        self.init_psi = (((np.prod(2*self.alpha)) / (np.pi)**self.ndim)**(1./4.) 
                        * np.exp((-self.alpha * (self.x_grid - self.x0)**2 
                          + 1.0j*self.k0*(self.x_grid - self.x0)).sum(axis=-1)))
        self.prop_psi = np.copy(self.init_psi)

#        self.x_grid.reshape(-1)

        self.tcf = np.zeros([0, 3], dtype=np.complex)
        self.nsteps = int((self.prop_time / self.time_step)) + 1
        
        if self.save_all_wf:
            self.save_steps = np.arange(self.nsteps)
            self.wfs = {}
        
        elif self.save_wf_times is not None:
            self.save_steps = self.save_wf_times / self.time_step
            self.wfs = {}
        else:
            self.save_steps = []
        
        return
   
    def build_H(self):
        
        if self.ndim == 1:
            D1, D2 = self.generate_weights(self.x_grid.shape[0], self.x_space)
            self.V = np.diag(self.model.calc_V(self.x_grid))
            self.T = -(1.0/(2.0*self.mass))*D2
            self.H = self.T + self.V

        if self.ndim == 2:
            self.V = np.diag(self.model.calc_V(self.x_grid))
            """
            Sinc pseudospectral kinetic energy for multiple dimensions
            Given a matrix element T_ij where i -> (x_1, y_2)
            and j -> (x_3, y_4), we get the total T_ij as the sum of
            cartesian components where they only contribute if the other
            axes are equal. Typically written with delta functions, but
            from a coding perspective, it looks like this
            
            T_ij = 0
            if x_1 == x_3:
                T_ij += KE_y
            if y_2 = y_4:
                T_ij += KE_x
            """
            D1x, self.D2x = self.generate_weights(self.nx, self.x_space)
            D1y, self.D2y = self.generate_weights(self.ny, self.y_space)
            #self.D2x = self.D2x.reshape(-1)
            #self.D2y = self.D2y.reshape(-1)
            
            
            x_bool = self.x_grid[None, :, 0] == self.x_grid[:, None, 0]
            y_bool = self.x_grid[None, :, 1] == self.x_grid[:, None, 1]
            #x_dx = (self.D2x[None, :] + self.D2x[:,None]) / 2.0
            #y_dy = (self.D2y[None, :] + self.D2y[:,None]) / 2.0
            D2x_exp = np.tile(self.D2x, [self.ny, self.ny])
            D2y_exp = np.tile(self.D2y[:,None,:, None], 
                                 [self.nx,self.nx]).reshape(self.nbasis, self.nbasis)
            #x = np.unique(self.x_grid[:,0]) 
            #y = np.unique(self.x_grid[:,1]) 
            #xx = [(float(i), float(j)) for i in x for j in x]
            #yy = [(float(i), float(j)) for i in y for j in y]

            #self.T = np.zeros([self.nbasis, self.nbasis])
            #for i in range(self.nbasis):
            #    for j in range(self.nbasis):
            #        #if self.x_grid[i,0] == self.x_grid[j,0]:
            #        if x_bool[i,j]: 
            #            self.T[i,j] += self.D2y[yy.index((self.x_grid[i,1], self.x_grid[j,1]))]
            #        #if xy[i,1] == xy[j,1]:
            #        if y_bool[i, j]:
            #            self.T[i,j] += self.D2x[xx.index((self.x_grid[i,0], self.x_grid[j,0]))]

            self.T = -(1.0/(2.0*self.mass)) * (y_bool * D2x_exp + x_bool * D2y_exp)
            #self.T *= -(1.0/(2.0*self.mass))# * (y_bool*self.D2x + x_bool*self.D2y)
            #self.T =  (y_bool*x_dx + x_bool*y_dy)
            #self.T = np.zeros([self.nbasis, self.nbasis])
            #self.T = -(1.0/(2.0*self.mass)) * (self.D2x[:,None] + self.D2y[None, :])#(self.D2x[None,:] + self.D2y[:,None])
            self.H = self.T + self.V
            print('Construction of Hamiltonian complete')

        if self.integrator  == "exp":
            #print('saving psi at time ', ttemp, step)
            self.Heigval, self.Heigvec = np.linalg.eig(self.H)
            self.expH = np.diag(np.exp(-1.0j*self.time_step*self.Heigval))
            self.propagator = np.dot(self.Heigvec, np.dot(self.expH, np.linalg.inv(self.Heigvec)))

        elif self.integrator == 'lanczos':
            self.c_lanc    = np.zeros([self.lanczos_dim], dtype = np.complex128)  #Coefficient vector in Lanczos basis
            self.c_lanc[0] = 1.0 + 0.0j
            self.propagator, self.Atrans = self.build_lanczos_H(np.copy(self.init_psi))
        print('Construction of propagator is complete')
        return
    
    def propagate(self):

        if hasattr(self, 'propagator'):
            pass
        else:
            self.build_H()
   
        for i_step in range(self.nsteps):
            self.single_step(i_step)

            
        return

    def single_step(self, i_step):

        if self.integrator == 'exp':
            self.exp_step(i_step)
        elif self.integrator == 'lanczos':
            self.lanczos_step(i_step)


    def exp_step(self, i_step):
        ttemp = i_step*self.time_step
        if i_step in self.save_steps:
            self.wfs['{:.0f}'.format(i_step)] = np.c_[self.x_grid, self.prop_psi]
        wf_norm = (self.y_space*self.x_space*np.absolute(self.prop_psi)**2).sum()
        if self.tcf_type == 1:
            tcf_val = self.y_space*self.x_space*(self.init_psi.conj()*self.prop_psi).sum()
        elif self.tcf_type == 2:
            tcf_val = self.y_space*self.x_space*(self.prop_psi*self.prop_psi).sum()
#        print(tcf_val)
        self.tcf = np.vstack((self.tcf, np.array([self.tcf_type * ttemp, tcf_val, wf_norm])))
        self.prop_psi = np.dot(self.propagator, self.prop_psi) 
        return
    
    def generate_weights(self, nx, x_space):
        D1 = np.zeros([nx, nx])
        D2 = np.zeros([nx, nx])
        d2diag = np.zeros([nx])
        d2diag.fill(-np.pi**2/(3*x_space**2))

        kmat = np.arange(nx)[None,:] - np.arange(nx)[:,None]
        D1 = (-1.0)**(kmat+1)/(x_space*kmat)
        D2 = (2.0*(-1.0)**(kmat+1))/(x_space**2*kmat**2)
        D1[np.isinf(D1)] = 0.0
        D2[np.isinf(D2)] = 0.0
        D2 = D2 + np.diag(d2diag)
        return D1, D2
    

    def lanczos_step(self, i_step):

        ttemp = i_step*self.time_step
        if i_step in self.save_steps:
            self.wfs['{:.0f}'.format(i_step)] = np.c_[self.x_grid, self.prop_psi]
        wf_norm = (self.y_space*self.x_space*np.absolute(self.prop_psi)**2).sum()
        if self.tcf_type == 1:
            tcf_val = self.y_space*self.x_space*(self.init_psi.conj()*self.prop_psi).sum()
        elif self.tcf_type == 2:
            tcf_val = self.y_space*self.x_space*(self.prop_psi*self.prop_psi).sum()
        self.tcf = np.vstack((self.tcf, np.array([self.tcf_type * ttemp, tcf_val, wf_norm])))


        if np.absolute(self.c_lanc[self.lanczos_dim -1])**2 > self.lanczos_thresh:
            #print("regenerating Lanczos Hamiltonian at step {}".format(i_step))
            self.propagator, self.Atrans = self.build_lanczos_H(np.copy(self.prop_psi))
            self.c_lanc    = np.zeros([self.lanczos_dim], 
                                        dtype = np.complex128)  #Coefficient vector in Lanczos basis
            self.c_lanc[0] = 1.0 + 0.0j
        
        
        self.c_lanc = np.dot(self.propagator, self.c_lanc)
        self.prop_psi = np.dot(self.Atrans, self.c_lanc)
    
        return


    def build_lanczos_H(self, a0):
        
        #print("Building new reduced Lanczos Hamiltonian")

        H_lanc  = np.zeros([self.lanczos_dim, 
                            self.lanczos_dim], 
                            dtype = np.complex128)    #Reduced Hamiltonian in Lanczos basis
        q_vec   = np.zeros([self.nbasis, 
                            self.lanczos_dim], 
                            dtype = np.complex128)    #non-orthogonal Krylov vectors
        k_vec   = np.zeros([self.nbasis, 
                            self.lanczos_dim], 
                            dtype = np.complex128)    #orthogonal Krylov vectors
        a_vec   = np.zeros([self.lanczos_dim], 
                            dtype = np.complex128)    #Inner products required to build H_lanc
        b_vec   = np.zeros([self.lanczos_dim-1],
                            dtype = np.complex128)    #Inner products required to build self.H_lanc
        c_lanc  = np.zeros([self.lanczos_dim],
                            dtype = np.complex128)    #Coefficient vector in Lanczos basis
       

        ##Calculate the iterative applications of the Hamiltonian to the initial state
        q_vec[:, 0] = a0#/(np.sqrt(np.sum(a0.conj()*a0)))
        k_vec[:, 0] = a0#/(np.sqrt(np.sum(a0.conj()*a0)))
        a_vec[0] = (self.y_space * self.x_space
                  * np.dot(k_vec[:, 0].T.conj(), np.dot(self.H, k_vec[:, 0])))
        q_vec[:, 1] = np.dot(self.H, k_vec[:, 0]) - a_vec[0]*k_vec[:, 0]
        b_vec[0] = np.sqrt(self.y_space * self.x_space 
                           * np.dot(q_vec[:, 1].T.conj(), q_vec[:, 1]))
        k_vec[:, 1] = (1./b_vec[0])*q_vec[:, 1]
        a_vec[1] = (self.y_space * self.x_space
                  * np.dot(k_vec[:, 1].T.conj(), np.dot(self.H, k_vec[:, 1])))
        
        #q_vec[:, 2] = np.dot(self.H, k_vec[:, 1]) - a_vec[1]*k_vec[:, 1] - b_vec[0]*k_vec[:, 0]

        for i in range(2, self.lanczos_dim):
            #q_vec[:, 2] = np.dot(self.H, k_vec[:, 1]) - a_vec[1]*k_vec[:, 1] - b_vec[0]*k_vec[:, 0]
            q_vec[:, i] = np.dot(self.H, k_vec[:, i-1]) - a_vec[i-1]*k_vec[:, i-1] - b_vec[i-2]*k_vec[:, i-2]
            b_vec[i-1] = np.sqrt(self.y_space * self.x_space 
                                * np.dot(q_vec[:, i].T.conj(), q_vec[:, i]))
            k_vec[:, i] = (1./b_vec[i-1])*q_vec[:, i]
            a_vec[i] = (self.y_space * self.x_space
                      * np.dot(k_vec[:, i].T.conj(), np.dot(self.H, k_vec[:, i])))
        
        ##General tridagonal where the three elements for row N are are 
        ##beta[N-1], alpha[N], beta[N]
        H_lanc[0, 0] = a_vec[0] 
        H_lanc[0, 1] = b_vec[0] 

        H_lanc[self.lanczos_dim -1, self.lanczos_dim - 1] = a_vec[self.lanczos_dim - 1] 
        H_lanc[self.lanczos_dim -1, self.lanczos_dim - 2] = b_vec[self.lanczos_dim - 2] 
        for i in range(1,self.lanczos_dim - 1):
            H_lanc[i, i - 1] = b_vec[i - 1]
            H_lanc[i, i]     = a_vec[i]
            H_lanc[i, i + 1] = b_vec[i]
        #print("is H_lanc Hermitian? {}".format(np.allclose(H_lanc.T.conj(), H_lanc)))
        Heigval, Heigvec = np.linalg.eig(H_lanc)
        expH = np.diag(np.exp(-1.0j*self.time_step*Heigval))
        #print"expH {}".format(expH)
        newH = np.dot(Heigvec, np.dot(expH, np.linalg.inv(Heigvec)))
        return newH, k_vec








