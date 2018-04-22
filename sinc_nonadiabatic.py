import time
import copy
import os
import numpy as np
import scipy as sp

import potential


class Sinc_Nonadiabatic(object):
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
#        self.tstep = params[7]
#        self.potential_name = params[8]       
        
        prop_defaults = {
                'model': potential.Morse_two_surf(),
                'n_surface':2,
                'electronic_representation':'diabatic',
                'init_surface':1,
                'mass':1.0,
                'x0': np.array([0.0]),
                'k0': np.array([0.0]),
                'wavepacket_width': np.array([1.0]),
                'ndim':1,
                'grid_spacing':1e-1,
                'grid_lims':np.array([[-5., 15.]]),
                'grid_points':None,
                'time_step':5.0,
                'prop_time':10000.0,
                'save_wf_times':None,
                'save_all_wf':False,
                'job_name':'test',
                'tcf_type':1,
                'integrator':'exp',
                'store_wf':False
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
        assert self.ndim == 1, "Only supports 1-D right now"

        if self.grid_spacing:
            assert self.grid_points is None, "Conflicting arguments, grid_points and grid_spacing" 
            if self.ndim == 1:
                self.nbasis = 1 + int((self.grid_lims[0,1] - self.grid_lims[0,0]) / self.grid_spacing)
                self.nx = self.nbasis
                self.x_grid = np.linspace(*self.grid_lims[0], self.nbasis).reshape(-1,1)
                self.x_space = float(self.x_grid[1,0] - self.x_grid[0,0])
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
                self.x_space = float(self.x_grid[1,0] - self.x_grid[0,0])
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

        
        self.init_psi = np.zeros([self.nbasis, self.n_surface], dtype = np.complex)
        self.init_psi[:, self.init_surface] = (((np.prod(2*self.alpha)) / (np.pi)**self.ndim)**(1./4.) 
                        * np.exp((-self.alpha * (self.x_grid - self.x0)**2 
                          + 1.0j*self.k0*(self.x_grid - self.x0)).sum(axis=-1)))
               
        self.prop_psi = np.copy(self.init_psi)

        #self.x_grid.reshape(-1)
#        self.x_space = self.x_grid[1] - self.x_grid[0]
        self.tcf = np.zeros([0,self.n_surface + 1], dtype=np.complex)
        self.pops = np.zeros([0,self.n_surface + 1])
        self.nsteps = int((self.prop_time / self.time_step)) + 1
        
        if self.save_wf_times is not None:
            self.save_steps = self.save_wf_times / self.time_step
            self.wfs = {}
        else:
            self.save_steps = []
        if self.save_all_wf:
            self.save_steps = np.arange(self.nsteps)
            self.wfs = {}
        return
   
    def build_H(self):
        self.D1, self.D2 = self.generate_weights(self.x_grid.shape[0], self.x_space)
    
        if self.electronic_representation == 'diabatic':
            self.V, self.V1 = self.model.calc_V(self.x_grid)
            self.T  = -(1.0/(2.0*self.mass))*self.D2
            self.H = np.copy(self.T) #+ np.diag(self.V[:,0,0])
            for i in range(1, self.n_surface):
                self.H = sp.linalg.block_diag(self.H, self.T)# + np.diag(self.V[:,i,i]))
            #Fill in off-diagonal coupling.  Should be fast so will leave loops
            for i in range(self.n_surface):
                for j in range(self.n_surface):
                    #print('Potential energy', i, j, i*self.nbasis, j*self.nbasis, (i+1)*self.nbasis+1, (j+1)*self.nbasis+1)
                    #print(np.diag(self.V[:,i,j]).shape, self.V[:,i,j])

                    self.H[i*self.nbasis:(i+1)*self.nbasis, j*self.nbasis:(j+1)*self.nbasis] += np.diag(self.V[:,i,j])
        elif self.electronic_representation == 'adiabatic':
            #I don't doubt there are betters ways to do this, but this hopefully works
            
            V, V1 = self.model.calc_V(self.x_grid)
            Va  = V[:, 0, 0]
            Vb  = V[:, 1, 1]
            Vc  = V[:, 0, 1]
            #Gradients
            V1a = V1[:, 0, 0, :]
            V1b = V1[:, 1, 1, :]
            V1c = V1[:, 0, 1, :]

            W, W1 = potential.calc_W(Va, Vb, Vc, V1a, V1b, V1c)
            Wa = W[:,0]
            Wb = W[:,1]
            dab =  potential.calc_dab(Va, Vb, Vc, V1a, V1b, V1c)
            #dab =  potential.calc_dab(Va, Vb, Vc, V1a, V1b, V1c)
            
            self.Vaa = np.diag(Wa)
            self.Vbb = np.diag(Wb)
            self.Vab = np.zeros([self.nbasis, self.nbasis])
            self.Vba = np.zeros([self.nbasis, self.nbasis])
        
            #self.Vab = - np.dot(self.refD1, np.dot(np.diag(dab), self.refD1))/self.m
            #self.Vba = self.Vab.T
            #self.Vba = - np.dot(self.refD1, np.dot(np.diag(dba), self.refD1))/self.m
            
            ##for i in xrange(self.nbasis):
            ##    self.Vab[i, :] = - dab[i]*self.refD1[i, :]/self.m
            ##    self.Vba[i, :] = - dba[i]*self.refD1[i, :]/self.m
            for i in range(self.nbasis):
                for j in range(self.nbasis):
                    self.Vab[i, j] = - 0.5*(dab[i,0,1,0] + dab[j,0,1,0])*self.D1[i, j]/self.mass
                    self.Vba[i, j] = - 0.5*(dab[i, 1, 0, 0] + dab[j, 1, 0, 0])*self.D1[i, j]/self.mass
                
            self.T  = -(1.0/(2.0*self.mass))*self.D2
            self.Ha = self.T + self.Vaa
            self.Hb = self.T + self.Vbb
            self.H = np.concatenate((np.concatenate((self.Ha, self.Vab), 1), np.concatenate((self.Vba, self.Hb), 1)))
 
        if self.integrator  == "exp":
            #print('saving psi at time ', ttemp, step)
            self.Heigval, self.Heigvec = np.linalg.eig(self.H)
            self.expH = np.diag(np.exp(-1.0j*self.time_step*self.Heigval))
            self.propagator = np.dot(self.Heigvec, np.dot(self.expH, np.linalg.inv(self.Heigvec)))

        return

    def exp_step(self, i_step):
        ttemp = i_step*self.time_step
        if i_step in self.save_steps:
            self.wfs['{:.0f}'.format(i_step)] = np.c_[self.x_grid, self.prop_psi]
        pops = (self.x_space*np.absolute(self.prop_psi**2)).sum(axis=0)
        self.pops = np.vstack((self.pops, np.hstack((ttemp, pops))))
        if self.tcf_type == 1:
            tcf_val = self.x_space*(self.init_psi.conj()*self.prop_psi).sum(axis=0)
        elif self.tcf_type == 2:
            tcf_val = self.x_space*(self.prop_psi*self.prop_psi).sum(axis=0)
        self.tcf = np.vstack((self.tcf, np.hstack((ttemp, tcf_val))))
        #self.prop_psi = np.dot(self.propagator, self.prop_psi.reshape(-1)).reshape(-1,self.n_surface)
        #Need the Fortran ordering of the reshape to return expected shape
        self.prop_psi = np.dot(self.propagator, np.dstack(self.prop_psi).ravel()).reshape(-1,self.n_surface, order='F')
        #print(self.propagator.shape, np.dot(self.propagator, np.dstack(self.prop_psi).ravel()).reshape(-1,self.n_surface, order='F'))

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

    def propagate(self):

        if hasattr(self, 'propagator'):
            pass
        else:
            self.build_H()
   
        for i_step in range(self.nsteps):
            self.single_step(i_step)

    def single_step(self, i_step):

        if self.integrator == 'exp':
            self.exp_step(i_step)
        elif self.integrator == 'lanczos':
            self.lanczos_step(i_step)



