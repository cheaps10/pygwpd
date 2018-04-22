import time
import copy
import os
import numpy as np
import scipy.sparse.linalg as spla

import potential
import basis_function as bf

############################################################
#                        Chad Heaps                        #
#                        02-08-2016                        #
#                                                          #
#       Program to run time-dependent Schrodinger          #
#       equation simulations using classical trajectory    #
#       guided Gaussian basis sets                         #
#                                                          #
#                                                          #      
#                                                          #
############################################################


class GWPD_SS_Model(object):
    """
    Covers dynamics on a single, analytical
    potential energy surface

    May use pseudospectral or Galerkin methods


    """

    def __init__(self, **kwargs):
        """
        Use a dictionary to (debatably) simplify parameter specification.
        While the defaults are allowed, they are somewhat nonsensical.
        You will likely not want to run a free particle at the origin
        with no kinetic energy.
        """
        prop_defaults = {
                'model': potential.FreeParticle(),
                'mass':1.0,
                'x0': np.array([0.0]),
                'k0': np.array([0.0]),
                'basis_function_width': np.array([1.0]),
                'basis_set_size':10,
                'add_cs':False,
                'time_step':1.0,
                'prop_time':10.0,
                'save_wf_times':None,
                'store_wf':False,
                'write_wf':False,
                'save_all_wf':False,
                'wf_grid_space':0.1,
                'job_name':'test',
                'tcf_type':1,
                'matrix_type':'pseudospectral',
                'solver':'lstsq',
                'svd_threshold':1e-3,
                'integrator':'RK4',
                'basis_compression':1.0,
                'basis_backup_file':None,
                'hh_cutoff':12.0,
                'galerkin_approx':'bat',
                'wf_renorm':False,
                }

        #This loop comes from
        #https://stackoverflow.com/questions/5899185/class-with-too-many-parameters-better-design-strategy
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

        self.nsteps = int((self.prop_time / self.time_step)) + 1
        self.ndim = self.x0.size
        self.nbasis = self.basis_set_size
        #Initialize time correlation function
        self.tcf = np.zeros([0, 3], dtype=np.complex)
        if self.save_wf_times is not None:
            self.save_steps = self.save_wf_times / self.time_step
        else:
            self.save_steps = []

        if self.save_all_wf:
            self.save_steps = np.arange(self.nsteps)
        return

####Set attributes
##    def set_is_propagated(self):
##        self.is_propagated = True
        return

    def setup_basis(self):
        """
        Function of the GWPD_Prop_SS class to generate
        the necessary Basis_function instance in order to
        start a calculation
        
        Input:
            None.  Uses information from the initialization

        Output:
            init_basis: An instance of the Basis_function class
            used for propagation

        Notes:
        2.  The initial position and momenta are saved in a 
            file with end _init_cond.  It can be read back in
            by changing the name of the file to init_cond.
            If you change basis set size, however, it will crash
        3.  The calculation of the initial coefficients
            is performed by projecting the non-orthogonal basis
            onto the initial state...using appropriate test 
            functions for the methods
        
        """

        ig = np.zeros([self.nbasis], dtype=np.complex128)
        ialpha = np.zeros([self.nbasis, self.ndim])
        ix = np.zeros([self.nbasis, self.ndim])
        ik = np.zeros([self.nbasis, self.ndim])

        #if os.path.isfile(job_name + "_init_cond"):
        #    ic = np.loadtxt(job_name + "_init_cond")
        if self.basis_backup_file:
            try:
                with open(self.outfile, 'a+') as f:
                    f.write('Reading basis set file\n')
                ic = np.loadtxt(self.basis_backup_file)
                ix = ic[0:self.nbasis, 0:self.ndim]
                ik = ic[0:self.nbasis, self.ndim:2*self.ndim]
            except:
                print('Basis set file specified not present!')
        else:
            for i in range(self.ndim): 
                ix[:, i] = np.random.normal(self.x0[i], 
                                            self.basis_compression*np.sqrt(1.0/self.basis_function_width[i]),
                                            self.nbasis)
                ik[:, i] = np.random.normal(self.k0[i],
                                            self.basis_compression*np.sqrt(self.basis_function_width[i]), 
                                            self.nbasis)
            if self.add_cs: 
                ix[-1,:] = self.x0
                ik[-1,:] = self.k0

        ig1 = np.sum(0.25*np.log((2.0*self.basis_function_width)/np.pi))
        ig.fill(ig1)
        for i in range(self.nbasis):
            ialpha[i, :] = self.basis_function_width
            #basis_save = np.concatenate((ix, ik), 1)
        np.savetxt(self.job_name + ".init_cond", np.c_[ix, ik])

        init_basis = bf.Basis_function(ialpha, ix, ik, ig, 
                                    np.zeros([self.nbasis], dtype=np.complex128))
        I0 = init_basis.calc_overlap()
        target_func = bf.Basis_function(self.basis_function_width, self.x0, self.k0, 
                                    np.sum(0.25*np.log((2.0*self.basis_function_width)/np.pi)),
                                    0.0 + 0.0j) 
        if self.matrix_type == 'galerkin':
            ##Create a second basis set with the initial state.  Then, calculate
            #the overlap and use the last row as the initial expansion coefficients
            target_basis = copy.deepcopy(init_basis)
            target_basis.a = np.vstack((target_basis.a, self.basis_function_width))
            target_basis.p = np.vstack((target_basis.p, self.k0))
            target_basis.xc = np.vstack((target_basis.xc, self.x0))
            target_basis.g = np.append(target_basis.g, target_func.g)

            target_overlap = target_basis.calc_overlap()
            overlap_inv = np.linalg.pinv(I0, rcond = 1e-8)
            init_a = target_overlap[:self.nbasis, self.nbasis]
            init_basis.c = np.dot(overlap_inv, init_a).reshape(-1,1)

        elif self.matrix_type == "pseudospectral": 
            disp = init_basis.xc[:, None, :] - init_basis.xc[None,:,:]
            a_temp = init_basis.a
            x_temp = init_basis.xc
            p_temp = init_basis.p

            init_psi_basis = np.exp((-a_temp[None,:,:]*disp**2 + 
                                        1.0j*p_temp[None,:,:]*disp).sum(axis=-1) 
                                        + init_basis.g[None, :])
            ex_disp = ix - target_func.xc[:]
            init_psi_exact = np.exp((-target_func.a*ex_disp**2 
                                    + 1.0j*target_func.p*ex_disp).sum(axis=-1)
                                    + target_func.g)
            psi_basis_inv = np.linalg.pinv(init_psi_basis, rcond = 1e-8)

            init_basis.c = np.dot(psi_basis_inv, init_psi_exact).reshape(-1,1)
        
        check_norm = np.dot(init_basis.c[:,0].T.conj(), np.dot(I0, init_basis.c[:,0]))
        #check_norm = np.dot(init_basis.c.T.conj(), np.dot(I0, init_basis.c))
        with open(self.outfile, 'a+') as f:
            f.write('Initial WF norm {:4.2e}\n'.format(check_norm))
        self.init_basis = init_basis
        self.prop_basis = copy.deepcopy(init_basis)
        self.dC0 = np.zeros([self.nbasis], dtype=np.complex)

        return

    

    def propagate(self):
        """
        Perform time-stepping loop for your favorite
        ODE solver.  Right now RK4 is the only coded solver

        """

        #Since we're using a fixed time step integration
        #We can just calculate how many time steps we need
        #to integrate rather than worrying about a variable
        #number of steps to reach prop_time

        self.dC0 = np.zeros([self.nbasis], dtype=np.complex)

        if self.integrator == 'RK4' or self.integrator == 'rk4':
            for step in range(self.nsteps):
                self.rk4_single_step(step)
                #if isinstance(dC0, int):
                #    break
#        if self.ndim <= 2:
#            self.prop_basis.plot_psi(self.job_name, nsteps, store_data = self.store_wf)
        return

    def rk_prep(self, deriv, tdat):
        """
        Converts the basis set at time t into the intermediate time steps of
        an integrator with differential quantities deriv and basis set tdat
        """

        temp_basis = copy.deepcopy(tdat)
        temp_basis.xc += deriv['xcdt']
        temp_basis.p += deriv['pdt']
        temp_basis.g += deriv['gdt']
        temp_basis.c += deriv['cdt']
        return temp_basis

    def rk4_single_step(self, step):
        """
        Time-step using RK4
        
            y_n+1 = y_n + h/6(k1 + 2*k2 + 2*k3 + 2*k4)
            t_n+1 = t_n + h

            4th order:
            k1 = f(t_n, y_n)
            k2 = f(t_n + h/2, y_n + h/2*k1)
            k3 = f(t_n + h/2, y_n + h/2*k2)
            k4 = f(t_n + h, h*k3)

        """
        ttemp = step*self.time_step
        dC0 = self.dC0
        ##Save wavefunction at times specified in save_wf_times
        if (step in self.save_steps) & (self.ndim < 3):
            self.prop_basis.plot_psi(self.job_name, step, 
                                    write_data = self.write_wf,
                                    store_data = self.store_wf,
                                    xspace = self.wf_grid_space)
        #If statement to check for bad trajectories and delete them from the basis set for Henon-Heiles models
        if 'HH' in self.model.__class__.__name__:# self.potential_name == "HH2dMCTDH" or self.potential_name == "HH6d" or self.potential_name == "HH4d" or self.potential_name == "HH10d":
            bad_vals = np.less(self.hh_cutoff, np.max(np.absolute(self.prop_basis.xc), axis=1))
            if np.any(bad_vals):
                bad_index = np.where(bad_vals)
                self.prop_basis.a  = np.delete(self.prop_basis.a,  bad_index, axis=0)
                self.prop_basis.p  = np.delete(self.prop_basis.p,  bad_index, axis=0)
                self.prop_basis.xc = np.delete(self.prop_basis.xc, bad_index, axis=0)
                self.prop_basis.g  = np.delete(self.prop_basis.g,  bad_index)
                self.prop_basis.c  = np.delete(self.prop_basis.c,  bad_index, axis=0)
                dC0 = np.delete(dC0, bad_index, axis=0)
                with open(self.outfile, 'a+') as f:
                    f.write('Basis set size at time {:5.3f} = {:4.0f}\n'.format(ttemp, self.prop_basis.c.size)) 

        ##Save TCF and normalization data
        I0 = self.prop_basis.calc_overlap()
        wf_norm = np.dot(self.prop_basis.c[:,0].conj(), np.dot(I0, self.prop_basis.c[:,0]))
        tcf_val = self.prop_basis.calc_tcf(self.init_basis, self.tcf_type) 
        self.tcf = np.vstack((self.tcf, np.array([self.tcf_type * ttemp, tcf_val, wf_norm])))
        #if self.tcf_type == 1:
        #    self.tcf = np.vstack((self.tcf, np.array([ttemp, tcf_val, wf_norm])))
        #elif self.tcf_type == 2:
        #    self.tcf = np.vstack((self.tcf, np.array([2.*ttemp, tcf_val, wf_norm])))
        #Write as time, Re(TCF), Im(TCF), Abs(TCF), |Psi|^2
        np.savetxt(self.job_name + ".tcf_" + str(self.prop_time), 
                   np.c_[self.tcf[:, 0].real, 
                         self.tcf[:, 1].real, 
                         self.tcf[:, 1].imag, 
                         np.absolute(self.tcf[:, 1]), 
                         self.tcf[:, 2].real])
                   #np.c_[self.tcf.view(float), np.absolute(self.tcf[:,1]])
        if self.wf_renorm:
            self.prop_basis.c/=np.sqrt(wf_norm)

        if wf_norm > 10.0:
            with open(self.outfile, 'a+') as f:
                f.write('Warning!! Norm is very large, at time = {:4.2f} exiting. {: 5.2}\n'.format(ttemp, wf_norm))
            return -1


        datk1 = self.calc_differential(self.prop_basis, dC0)
        datk2 = self.calc_differential(self.rk_prep({i:j*(self.time_step/2.0) 
                                                        for i,j in datk1.items()},    
                                                    self.prop_basis), 
                                            datk1['cdt'])
        datk3 = self.calc_differential(self.rk_prep({i:j*(self.time_step/2.0) 
                                                        for i,j in datk2.items()},    
                                                    self.prop_basis),
                                            datk2['cdt'])
        datk4 = self.calc_differential(self.rk_prep({i:j*(self.time_step) 
                                                        for i,j in datk3.items()},    
                                                    self.prop_basis),
                                            datk3['cdt'] )
        
        #Update all of the basis function parameters
        self.prop_basis.xc += (self.time_step/6.0)*(datk1['xcdt'] 
                                   + 2.0*datk2['xcdt'] 
                                   + 2.0*datk3['xcdt'] 
                                       + datk4['xcdt']) 
        self.prop_basis.p += (self.time_step/6.0)*(datk1['pdt'] 
                                  + 2.0*datk2['pdt'] 
                                  + 2.0*datk3['pdt'] 
                                      + datk4['pdt']) 
        self.prop_basis.g += (self.time_step/6.0)*(datk1['gdt'] 
                                  + 2.0*datk2['gdt'] 
                                  + 2.0*datk3['gdt'] 
                                      + datk4['gdt']) 
        self.prop_basis.c += (self.time_step/6.0)*(datk1['cdt'] 
                                  + 2.0*datk2['cdt'] 
                                  + 2.0*datk3['cdt'] 
                                      + datk4['cdt']) 
        self.dC0 = copy.deepcopy(datk4['cdt'])
        
        return

    def calc_differential(self, temp_basis, dC0):# kbasis, basis, ik, system, tstep):
        """
            Calculates the time-derivative quantities of
            the basis functions and QM coefficients 
            for the appropriate runge-kutta step
       

            Input:
               temp_basis: An instance of Basis_function updated for the appropriate
                            RK-step
            Output:
                time derivatives of all quantities
                

        """

        diffs = {}
        ##Time derivatives of basis set using classic eqs of motion
        V = self.model.calc_V(temp_basis.xc)
        V1  = self.model.calc_V1(temp_basis.xc)
        diffs['xcdt'] = temp_basis.p/self.mass
        diffs['pdt'] = -V1
        diffs['gdt']  = -1.0j*(np.sum(-temp_basis.p**2/(2.0*self.mass) 
                                + temp_basis.a/self.mass, axis=1)
                                + V)
        #Time dependence of width parameter is long deprecated after
        #many numerical issues with thawed Gaussians
        #adt = np.zeros([self.nbasis, self.ndim])
        
        #Call appropriate QM method
        if self.matrix_type == "pseudospectral":
            diffs['cdt'] = self.calc_dC_pseudospectral(temp_basis, V, diffs, dC0)
        elif self.matrix_type == "galerkin":
            diffs['cdt'] = self.calc_dC_galerkin(temp_basis, diffs, dC0)
        
        return diffs

    def calc_dC_pseudospectral(self, gbasis, V, diffs, dC0):
        """
        Calculate the differential change in the C coefficients
        using the pseudospectral method

        The operations in this function rely heavily on NumPy broadcasting.  I tested them years ago
        and they should be correct.  As is the nature with broadcasting, they lead to large
        in-memory arrays for larger basis sets and more dimensions.  If I continued with this project,
        I would port this function to C/C++.

        Input:
            gbasis: Instance of Basis_function containing the basis set representing a particular RK step

        Output:
            dC: Nbasis long vector of time-derivatives of coefficients
            
        """
        tempC = gbasis.c
        nfunc = tempC.size
       

        xcdt = diffs['xcdt'] 
        pdt = diffs['pdt'] 
        gdt = diffs['gdt'] 
        ##Calculate displacements
        disp = gbasis.xc[:, None, :] - gbasis.xc[None,:,:]
        ##psi_basis is overlap or collocation matrix
        psi_basis = np.exp((-gbasis.a[None,:,:]*disp**2 + 1.0j*gbasis.p[None,:,:]*disp).sum(axis=-1) + gbasis.g[None, :])
        #Kinetic energy
        H = ((-1./(2.*self.mass))*(-2.*gbasis.a[None, :, :] 
            + (-2.*gbasis.a[None, :, :]*(disp) + 1.0j*gbasis.p[None, :, :])**2)).sum(axis=-1)
        
        #time derivative of the overlap matrix
        psi_tderiv = (2.0*gbasis.a[None, :, :]*(disp)*xcdt[None, :, :]
                             + 1.0j*pdt[None, :, :]*(disp)
                             - 1.0j*gbasis.p[None, :, :]*xcdt[None, :, :]).sum(axis=-1)
       
        psi_tderiv[:, :] += gdt[None, :]
        
        #Build Hamiltonian a
        H[:, :] += V[:, None]
        H[:, :] *= psi_basis[:, :]
        psi_tderiv[:, :] *= psi_basis[:, :]

        #Construct normal equations for TDSE
        #in non-orthogonal time-dependent basis
        #solve TDSE
        #Ax=b
        #b = A.T.conj() * -i*(H - i*dPsi/dt)*c
        Bvec = np.dot(psi_basis.T.conj(),
                      -1.0j*np.dot(H - 1.0j*psi_tderiv, 
                                  gbasis.c[:,0]))
        aA = np.dot(psi_basis.T.conj(), psi_basis)
        if self.solver == 'lstsq':
            dC = np.linalg.lstsq(aA, Bvec, rcond=self.svd_threshold)[0]
        elif self.solver == 'bicg':
            spdc = spla.bicg(aA, Bvec, x0=dC0, maxiter=1000, tol=5e-6)
            #spdc = spla.lgmres(aA, Bvec, x0=dC0, tol=5e-7, maxiter=1000)
            dC = spdc[0] 
            if spdc[1] != 0:
                with open(self.outfile, 'a+') as f:
                    f.write('Warning!! conjugate gradient solver did not converge for dC\n')
                    f.write('Solver output is {}\n'.format(spdc[1:]))

        return dC.reshape(-1,1)

    def calc_dC_galerkin(self, gbasis, diffs, dC0):
        """
        Calculate the differential change in the C coefficients
        using the spectral method 

        Input:
            gbasis: Instance of Basis_function containing the basis set representing a particular RK step

        Output:
            dC: Nbasis long vector of time-derivatives of coefficients
            
        """

        #Calculate the time derivative of the overlap matrix
        I0, H, overlap_dt = gbasis.calc_H_and_odt(self.model, diffs, 
                                                self.galerkin_approx, self.mass)

        #Construct normal equations for TDSE
        #in non-orthogonal time-dependent basis
        
        aA = np.dot(I0.T.conj(), I0)
        Bvec = -1.0j*np.dot(I0.T.conj(), 
                        np.dot(H - 1.0j*overlap_dt, gbasis.c[:,0]))
        if self.solver == 'lstsq':
            dC = np.linalg.lstsq(aA, Bvec, rcond=self.svd_threshold)[0]
        elif self.solver == 'bicg':
            spdc = spla.bicg(aA, Bvec, x0=dC0, maxiter=100000)
            dC = spdc[0] 
            if spdc[1] != 0:
                with open(self.outfile, 'a+') as f:
                    f.write('Warning!! conjugate gradient solver did not converge for dC\n')
                    f.write('Solver output is {}\n'.format(spdc[1:]))
    
        return dC.reshape(-1,1)





#class RK4(GWPD_SS_Model):
#    """
#    A child class of GWPD_SS_Model responsible for 
#    ODE propagation.  Still toying with the best way to handle
#    the data
#    """
#
#    def __init__(self):
#        super().__init__()        
#
#    def propagate(self):
#
#
#
#
#










