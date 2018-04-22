import copy
import os
import warnings
import numpy as np

#import scipy.sparse.linalg as spla

import potential
import basis_function as bf

"""
    Chad Heaps
    April 2018

    Program to perform nonadiabatic dynamics
    on multi-dimensional model systems using either
    the diabatic or adiabatic representations

    The diatabic to adiabatic transformation is limited.  I don't think I ever extended it to 
    N surfaces, although the rest of the code does support N surfaces, just in the diabatic representation

"""

class GWPD_Nonadiabatic_Model(object):
    """
    Class structure for propgation of wave packet
    on multiple surfaces using a time-dependent
    Gaussian wave packet
    """

    def __init__(self, **kwargs):
        prop_defaults = {
                'model': potential.Morse_two_surf(),
                'mass':2000.0,
                'x0': np.array([4.0]),
                'k0': np.array([0.0]),
                'basis_function_width': np.array([0.5]),
                'basis_set_size':10,
                'add_cs':False,
                'time_step':1.0,
                'prop_time':1000.0,
                'save_wf_times':None,
                'write_wf':True,
                'store_wf':False,
                'save_all_wf':False,
                'job_name':'test_nonadiabatic',
                'tcf_type':1,
                'basis_velocity':'ehrenfest',
                'electronic_rep':'diabatic',
                'init_surface':0,
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
        self.nsurf = self.model.n_surface
        #Initialize time correlation function.
        self.tcf = np.zeros([0, 1+self.nsurf], dtype=np.complex)
        #Initialize time-dependent state population storage
        self.pops = np.zeros([0, 1+self.nsurf])
        if self.save_wf_times is not None:
            self.save_steps = self.save_wf_times / self.time_step
        else:
            self.save_steps = []
        if self.save_all_wf:
            self.save_steps = np.arange(self.nsteps)

        self.save_xc = np.zeros([self.nbasis, self.ndim, self.nsteps])
        self.save_c = np.zeros([self.nbasis, self.nsurf, self.nsteps], dtype=np.complex)
        if self.electronic_rep == 'adiabatic' and self.ndim > 1:
            warnings.warn('Warning!! Adiabatic surface generation from diabatic'
                          ' states may be broken for more than 1 dimension.')
        if (self.electronic_rep == 'adiabatic') and (self.nsurf > 2):
            raise InputError('Adiabatic states are only '
                              'available for 2 surface systems!')

        self.store_xc = np.zeros([self.nbasis, self.nsteps])

        return
    def setup_basis(self):
        """
        Generate the necessary Basis_function instance in order to
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
                                    np.zeros([self.nbasis, self.nsurf], dtype=np.complex128))
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
            init_basis.c[:, self.init_surface] = np.dot(overlap_inv, init_a)

        elif self.matrix_type == "pseudospectral": 
            disp = init_basis.xc[:, None, :] - init_basis.xc[None,:,:]
            a_temp = init_basis.a
            p_temp = init_basis.p

            init_psi_basis = np.exp((-a_temp[None,:,:]*disp**2 + 
                                        1.0j*p_temp[None,:,:]*disp).sum(axis=-1) 
                                        + init_basis.g[None, :])
            ex_disp = ix - target_func.xc[:]
            init_psi_exact = np.exp((-target_func.a*ex_disp**2 
                                    + 1.0j*target_func.p*ex_disp).sum(axis=-1)
                                    + target_func.g)
            psi_basis_inv = np.linalg.pinv(init_psi_basis, rcond = 1e-8)

            init_basis.c[:, self.init_surface] = np.dot(psi_basis_inv, init_psi_exact)
        
        check_norm = np.dot(init_basis.c[:,self.init_surface].T.conj(), 
                                np.dot(I0, init_basis.c[:,self.init_surface]))
        #check_norm = np.dot(init_basis.c.T.conj(), np.dot(I0, init_basis.c))
        with open(self.outfile, 'a+') as f:
            f.write('Initial WF norm {:4.2e}\n'.format(check_norm))

        self.init_basis = init_basis
        self.prop_basis = copy.deepcopy(init_basis)
        self.dC0 = np.zeros([self.nbasis, self.nsurf], dtype=np.complex)

        return

    def propagate(self):
        """
        Perform time-stepping loop for your favorite
        ODE solver.  Right now RK4 is the only coded one

        """

        #Since we're using a fixed time step integration
        #We can just calculate how many time steps we need
        #to integrate rather than worrying about a variable
        #number of steps to reach prop_time
        self.dC0 = np.zeros([self.nbasis, self.nsurf], dtype=np.complex)

        if self.integrator == 'RK4' or self.integrator == 'rk4':
            for step in range(self.nsteps):
                return_val = self.rk4_single_step(step)
                if return_val < 0:
                    break
                #if isinstance(dC0, int):
                #    break


        #self.prop_basis.plot_psi(self.job_name, nsteps*self.time_step, store_data = self.store_wf)
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
            #print('saving psi at time ', ttemp, step)
            self.prop_basis.plot_psi(self.job_name, step, 
                                    write_data = self.write_wf,
                                    store_data = self.store_wf)
        self.save_xc[:, :, step] = self.prop_basis.xc
        self.save_c[:, :, step] = self.prop_basis.c
        ##Save TCF and normalization data
        I0 = self.prop_basis.calc_overlap()
        #wf_norm = np.dot(self.prop_basis.c[:,0].conj(), np.dot(I0, self.prop_basis.c[:,0]))
        wf_norm = np.dot(self.prop_basis.c.T.conj(), np.dot(I0, self.prop_basis.c)).real.diagonal()
        tcf_val = self.prop_basis.calc_tcf(self.init_basis, self.tcf_type, i_surf = np.arange(self.nsurf)).diagonal()

#        self.tcf = np.vstack((self.tcf, np.array([ttemp, tcf_val])))
#        self.pops = np.vstack((self.pops, np.array([ttemp, wf_norm])))
        self.tcf = np.vstack((self.tcf,np.hstack((self.tcf_type*ttemp, tcf_val))))
        self.pops = np.vstack((self.pops, np.hstack((ttemp, wf_norm))))
        np.savetxt(self.job_name + ".tcf_" + str(self.prop_time), self.tcf.view(float))
        np.savetxt(self.job_name + ".pops_" + str(self.prop_time), self.pops)
        if self.wf_renorm:
#            print('Renormalization weight ', wf_norm.sum(), np.sqrt(wf_norm.sum()) )
            self.prop_basis.c/=np.sqrt(wf_norm.sum())

        if wf_norm.sum() > 10.0:
            with open(self.outfile, 'a+') as f:
                f.write(('Warning!! Norm is very large, at time = {:4.2f} exiting.' 
                         + self.nsurf*'{: 5.2} ' + '\n').format(ttemp, *wf_norm))
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
        
        return 0


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
        ##Time derivatives of basis set using Ehrenfest trajectories
        #The diabatic surfaces are gradients are stored in two tensors
        #V_diabat = np.zeros([Np, Nsurf, Nsurf])
        #V1_diabat = np.zeros([Np, Nsurf, Nsurf, Ndim])
        V_diabat, V1_diabat = self.model.calc_V(temp_basis.xc)

        #This code is very hazy in my head
        if self.electronic_rep == "adiabatic":
            ##Explicitly define ground (Va) excited (Vb)
            ##and coupling (Vc) surfaces
            #PES values
            Va  = V_diabat[:, 0, 0]
            Vb  = V_diabat[:, 1, 1]
            Vc  = V_diabat[:, 0, 1]
            #Gradients
            V1a = V1_diabat[:, 0, 0, :]
            V1b = V1_diabat[:, 1, 1, :]
            V1c = V1_diabat[:, 0, 1, :]

            #These functions are the ones that may not work
            #for multiple dimensions.  I thought I ran adiabatic
            #surfaces in multiple dimensions, but don't know where the code is
            ##Generate adiabatic surfaces and gradients
            V, V1 = potential.calc_W(Va, Vb, Vc, V1a, V1b, V1c)
            ##Calculate non-adiabatic derivative coupling
            dab = potential.calc_dab(Va, Vb, Vc, V1a, V1b, V1c)

            ##Generate Ehrenfest potentials and forces
            weights = np.absolute(temp_basis.c)**2
            ##New versions
            ##(|cA|**2*Va + |cB|**2*Vb)/(|cA|**2 + |cB|**2)
            V_eh = np.sum(weights*V, axis=-1)/np.sum(weights, axis=-1) #Should element wise multily cA*Va, then sum over all the surfaces
            V1_eh = np.sum(weights[:, :, None]*V1, axis=-2)
            #The next line adds the derivative coupling contribution to the adiabatic surface gradients.  
            #Upon first inspection, I am not certain how I set this up, but I believe it works, at least if dab is correct
            V1_eh += np.apply_over_axes(np.sum, 
                                       ((temp_basis.c[:, None, :, None]*temp_basis.c[:, :, None, None].conj()).real*dab
                                       *(V[:, :, None, None] - V[:, None, :, None])),
                                       [1,2]).reshape(self.nbasis, self.ndim)

            V1_eh/=np.sum(weights, axis=-1)[:, None]

        #The diabatic surfaces were used in the paper and therefore I am more confident
        #in their accuracy.  Still, the broadcasting obscures the interpretation a bit
        if self.electronic_rep == "diabatic":
            V = V_diabat
            V1 = V1_diabat
            ##Generate Ehrenfest potentials and forces for diabatic surfaces
            weights = (np.absolute(temp_basis.c)**2).sum(axis=-1)
            ##New versions
            ##(|cA|**2*Va + |cB|**2*Vb + a1*.conj()a2*V12 + a2.conj()*a1*V21)/(|cA|**2 + |cB|**2)
            ##W = np.sum(weights*V, axis=-1)/np.sum(weights, axis=-1) #Should element wise multiply cA*Va, then sum over all the surfaces
            
            part1 = (temp_basis.c[:, None, :]*temp_basis.c[:, :, None].conj())
            V_eh = ((part1*V_diabat).sum(axis=-1).sum(axis=-1)).real/weights
            #Similar idea here
            #part2 has dimensions (Nbasis, Nsurf, Nsurf, Ndim)
            #Then there is element-wise multiplication with V1 where the coefficients in part2
            #are broadcasted over every dimension
            #The two summations sum over the different surface combinations,
            #leaving an (Nbasis, Ndim) matrix
            part2 = (temp_basis.c[:, None, :, None]*temp_basis.c[:, :, None, None].conj())
            V1_eh = ((part2*V1_diabat).sum(axis = 1).sum(axis = 1)).real/weights[:, None]

        diffs['xcdt'] = temp_basis.p/self.mass
        diffs['pdt'] = -V1_eh
        diffs['gdt']  = -1.0j*(np.sum(-temp_basis.p**2/(2.0*self.mass) 
                                + temp_basis.a/self.mass, axis=1)
                                + V_eh)
        
        #Call appropriate QM method
        if self.matrix_type == "pseudospectral":
            if self.electronic_rep == 'adiabatic':
                diffs['cdt'] = self.calc_dC_pseudospectral_adiabatic(temp_basis, V, dab, diffs, dC0)
            if self.electronic_rep == 'diabatic':
                diffs['cdt'] = self.calc_dC_pseudospectral_diabatic(temp_basis, V, diffs, dC0)
        #elif self.matrix_type == "galerkin":
        #    diffs['cdt'] = self.calc_dC_galerkin(temp_basis, diffs, dC0)
        #print(diffs['cdt']) 
        return diffs

    def calc_dC_pseudospectral_adiabatic(self, gbasis, V, dab, diffs, dC0):
        """
        Calculate the differential change in the C coefficients
        using the pseudospectral method for adiabatic surfaces

        Although this is generalized to N-dimensions, there are some
        for-loops that will probably be costly
        
        My generation of model adiabatic surfaces is limited to 2 surfaces
        right now.  The nsurface extension may have been with molecular
        systems in mind.
        
        """
     
        xcdt = diffs['xcdt'] 
        pdt = diffs['pdt'] 
        gdt = diffs['gdt'] 
        ##Calculate displacements
        disp = gbasis.xc[:, None, :] - gbasis.xc[None,:,:]
        ##psi_basis is overlap or collocation matrix
        psi_basis = np.exp((-gbasis.a[None,:,:]*disp**2 + 1.0j*gbasis.p[None,:,:]*disp).sum(axis=-1) + gbasis.g[None, :])
        #Kinetic energy
        T = ((-1./(2.*self.mass))*(-2.*gbasis.a[None, :, :] 
            + (-2.*gbasis.a[None, :, :]*(disp) + 1.0j*gbasis.p[None, :, :])**2)).sum(axis=-1)
        
        T*=psi_basis  
        psi_xderiv = (-2.0*gbasis.a[None, :, :]*(disp) + 1.0j*gbasis.p[None, :, :])*psi_basis[:, :, None]
        #time derivative of the overlap matrix

        psi_tderiv = ((2.0*gbasis.a[None, :, :]*(disp)*xcdt[None, :, :]
                             + 1.0j*pdt[None, :, :]*(disp)
                             - 1.0j*gbasis.p[None, :, :]*xcdt[None, :, :]).sum(axis=-1) 
                             + gdt[None, :])*psi_basis

        
  
        #previously H was just a matrix of Nbasis x Nbasis.  Now we have a tensor
        #that is Nbasis x Nbasis x Nsurf
        ##Wmat should be, like H, Nbasis x Nbasis x Nsurf
        Wmat = V[:, None, :]*psi_basis[:, :, None] 

        #dab needs to cover all surface combinations and dimensions
        #In staying consistent with the notation for the trajectories,
        #the dimensions will be Nbasis x Nbasis x Nsurf x Nsurf x Ndim
        #The first two make the usual matrix form of the potential spanning the grid points
        #the Nsurf x Nsurf accounts for all combinations of i and j for  dij
        #The last index takes care of multiple dimensions
        
        #The original code used loops, but the line below should be a proper
        #vectorization/broadcasting of the process
        ##dabMat = np.zeros([self.nbasis, self.nbasis, self.nsurf, self.nsurf, self.ndim], dtype=np.complex128)
        ##for i in range(self.nsurf):
        ##    for j in range(self.nsurf):
        ##        if i < j:
        ##            for k in range(self.ndim):
        ##                dabMat[:, :, i, j, k] = - dab[:, None, i, j, k]*psi_xderiv[:, :, k]/self.mass#[k]
        ##                dabMat[:, :, j, i, k] = -dabMat[:, :, i, j, k]
        dabMat = -dab[:,None,:,:]*psi_xderiv[:,:,None,None,:]/self.mass 
        Amat_inv = np.linalg.pinv(np.dot(psi_basis.T.conj(), psi_basis), rcond=self.svd_threshold)
        
        ##Now we need to build the Hamiltonian and calculate dC
        ##It will generally be (T + Vii)*ci + (T + Vij)*cj for i!=j
        #The b-vector now is Nbasis x Nsurf
        #Bvec = np.zeros([self.nbasis, self.nsurf], dtype = np.complex128)
        newB = np.zeros([self.nbasis, self.nsurf], dtype = np.complex128)
        dC = np.zeros([self.nbasis, self.nsurf], dtype = np.complex128)
    
        ##I started trying to vectorize this and never completed it
        #Bvec = (-1.0j*np.dot(np.moveaxis((T[:,:,None] 
        #                             + Wmat - 1.0j*psi_tderiv[:,:,None]),
        #                             -1,0),
        #                             gbasis.c))[0,:,:]

        #off_diag = (-1.0j*np.dot(np.moveaxis(np.moveaxis(np.moveaxis(dabMat, 
        #                                                             -1,0), 
        #                                                 -1,0), 
        #                                     -1,0),
        #                          gbasis.c)).sum(axis=2)[0,:,:]#.sum(axis=-1)

        #off_diag = np.moveaxis(off_diag,0,1)

        Bvec = np.zeros([self.nbasis, self.nsurf], dtype = np.complex128)
        for i in range(self.nsurf):
            Bvec[:, i] += -1.0j*np.dot((T + Wmat[:, :, i]) - 1.0j*psi_tderiv, gbasis.c[:, i])
            for j in range(self.nsurf):
                for k in range(self.ndim):
                    Bvec[:, i] += -1.0j*np.dot(dabMat[:, :, i, j, k], gbasis.c[:, j])
                #Bvec[:,i] -=off_diag[:,i,j]
                    
            newB[:, i] = np.dot(psi_basis.T.conj(), Bvec[:, i]) 
            dC[:, i] = np.dot(Amat_inv, newB[:, i])
            #print(np.allclose(old_Bvec, Bvec))

        return dC

    def calc_dC_pseudospectral_diabatic(self, gbasis, V, diffs, dC0):
        """
        Calculate the differential change in the C coefficients
        using the pseudospectral method for diabatic surfaces

        Still has loops over all of the surfaces.  At some level,
        this is required, but I suspect it could be improved
        
        """
     
        xcdt = diffs['xcdt'] 
        pdt = diffs['pdt'] 
        gdt = diffs['gdt'] 
        ##Calculate displacements
        disp = gbasis.xc[:, None, :] - gbasis.xc[None,:,:]
        ##psi_basis is overlap or collocation matrix
        psi_basis = np.exp((-gbasis.a[None,:,:]*disp**2 + 1.0j*gbasis.p[None,:,:]*disp).sum(axis=-1) + gbasis.g[None, :])
        #Kinetic energy
        T = ((-1./(2.*self.mass))*(-2.*gbasis.a[None, :, :] 
            + (-2.*gbasis.a[None, :, :]*(disp) + 1.0j*gbasis.p[None, :, :])**2)).sum(axis=-1)
        T*=psi_basis  
        #time derivative of the overlap matrix
        psi_tderiv = ((2.0*gbasis.a[None, :, :]*(disp)*xcdt[None, :, :]
                             + 1.0j*pdt[None, :, :]*(disp)
                             - 1.0j*gbasis.p[None, :, :]*xcdt[None, :, :]).sum(axis=-1) 
                             + gdt[None, :])*psi_basis

        #print(V[:, None, :, :].shape, psi_basis[:, :, None, None].shape) 
        Vmat = V[:, None, :, :]*psi_basis[:, :, None, None]        
        #print(Vmat.shape)
        Amat_inv = np.linalg.pinv(np.dot(psi_basis.T.conj(), psi_basis), rcond=self.svd_threshold)
        ##Now we need to build the Hamiltonian and calculate dC
        ##It will generally be (T + Vii)*ci + (T + Vij)*cj for i!=j
        #The b-vector now is Nbasis x Nsurf
                
        Bvec = np.zeros([self.nbasis, self.nsurf], dtype = np.complex128)
        newB = np.zeros([self.nbasis, self.nsurf], dtype = np.complex128)
        dC = np.zeros([self.nbasis, self.nsurf], dtype = np.complex128)
        for i in range(self.nsurf):
            Bvec[:, i] += -1.0j*np.dot((T + Vmat[:, :, i, i]) - 1.0j*psi_tderiv, gbasis.c[:, i])
            for j in range(self.nsurf):
                if j != i:
                    Bvec[:, i] += -1.0j*np.dot(Vmat[:, :, i, j], gbasis.c[:, j])
                    #print('i,j Vmat[:,:,i,j] ', i, j, Vmat[:,:,i,j])
                    #print('contribution for i,j ', i,j, -1.0j*np.dot(Vmat[:, :, i, j], gbasis.c[:, j]))
            newB[:, i] = np.dot(psi_basis.T.conj(), Bvec[:, i]) 
            dC[:, i] = np.dot(Amat_inv, newB[:, i])
            #dC[:, i] = np.dot(Amat_inv, np.dot(psi_basis.T.conj(), Bvec[:, i]))


        return dC


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

#    def __init__(self, expression, message):
    def __init__(self, message):
#        self.expression = expression
        self.message = message
        return

