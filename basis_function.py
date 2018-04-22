import copy
import string
import re
import time
import copy
import os.path
import numpy as np
import scipy as sp
from scipy import constants
from scipy import integrate



hartreetokjmol = 2625.49962
kjmoltohartree = 1.0/hartreetokjmol
au2femtosec = 2.4189e-2
femtosec2au = 1.0/2.4189e-2



class Basis_function(object):
    """
    Class for a Gaussian basis function parameters
    Attributes:
        a: real [N-basis x  N-dimensional] matrix of widths
        p: real [N-basis x  N-dimensional] matrix of momenta
        g: complex scalar norm and phase determined by the local 
            harmonic approximation
        a: N-dimensional vector of widths
        adt: No longer used time-derivative of widths
        xcdt: real [N-basis x  N-dimensional] matrix
              time-derivative of positions given by p_i/m
        pdt: real [N-basis x  N-dimensional] matrix
              time-derivative of momenta given by - dV(x_i)/dx
        gdt: time derivative of the phase
        c:  A vector of the QM expansion coefficients for
                the basis functions.
    

    """
    def __init__(self, width, center, momentum, phase, c):
        self.a = width
        self.xc = center
        self.p = momentum
        self.g = phase
        self.c = c
   
        self.wfs = {}
        
        return
    def calc_overlap(self):
        """
        Gaussian Overlap Integrals

        For a basis set with N basis functions
        returns I0 = array([N,N], dtype=complex)
        """
        ndim = self.a.shape[1]

        A = self.a[:, None] + self.a[None, :]
        B = (2.0*self.a[None, :]*self.xc[None, :] + 1.0j*self.p[None, :]
             + 2.0*self.a[:, None]*self.xc[:, None] - 1.0j*self.p[:, None])
        C = (np.conjugate(self.g[:, None]) + self.g[None, :]
                   +np.sum(-1.0j*self.p[None, :]*self.xc[None, :] 
                    + 1.0j*self.p[:, None]*self.xc[:, None]
                    - self.a[None, :]*self.xc[None, :]**2
                    - self.a[:, None]*self.xc[:, None]**2, axis=-1))
        I0_temp = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
                   *np.exp(np.sum((1./4.)*B*(1/A)*B, axis=-1) + C))
        return I0_temp
    
    def calc_tcf(self, ibasis, i_type, i_surf = 0):
        """
        Calculates the time correlation function (TCF) between the
        current basis function and the inital basis function.

        Calculates the overlap matrix between the two basis sets
        as I0_temp

        Takes the matrix-vector products of the basis function
        coefficients and overlap matrix to produce scalar value
       
        i_type=1: Calculates normal TCF
        i_type=2: Exploits property of real initial wave function
        to calculate TCF at time = 2*t

        """
        ndim = ibasis.a.shape[1]
        
        if i_type == 1:
            A = ibasis.a[:, None] + self.a[None, :]
            B = (2.0*self.a[None, :]*self.xc[None, :] + 1.0j*self.p[None, :]
                 + 2.0*ibasis.a[:, None]*ibasis.xc[:, None] - 1.0j*ibasis.p[:, None])
            C = (ibasis.g[:, None].conj() + self.g[None, :]
                       +np.sum(-1.0j*self.p[None, :]*self.xc[None, :] 
                        + 1.0j*ibasis.p[:, None]*ibasis.xc[:, None]
                        - self.a[None, :]*self.xc[None, :]**2
                        - ibasis.a[:, None]*ibasis.xc[:, None]**2, axis=-1))
            #print"new C shape {}\nC = {}".format(C.shape, C)
            I0_temp = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
                       *np.exp(np.sum((1./4.)*B*(1/A)*B, axis=-1) + C))

            return np.dot(ibasis.c[:,i_surf].T.conj(), 
                            np.dot(I0_temp, self.c[:,i_surf]))

        elif i_type == 2:
            A = self.a[:, None] + self.a[None, :]
            B = (2.0*self.a[None, :]*self.xc[None, :] + 1.0j*self.p[None, :]
                 + 2.0*self.a[:, None]*self.xc[:, None] + 1.0j*self.p[:, None])
            C = (self.g[:, None] + self.g[None, :]
                       +np.sum(-1.0j*self.p[None, :]*self.xc[None, :]
                              - 1.0j*self.p[:, None]*self.xc[:, None]
                              - self.a[None, :]*self.xc[None, :]**2
                              - self.a[:, None]*self.xc[:, None]**2, axis=-1))
            #print"new C shape {}\nC = {}".format(C.shape, C)
            I0_temp = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
                       *np.exp(np.sum((1./4.)*B*(1/A)*B, axis=-1) + C))
        
            return np.dot(self.c[:,i_surf].T, np.dot(I0_temp, self.c[:,i_surf]))
            
#            A = self.a[:, None] + self.a[None, :]
#            B = (2.0*self.a[None, :]*self.xc[None, :] - 1.0j*self.p[None, :]
#                 + 2.0*self.a[:, None]*self.xc[:, None] + 1.0j*self.p[:, None])
#            C = (self.g[:, None] + self.g[None, :]
#                       +np.sum(1.0j*self.p[None, :]*self.xc[None, :] 
#                        - 1.0j*self.p[:, None]*self.xc[:, None]
#                        - self.a[None, :]*self.xc[None, :]**2
#                        - self.a[:, None]*self.xc[:, None]**2, axis=-1))
#            #print"new C shape {}\nC = {}".format(C.shape, C)
#            I0_temp = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
#                       *np.exp(np.sum((1./4.)*B*(1/A)*B, axis=-1) + C))
#            return np.dot(self.c[:, i_surf],
#                    np.dot(I0_temp, self.c[:, i_surf]))
#            A = ibasis.a[:, None] + self.a[None, :]
#            B = (2.0*self.a[None, :]*self.xc[None, :] + 1.0j*self.p[None, :]
#                 + 2.0*ibasis.a[:, None]*ibasis.xc[:, None] + 1.0j*ibasis.p[:, None])
#            C = (ibasis.g[:, None] + self.g[None, :]
#                       +np.sum(-1.0j*self.p[None, :]*self.xc[None, :] 
#                        - 1.0j*ibasis.p[:, None]*ibasis.xc[:, None]
#                        - self.a[None, :]*self.xc[None, :]**2
#                        - ibasis.a[:, None]*ibasis.xc[:, None]**2, axis=-1))
#            #print"new C shape {}\nC = {}".format(C.shape, C)
#            I0_temp = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
#                       *np.exp(np.sum((1./4.)*B*(1/A)*B, axis=-1) + C))
#
##            return np.dot(ibasis.c[:, i_surf],
#            return np.dot(self.c[:, i_surf],
#                    np.dot(I0_temp, self.c[:, i_surf]))


    
    def calc_H_and_odt(self, potential, diffs, pot_approx, mass):
        """Galerkin Hamiltonian and time-derivative of overlap matrix
        calculation
        """
        nbasis = self.a.shape[0]
        ndim = self.a.shape[1]


        a_temp = self.a
        x_temp = self.xc
        p_temp = self.p
        x_dot = diffs['xcdt']
        p_dot = diffs['pdt']
        g_dot = diffs['gdt']
        A = self.a[:, None] + self.a[None, :]
        B = (2.0*self.a[None, :]*self.xc[None, :] + 1.0j*self.p[None, :]
             + 2.0*self.a[:, None]*self.xc[:, None] - 1.0j*self.p[:, None])
        C = (np.conjugate(self.g[:, None]) + self.g[None, :]
                   +np.sum(-1.0j*self.p[None, :]*self.xc[None, :] 
                    + 1.0j*self.p[:, None]*self.xc[:, None]
                    - self.a[None, :]*self.xc[None, :]**2
                    - self.a[:, None]*self.xc[:, None]**2, axis=-1))

        I0 = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
                   *np.exp(np.sum((1./4.)*B*(1/A)*B, axis=-1) + C))

        I1 = np.sum((1./(2.*A))*B, axis=-1)
        I2 = np.sum(((1./(2.*A))*B)**2 + (1./(2.*A)), axis=-1)*I0

        #Scalars for the integral terms in the kinetic energy
        #The integrals are different products of gaussians and cartesian coordinates
        #H0*int(conj(f(x)), f(x)) + H1*int(conj(f(x)), x*f(x))  + H2*int(conj(f(x)), x**2*f(x))
        H0 = np.sum(- p_temp[None, :, :]**2 
              + 4.*a_temp[None, :, :]**2.*x_temp[None, :, :]**2 
              + 4.j*a_temp[None, :, :]*x_temp[None, :, :]*p_temp[None, :, :]
              - 2.*a_temp[None, :, :], axis=-1)

        H1 = np.sum(((1./(2.*A))*B)*(-8.*a_temp[None, :, :]**2*x_temp[None, :, :] - 4.j*a_temp[None, :, :]*p_temp[None, :, :]), axis=-1)

        H2 = np.sum((((1./(2.*A))*B)**2 + (1./(2.*A)))*(4.*a_temp[None, :, :]**2), axis=-1)


        #Scalars for the integral terms in the overlap time derivative
        #dt0*int(conj(f(x)), f(x)) + dt1*int(conj(f(x)), x*f(x))  + dt2*int(conj(f(x)), x**2*f(x))
        dt1 = np.sum(((1./(2.*A))*B)*
                (2.0*a_temp[None, :, :]*x_dot[None, :, :] + 1.0j*p_dot[None, :, :]), axis = -1)
        dt0 = (np.sum(- 2.0*a_temp[None, :, :]*x_temp[None, :, :]*x_dot[None, :, :]
                      - 1.0j*p_temp[None, :, :]*x_dot[None, :, :]
                      - 1.0j*p_dot[None, :, :]*x_temp[None, :, :] , axis = -1) + g_dot[None, :])


        T = -(1./(2.*mass))*(H0 + H1 + H2)*I0
        ###SPA
        #Calculate midpoint of basis functions and calculate V there
        #In practice, you can exploit the fact that only (N*(N+1))/2
        #are unique (I think? been a few years), but not an issue
        #with model systems
        if pot_approx == "spa":
            centers = (x_temp[None, :] + x_temp[:, None])/2.0
            centers = centers.reshape(nbasis**2, ndim)
            V = potential.calc_V(centers).reshape(nbasis, nbasis)*I0

        #BAT
        #Calculate V at each basis function center and then average the points
        #for each integral.  Calculate gradient for improved accuracy.  You
        #will always have the gradient
        if pot_approx == "bat":
            tempV = potential.calc_V(self.xc)
            V1 = potential.calc_V1(self.xc)
            V = np.zeros([nbasis, nbasis], dtype=np.complex128)
            part_V1a = np.sum((((1./(2.*A))*B) - x_temp[None])*V1[None], axis=-1)*I0
            part_V1b = np.sum((((1./(2.*A))*B) - x_temp[:, None])*V1[:, None], axis=-1)*I0

            V += ((tempV[None, :] + tempV[:, None])/2.0)*I0
            V += (part_V1a + part_V1b)/2.0

        #Special cases where the potential energy is analytically solvable for Gaussians
        #As expected, this is the best option when available
        system = potential.__class__.__name__
        if pot_approx == "analytical":
            ##HenonHeiles integrals
            if system == "HH2dMCTDH":
                V = potential.calc_V(A, B, I0, I2)
            if system == "HH4d":
                V = potential.calc_V(A, B, I0, I2)
            if system == "HH6d":
                V = potential.calc_V(A, B, I0, I2)

            elif system == "Morse":
                V = potential.calc_V_analytical(self, C)


        H = T + V
        odt_temp = (dt0 + dt1)*I0

        return I0, H, odt_temp



    def plot_psi(self, job_name, step_number, write_data = True, 
                    store_data = False, xspace = 0.1, i_surf = 0,
                    x_lims = None):
        """
            Plots the basis set, wave function, density, etc.
    
            Only useful for 1- or 2-dimensional problems

            The options at the end are primarily added to aid in making
            animations.  It is much easier if you keep the xspacing and
            x limits identical at all times if you want to make an animation
            in 2-D
    
        """
    
    
        ndim  = self.a.shape[1]
        
        #if err_calc == 1:
        #    psi_file = np.loadtxt("ref_morse_1_psi_" + str(step_number))
        #    ref_psi = np.zeros([psi_file.shape[0], 2], dtype = np.complex128)
        #    x_points = psi_file[:, 0]
        #    ref_psi[:,1] = psi_file[:,1] + 1.0j*psi_file[:,2]
        #elif err_calc == 2:
        #    ref_dens = np.loadtxt("ref_morse_1_dens_" + str(step_number))
        #    x_points = ref_dens[:, 0]
        #else:     
        #Add buffer for density beyond maximum basis coordinates
        if x_lims is not None:
            if ndim == 1:   
                min_x = x_lims[0]
                max_x = x_lims[1]
            elif ndim == 2:   
                min_x = x_lims[:,0] 
                max_x = x_lims[:,1]
        else:
            min_x = np.min(self.xc, axis=0) - 5.0
            max_x = np.max(self.xc, axis=0) + 5.0
    
        if ndim == 1:
            npoints = np.int((max_x - min_x)/xspace)
            x_points = np.linspace(min_x, max_x, npoints).reshape(-1,1)
            
            tot_points = x_points.size 
           
            #The old code didn't fit the new indexing rules.  Not certain this still works
            disp = x_points[:, None, :] - self.xc[None,:, :]
            psi_data = np.exp((-self.a[None,:,:]*disp**2 + 1.0j*self.p[None,:,:]*disp).sum(axis=-1) + self.g[None, :])
            #plot_psi  = np.zeros([tot_points, 4])
            #plot_psi[:, 0]  = x_points.flatten()
            
            plot_psi_data = np.dot(psi_data, self.c)#[:, i_surf])
            #plot_psi_data = np.dot(psi_data, self.c[:, i_surf])
            #plot_psi_data = np.dot(psi_data, self.c)#[:, i_surf])
            #plot_psi[:, 1] = plot_psi_data.real
            #plot_psi[:, 2] = plot_psi_data.imag
            #plot_psi[:, 3] = plot_psi_data.real**2 + plot_psi_data.imag**2
    
        elif ndim == 2:
            npoints = ((max_x - min_x)/xspace).astype(int)
            x, y = np.meshgrid(np.linspace(min_x[0], max_x[0], npoints[0]),
                               np.linspace(min_x[1], max_x[1], npoints[1]))
            x_points = np.c_[x.reshape(-1), y.reshape(-1)]
    
            tot_points = x_points.shape[0]
            #plot_psi  = np.zeros([tot_points, 5])
            #plot_psi[:, :2]  = x_points
            disp = x_points[:, None, :] - self.xc[None,:, :]
            psi_data = np.exp((-self.a[None,:,:]*disp**2 + 1.0j*self.p[None,:,:]*disp).sum(axis=-1) + self.g[None, :])
    
            plot_psi_data = np.dot(psi_data, self.c)
            #plot_psi_data = np.dot(psi_data, self.c[:, i_surf])
            #plot_psi[:, 2] = plot_psi_data.real
            #plot_psi[:, 3] = plot_psi_data.imag
            #plot_psi[:, 4] = plot_psi_data.real**2 + plot_psi_data.imag**2
    
        if write_data: 
            np.savetxt(job_name + '.wf_{:4.2f}'.format(step_number), np.c_[x_points, plot_psi_data.view(float), np.abs(plot_psi_data)**2])
        if store_data: 
            self.wfs['{:.0f}'.format(step_number)] = np.c_[x_points, plot_psi_data]
            return
        #if err_calc == 1:
        #    error = 0
        #    for i in xrange(x_points.size):
        #        error += np.absolute(psi_data[i] - ref_psi[i,1])**2
        #        #error += np.absolute(dens_data[i, 1] - ref_dens[i,1])**2
        #    print"t = {} error = {}".format(time, error)
        #elif err_calc == 2:
        #    #error = 0
        #    error = np.sum((plot_dens[:,1] - ref_dens[:,1])**2)
        #    #for i in xrange(x_points.size):
        #    #    #error += np.absolute(psi_data[i] - ref_psi[i,1])**2
        #    #    error += np.absolute(plot_dens[i, 1] - ref_dens[i,1])**2
        #    print"t = {} error = {}".format(time, error)
        #
        #else:
        #    error = 0.0
        #if system == "Eckart":
        #    bar_center = 0.0
        #    r_len = 0
        #    while plot_dens[r_len, 0] < bar_center:
        #        r_len+=1
        #    r_packet = np.zeros([r_len, 2])
        #    t_packet = np.zeros([npoints - r_len, 2])
        #    r_packet[:, 0] = plot_dens[:r_len, 0]
        #    t_packet[:, 0] = plot_dens[r_len:, 0]
        #    r_packet[:, 1] = plot_dens[:r_len, 1]
        #    t_packet[:, 1] = plot_dens[r_len:, 1]
        #
        #    r_int = sp.integrate.simps(r_packet[:, 1], r_packet[:, 0])
        #    t_int = sp.integrate.simps(t_packet[:, 1], t_packet[:, 0])
        #    tot_int = r_int + t_int
        #    print"Reflected = {}   Transmitted = {} Total norm {}".format(r_int, t_int, tot_int)
        #    print"Renormalized: Reflected = {} Transmitted = {}".format(r_int/tot_int, t_int/tot_int)
        #else:
        #    t_int = 0.0
    
    
    
        #return error, t_int







