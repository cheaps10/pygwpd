import numpy as np
import scipy as sp

#import basis_function

"""

Chad Heaps
April 2018

This file contains classes for evaluation of different potentials
and their gradients. Most have been tested and work.  The Henon-Heiles
could be vectorized further but they get the job done.

There are single-surface potentials first and towards the bottom
some common diabatic potentials for nonadiabatic systems

"""

###Single Surface





class FreeParticle(object):
    """
        Free Particle
    """
    def calc_V(self, x):
        """
            Calculate V for Free Particle
        """

        return np.zeros([x.shape[0]])

    def calc_V1(self, x):
        """
            Calculate V1 for Free Particle
        """
        return np.zeros(x.shape)

class Eckart(object):
    def __init__(self, sys_params = 1):
        """
            Eckart barrier
            Parameterized by a: barrier height and b: barrier width
        """
        #Tannor barrier
        if sys_params == 1:
            self.a = 40.0
            self.b = 4.32
        #Garaschuk barrier
        elif sys_params == 2:
            self.a = 16.0
            self.b = 1.3624
        elif len(sys_params) == 2:
            self.a = sys_params[0]
            self.b = sys_params[1]

    def calc_V(self, x):
        return self.a*(1/(np.cosh(self.b*x[:, 0])**2))

    def calc_V1(self, x):
        return -2.0*self.b*self.a*np.tanh(self.b*x)*((1/np.cosh(self.b*x))**2)
    
    def calc_V2(self, x):
        return 2.0*self.b**2*self.a*(np.cosh(2.0*self.b*x)-2.0)*(1.0/((np.cosh(self.b*x)**2)**2))

class Harmonic(object):
    def __init__(self, sys_params):
        """
            1-D Harmonic well
        """
        self.omega = sys_params[0]
        self.x0 = sys_params[1]
        self.m = sys_params[2] 
    def calc_V(self, x):
        return 0.5*self.m*self.omega**2*(x[:,0] - self.x0)**2

    def calc_V1(self, x):
        return self.m*self.omega**2*(x - self.x0)

    def calc_V2(self, x):
        return self.m*self.omega**2

class Harmonic2d(object):
    def __init__(self, sys_params):
        """
            2-D Harmonic Well
        """
        self.omega = sys_params[0]
        self.x0 = sys_params[1]
        self.y0 = sys_params[2]
        self.m = sys_params[3] 
    def calc_V(self, x):
        return 0.5*self.m*self.omega**2*((x[:, 0] - self.x0)**2 + (x[:, 1] - self.y0)**2)

    def calc_V1(self, x):
        Vx = 0.5*self.m*self.omega**2*(2.*(x[:, 0] - self.x0))
        Vy = 0.5*self.m*self.omega**2*(2.*(x[:, 1] - self.y0))
        return np.array([Vx,Vy]).T

    def calc_V2(self, x):
        return self.m*self.omega**2


class Quartic(object):
    def __init__(self, sys_params):
        self.omega = sys_params[0]
        self.x0 = sys_params[1]
        self.a = sys_params[2]
        self.m = sys_params[3] 
    def calc_V(self, x):
        return 0.5*self.m*self.omega**2*(x - self.x0)**2 + self.a*(x-self.x0)**4

    def calc_V1(self, x):
        return self.m*self.omega**2*(x - self.x0)+ 4.0*self.a*(x - self.x0)**3

    def calc_V2(self, x):
        return self.m*self.omega**2 + 12*self.a*(x - self.x0)**2

class DoubleWell(object):
    def __init__(self, sys_params = [-0.0068, 0.003, 0.0]):
        self.a2 = sys_params[0]
        self.a4 = sys_params[1]
        self.x0 = sys_params[2]
    def calc_V(self, x):
        return self.a2*(x - self.x0)**2 + self.a4*(x-self.x0)**4

    def calc_V1(self, x):
        return 2.0*self.a2*(x - self.x0) + 4.0*self.a4*(x-self.x0)**3

    def calc_V2(self, x):
        return 2.0*self.a2 + 12.0*self.a4*(x-self.x0)**2

class Morse(object):
    def __init__(self, sys_params = [10.25, 0.2209]):
        self.D = sys_params[0]
        self.a = sys_params[1]
    
    def calc_V(self, x):
        return self.D*(np.exp(-2.0*self.a*x[:,0]) - 2.0*np.exp(-self.a*x[:,0]))
    
    def calc_V1(self, x):
        return self.D*(-2.0*self.a*np.exp(-2.0*self.a*x) + 2.0*self.a*np.exp(-self.a*x))

    def calc_V2(self, x):
        return self.D*(4*self.a**2*np.exp(-2.0*self.a*x) - 2.0*self.a**2*np.exp(-self.a*x))
    
    def calc_V_analytical(self, gbasis, C):
            ndim = gbasis.a.shape[1]
            A = gbasis.a[:, None] + gbasis.a[None, :]

            B1 = (2.0*gbasis.a[None, :]*gbasis.xc[None, :] 
                + 2.0*gbasis.a[:, None]*gbasis.xc[:, None] 
                - 2.0*self.a
                + 1.0j*gbasis.p[None, :] - 1.0j*gbasis.p[:, None])
            B2 = (2.0*gbasis.a[None, :]*gbasis.xc[None, :] 
                + 2.0*gbasis.a[:, None]*gbasis.xc[:, None] 
                - self.a
                + 1.0j*gbasis.p[None, :] - 1.0j*gbasis.p[:, None])

            V1_temp = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
                    *np.exp(np.sum((1./4.)*B1*(1/A)*B1, axis=-1) + C))
            V2_temp = (np.sqrt(sp.pi**ndim/np.prod(A, axis=-1))
                    *np.exp(np.sum((1./4.)*B2*(1/A)*B2, axis=-1) + C))
            V = self.D*(V1_temp - 2.*V2_temp)
            return V


class HH2dHeller(object):
    
    def __init__(self,sys_params = [1.3, 0.7, -0.1, 0.1]):
        """
            A Henon-Heiles model from one of Heller's early papers
        """
        self.w1 = sys_params[0]
        self.w2 = sys_params[1]
        self.l = sys_params[2]
        self.n = sys_params[3]
    def calc_V(self, x):
        return (0.5*self.w1**2*x[:,0]**2 + 0.5*self.w2**2*x[:,1]**2
                + self.l*x[:,1]*(x[:,0]**2 + self.n*x[:,1]**2))
    
    def calc_V1(self, x):
        Vx = self.w1**2*x[:,0] + 2.0*self.l*x[:,0]*x[:,1]
        Vy = self.w2**2*x[:, 1] + self.l*(x[:, 0]**2 + 3.*self.n*x[:, 1]**2)
        for i in range(x.shape[0]):
            if np.absolute(x[i, 0]) > 10.0:
                Vx[i] = 0.0
            if np.absolute(x[i, 1]) > 10.0:
                Vy[i] = 0.0 
        return np.array([Vx, Vy]).T

class HH2dMCTDH(object):
    def __init__(self, sys_params = np.sqrt(0.0125)):
        """
            2-D version of Henon-Heiles used in paper
        """
        self.l = sys_params
    def calc_V(self, x):
        return (0.5*(x[:,0]**2 + x[:,1]**2)
                + self.l*(x[:,1]*x[:,0]**2 - (1./3.)*x[:,1]**3))
    
    def calc_V1(self, x):
        Vx = x[:,0] + 2.0*self.l*x[:,0]*x[:,1]
        Vy = x[:, 1] + self.l*(x[:, 0]**2 - x[:, 1]**2)
        return np.array([Vx, Vy]).T

    def calc_V_analytical(self, A,B, I0, I2):
        x2y = ((1./(2.*A[:, :, 1]))*B[:, :, 1]*(((1./(2.*A[:, :, 0]))*B[:, :, 0])**2 + (1./(2.*A[:, :, 0]))))*I0
        y3 = (3.*((1./(2.*A[:, :, 1]))**2*B[:, :, 1]) + ((1./(2.*A[:, :, 1]))*B[:, :, 1])**3)*I0
        V = 0.5*I2  + np.sqrt(0.0125)*(x2y - (y3/3.))
        return V

class HH4d(object):
    def __init__(self, sys_params = np.sqrt(0.0125)):
        """
            4-D version of Henon-Heiles used in paper
        """
        self.l = sys_params
    def calc_V(self, x):
        return (0.5*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2 + x[:, 3]**2)
                + self.l*((x[:, 1]*x[:, 0]**2 - (1./3.)*x[:, 1]**3)
                        + (x[:, 2]*x[:, 1]**2 - (1./3.)*x[:, 2]**3)
                        + (x[:, 3]*x[:, 2]**2 - (1./3.)*x[:, 3]**3)))
    
    def calc_V1(self, x):
        V1 = np.zeros(x.shape)
        V1[:, 0] = x[:, 0] + 2.0*self.l*x[:, 0]*x[:, 1]
        V1[:, 1] = x[:, 1] + self.l*(2.*x[:, 1]*x[:, 2] + x[:, 0]**2 - x[:, 1]**2)
        V1[:, 2] = x[:, 2] + self.l*(2.*x[:, 2]*x[:, 3] + x[:, 1]**2 - x[:, 2]**2)
        V1[:, 3] = x[:, 3] + self.l*(x[:, 2]**2 - x[:, 3]**2)
        return V1

    def calc_V_analytical(self, A,B, I0, I2):
        x2y_1 = ((1./(2.*A[:, :, 1]))*B[:, :, 1]*(((1./(2.*A[:, :, 0]))*B[:, :, 0])**2 + (1./(2.*A[:, :, 0]))))*I0
        x2y_2 = ((1./(2.*A[:, :, 2]))*B[:, :, 2]*(((1./(2.*A[:, :, 1]))*B[:, :, 1])**2 + (1./(2.*A[:, :, 1]))))*I0
        x2y_3 = ((1./(2.*A[:, :, 3]))*B[:, :, 3]*(((1./(2.*A[:, :, 2]))*B[:, :, 2])**2 + (1./(2.*A[:, :, 2]))))*I0
        y3_1 = (3.*((1./(2.*A[:, :, 1]))**2*B[:, :, 1]) + ((1./(2.*A[:, :, 1]))*B[:, :, 1])**3)*I0
        y3_2 = (3.*((1./(2.*A[:, :, 2]))**2*B[:, :, 2]) + ((1./(2.*A[:, :, 2]))*B[:, :, 2])**3)*I0
        y3_3 = (3.*((1./(2.*A[:, :, 3]))**2*B[:, :, 3]) + ((1./(2.*A[:, :, 3]))*B[:, :, 3])**3)*I0
        V = 0.5*I2  + np.sqrt(0.0125)*(x2y_1 - (y3_1/3.)
                                     + x2y_2 - (y3_2/3.)
                                     + x2y_3 - (y3_3/3.))

        return V



class HH6d(object):
    def __init__(self, sys_params = np.sqrt(0.0125)):
        """
            6-D version of Henon-Heiles used in paper
        """
        self.l = sys_params
    def calc_V(self, x):
        return (0.5*np.sum(x**2, axis=1)
                + self.l*((x[:, 1]*x[:, 0]**2 - (1./3.)*x[:, 1]**3)
                        + (x[:, 2]*x[:, 1]**2 - (1./3.)*x[:, 2]**3)
                        + (x[:, 3]*x[:, 2]**2 - (1./3.)*x[:, 3]**3)
                        + (x[:, 4]*x[:, 3]**2 - (1./3.)*x[:, 4]**3)
                        + (x[:, 5]*x[:, 4]**2 - (1./3.)*x[:, 5]**3)))
    
    def calc_V1(self, x):
        V1 = np.zeros(x.shape)
        V1[:, 0] = x[:, 0] + 2.0*self.l*x[:, 0]*x[:, 1]
        V1[:, 1] = x[:, 1] + self.l*(2.*x[:, 1]*x[:, 2] + x[:, 0]**2 - x[:, 1]**2)
        V1[:, 2] = x[:, 2] + self.l*(2.*x[:, 2]*x[:, 3] + x[:, 1]**2 - x[:, 2]**2)
        V1[:, 3] = x[:, 3] + self.l*(2.*x[:, 3]*x[:, 4] + x[:, 2]**2 - x[:, 3]**2)
        V1[:, 4] = x[:, 4] + self.l*(2.*x[:, 4]*x[:, 5] + x[:, 3]**2 - x[:, 4]**2)
        V1[:, 5] = x[:, 5] + self.l*(x[:, 4]**2 - x[:, 5]**2)
        return V1

    def calc_V_analytical(self, A,B, I0, I2):
        x2y_1 = ((1./(2.*A[:, :, 1]))*B[:, :, 1]*(((1./(2.*A[:, :, 0]))*B[:, :, 0])**2 + (1./(2.*A[:, :, 0]))))*I0
        x2y_2 = ((1./(2.*A[:, :, 2]))*B[:, :, 2]*(((1./(2.*A[:, :, 1]))*B[:, :, 1])**2 + (1./(2.*A[:, :, 1]))))*I0
        x2y_3 = ((1./(2.*A[:, :, 3]))*B[:, :, 3]*(((1./(2.*A[:, :, 2]))*B[:, :, 2])**2 + (1./(2.*A[:, :, 2]))))*I0
        x2y_4 = ((1./(2.*A[:, :, 4]))*B[:, :, 4]*(((1./(2.*A[:, :, 3]))*B[:, :, 3])**2 + (1./(2.*A[:, :, 3]))))*I0
        x2y_5 = ((1./(2.*A[:, :, 5]))*B[:, :, 5]*(((1./(2.*A[:, :, 4]))*B[:, :, 4])**2 + (1./(2.*A[:, :, 4]))))*I0
        y3_1 = (3.*((1./(2.*A[:, :, 1]))**2*B[:, :, 1]) + ((1./(2.*A[:, :, 1]))*B[:, :, 1])**3)*I0
        y3_2 = (3.*((1./(2.*A[:, :, 2]))**2*B[:, :, 2]) + ((1./(2.*A[:, :, 2]))*B[:, :, 2])**3)*I0
        y3_3 = (3.*((1./(2.*A[:, :, 3]))**2*B[:, :, 3]) + ((1./(2.*A[:, :, 3]))*B[:, :, 3])**3)*I0
        y3_4 = (3.*((1./(2.*A[:, :, 4]))**2*B[:, :, 4]) + ((1./(2.*A[:, :, 4]))*B[:, :, 4])**3)*I0
        y3_5 = (3.*((1./(2.*A[:, :, 5]))**2*B[:, :, 5]) + ((1./(2.*A[:, :, 5]))*B[:, :, 5])**3)*I0
        V = 0.5*I2  + np.sqrt(0.0125)*(x2y_1 - (y3_1/3.)
                                     + x2y_2 - (y3_2/3.)
                                     + x2y_3 - (y3_3/3.)
                                     + x2y_4 - (y3_4/3.)
                                     + x2y_5 - (y3_5/3.))


        return V


class HH10d(object):
    def __init__(self, sys_params = np.sqrt(0.0125)):
        """
            10-D version of Henon-Heiles
        """
        self.l = sys_params
    def calc_V(self, x):
        return (0.5*np.sum(x**2, axis=1)
                + self.l*((x[:, 1]*x[:, 0]**2 - (1./3.)*x[:, 1]**3)
                        + (x[:, 2]*x[:, 1]**2 - (1./3.)*x[:, 2]**3)
                        + (x[:, 3]*x[:, 2]**2 - (1./3.)*x[:, 3]**3)
                        + (x[:, 4]*x[:, 3]**2 - (1./3.)*x[:, 4]**3)
                        + (x[:, 5]*x[:, 4]**2 - (1./3.)*x[:, 5]**3)
                        + (x[:, 6]*x[:, 5]**2 - (1./3.)*x[:, 6]**3)
                        + (x[:, 7]*x[:, 6]**2 - (1./3.)*x[:, 7]**3)
                        + (x[:, 8]*x[:, 7]**2 - (1./3.)*x[:, 8]**3)
                        + (x[:, 9]*x[:, 8]**2 - (1./3.)*x[:, 9]**3)))
    
    def calc_V1(self, x):
        V1 = np.zeros(x.shape)
        V1[:, 0] = x[:, 0] + 2.0*self.l*x[:, 0]*x[:, 1]
        V1[:, 1] = x[:, 1] + self.l*(2.*x[:, 1]*x[:, 2] + x[:, 0]**2 - x[:, 1]**2)
        V1[:, 2] = x[:, 2] + self.l*(2.*x[:, 2]*x[:, 3] + x[:, 1]**2 - x[:, 2]**2)
        V1[:, 3] = x[:, 3] + self.l*(2.*x[:, 3]*x[:, 4] + x[:, 2]**2 - x[:, 3]**2)
        V1[:, 4] = x[:, 4] + self.l*(2.*x[:, 4]*x[:, 5] + x[:, 3]**2 - x[:, 4]**2)
        V1[:, 5] = x[:, 5] + self.l*(2.*x[:, 5]*x[:, 6] + x[:, 4]**2 - x[:, 5]**2)
        V1[:, 6] = x[:, 6] + self.l*(2.*x[:, 6]*x[:, 7] + x[:, 5]**2 - x[:, 6]**2)
        V1[:, 7] = x[:, 7] + self.l*(2.*x[:, 7]*x[:, 8] + x[:, 6]**2 - x[:, 7]**2)
        V1[:, 8] = x[:, 8] + self.l*(2.*x[:, 8]*x[:, 9] + x[:, 7]**2 - x[:, 8]**2)
        V1[:, 9] = x[:, 9] + self.l*(x[:, 8]**2 - x[:, 9]**2)
        return V1


class DoubleWell2D(object):
    def __init__(self):
        self.A  = 1.8897
        self.B  = 2.*1.8897
        self.delta = (self.A+self.B)/2.0
        self.c     = 800. / 2625.49962
        self.y0    = 2.*1.8897
        self.k    = 14.45

    def calc_V(self, x):
        return ((self.c/((self.delta - self.A)**2*(self.delta-self.B)**2))*
                (x[:, 0] - self.A)**2*(x[:, 0] - self.B)**2
                + (self.k/2.0)*(x[:,1] - self.y0)**2)
    def calc_V1(self, x):
        Vx = ((4.0*self.c/((self.delta - self.A)**2*(self.delta-self.B)**2))*
                (x[:, 0] - self.A)*(x[:, 0] - self.B))
        Vy =  (self.k)*(x[:,1] - self.y0)
        return np.array([Vx, Vy]).T

class NOCl(object):
    """
        The NOCl PES from the 1991 MCTDH paper
        My implementation never quite worked
    """
    def __init__(self, sys_params):
        self.a = np.array([0.6816, -0.9123, 0.4115])
        self.da = np.array([2.0*0.6816, -3.0*0.9123, 4.0*0.4115])
        self.alpha = 1.5
        self.b     = 1.1
        self.red    = 4.315
        self.rev    = 2.136
        self.etheta = 127.4
        self.C = np.zeros([4, 5, 7])
        self.C[0, 0, :] =  [ 0.03848160,  0.024787500,  0.02709330,   0.00126791,  0.00541285,  0.03136290,  0.017244900]
        self.C[0, 1, :] =  [ 0.00834237,  0.003987130,  0.00783319,   0.02948870, -0.01543870, -0.06219840, -0.033795100]
        self.C[0, 2, :] =  [ 0.00161625, -0.000156330, -0.01899820,  -0.00753297,  0.00383665, -0.00758225, -0.009044930]
        self.C[0, 3, :] =  [-0.00101010,  0.000619148, -0.01498120,  -0.01997220,  0.00873013,  0.03761180,  0.022152300]
        self.C[0, 4, :] =  [-0.00036890,  0.000164037, -0.00331809,  -0.00567787,  0.00268662,  0.01344830,  0.008458500]
        self.C[1, 0, :] =  [-0.05586660, -0.027657600,  0.09349320,  -0.02956380, -0.15436000,  0.07961190,  0.135121000]
        self.C[1, 1, :] =  [ 0.05821690,  0.038440400,  0.07811400,   0.18555600, -0.06416560, -0.17597600, -0.010499400]     
        self.C[1, 2, :] =  [ 0.05228500,  0.047272400, -0.21600800,  -0.14777500,  0.34928300,  0.28458000,  0.003844490]
        self.C[1, 3, :] =  [ 0.02126090,  0.029059700, -0.10912400,   0.03104450,  0.26251300, -0.25065300, -0.369466000]
        self.C[1, 4, :] =  [ 0.00334178,  0.003906100, -0.00110452,   0.05820290,  0.06795240, -0.16459000, -0.165337000]   
        self.C[2, 0, :] =  [-0.16318600, -0.180535000,  0.04692000,   0.47167300,  0.40326700, -0.71807100, -0.761199000]
        self.C[2, 1, :] =  [-0.02906740, -0.013617200, -0.10895200,  -1.68269000, -1.26730000,  3.17648000,  2.927930000]
        self.C[2, 2, :] =  [ 0.12122800,  0.202308000,  0.48361300,   1.29095000, -0.17448300, -2.46050000, -1.365970000]
        self.C[2, 3, :] =  [ 0.10723300,  0.115213000, -0.36610200,   0.81266200,  1.76038000, -1.19665000, -1.773920000]
        self.C[2, 4, :] =  [ 0.02327670,  0.030493200, -0.19455000,  -0.03075170,  0.53936500,  0.12020300, -0.251289000]
        self.C[3, 0, :] =  [ 0.08389750,  0.198853000, -0.09947660,  -0.82240900, -0.58600600,  1.17402000,  1.173780000]
        self.C[3, 1, :] =  [-0.18204700, -0.245637000,  0.13039600,   2.85439000,  2.44277000, -5.36406000, -5.228060000]
        self.C[3, 2, :] =  [-0.22749300, -0.470604000, -0.67055500,  -1.66997000,  0.26867700,  3.71822000,  2.106780000]
        self.C[3, 3, :] =  [-0.13635000, -0.193843000,  0.62607600,  -1.55192000, -3.22512000,  3.03851000,  4.013640000]
        self.C[3, 4, :] =  [-0.02625540, -0.039129100,  0.31285800,  -0.12206300, -1.03112000,  0.28978000,  0.878604000]


        self.qexp = np.arange(2,5)
        self.dqexp = np.arange(1,4)
        self.ijk = np.zeros([4, 5, 7, 3])
        for i in range(5):
            for j in range(7):
                self.ijk[:, i, j, 0] = np.arange(4)
        for i in range(4):
            for j in range(7):
                self.ijk[i, :, j, 1] = np.arange(5)
        for i in range(4):
            for j in range(5):
                self.ijk[i, j, :, 2] = np.arange(7)


    def calc_V(self, x):
        
        qv = x[:, 0] - self.rev
        qd = 1 - np.exp(-self.alpha*(x[:,1] - self.red))
#        qtheta = np.exp(-self.b*np.cos(x[:, 2])) - np.exp(-self.b*np.cos(self.etheta))
        
        V = np.sum(self.a[None, :]*qv[:, None]**self.qexp[None, :], axis=-1)

        big_matrix = self.C[None, :, :, :]*x[:, 0, None, None]**self.ijk[:, :, :, 0]*x[:, 1, None, None]**self.ijk[:, :, :, 1]*x[:, 2, None, None]**self.ijk[:, :, :, 2]

        #big_v = np.apply_over_axes(np.sum, big_matrix, [-1, -1, -1])
        #V += (1.0 - qd)*big_v

        V += (1.0 - qd)*np.apply_over_axes(np.sum, big_matrix, [-1, -1, -1])

        return V
    
    def calc_V1(self, x):
        
        V1 = np.zeros([x.shape])
        dqv = np.ones([x.shape[0]])
        dqd =   self.alpha*np.exp(-self.alpha*(x[:,1] - self.red))
        dqtheta = -self.b*np.exp(-self.b*np.cos(x[:, 2]))
        qv = x[:, 0] - self.rev
        qd = 1 - np.exp(-self.alpha*(x[:,1] - self.red))
        #V1[:, 0] += np.sum(self.da[None, :]*qv[:, None]**self.dqexp[None, :]*dqv, axis=-1)
        V1[:, 0] += np.sum(self.qexp[None, :]*self.a[None, :]*qv[:, None]**(self.qexp[None, :] - 1.0)*dqv, axis=-1)
    
        big_matrixV = ((self.ijk[:, :, :, 0])*
                      self.C[None, :, :, :]
                      *x[:, 0, None, None]**(self.ijk[:, :, :, 0] - 1)*dqv[:, None, None]
                      *x[:, 1, None, None]**self.ijk[:, :, :, 1]
                      *x[:, 2, None, None]**self.ijk[:, :, :, 2])

        big_matrixD = ((self.ijk[:, :, :, 1])*
                      self.C[None, :, :, :]
                      *x[:, 0, None, None]**self.ijk[:, :, :, 0]
                      *x[:, 1, None, None]**(self.ijk[:, :, :, 1] - 1)*dqd[:, None, None]
                      *x[:, 2, None, None]**self.ijk[:, :, :, 2])
        big_matrixT = ((self.ijk[:, :, :, 2])*
                      self.C[None, :, :, :]
                      *x[:, 0, None, None]**self.ijk[:, :, :, 0]
                      *x[:, 1, None, None]**self.ijk[:, :, :, 1]
                      *x[:, 2, None, None]**(self.ijk[:, :, :, 2] - 1)*dqtheta[:, None, None])

        
        V1[:, 0] += (1.0 - qd)*np.apply_over_axes(np.sum, big_matrixV, [-1, -1, -1])
        V1[:, 1] +=  - np.apply_over_axes(np.sum, big_matrixD, [-1, -1, -1])
        V1[:, 2] += (1.0 - qd)*np.apply_over_axes(np.sum, big_matrixT, [-1, -1, -1])
        
        return V1

###Nonadiabatic

class Tully1(object):
    """
    Caculates energy and gradients for Tully model 1 Potentials
    """

    def __init__(self, sys_params = [0.01, 1.6, 0.005, 1.0]):
        self.a = sys_params[0]
        self.b = sys_params[1]
        self.c = sys_params[2]
        self.d = sys_params[3]
        self.n_surface = 2

    def calc_V(self, x):
        V = np.zeros([x.shape[0], 2, 2])
        V1 = np.zeros([x.shape[0], 2, 2, x.shape[1]])
        
        #Diagonal Diabatic energies
        #V[:, 0, 0] = self.a*np.tanh(x[:, 0]*self.b) 
        #V[:, 1, 1] = - self.a*np.tanh(x[:, 0]*self.b)
        V[:, 0, 0] = self.a*np.tanh(x[:, 0]*self.b) 
        V[:, 1, 1] = - self.a*np.tanh(x[:, 0]*self.b)
        #Diabatic coupling
        V[:, 0, 1] = self.c*np.exp(-self.d*(x[:, 0]**2))
        V[:, 1, 0] = V[:, 0, 1]
        ##Gradients
        V1[:, 0, 0, :] = self.a * self.b * (1/(np.cosh(self.b * x) ** 2))
        V1[:, 1, 1, :] = - self.a * self.b * (1/(np.cosh(self.b * x) ** 2))
        V1[:, 0, 1, :] = -2.0*self.c*self.d*x*np.exp(-self.d*(x**2))
        V1[:, 1, 0, :] = V1[:, 0, 1, :]
        return V, V1


class Tully2(object):                
    """
    Caculates energy and gradients for Tully model 2 Potentials
    """
    def __init__(self, sys_params = [0.10, 0.28, 0.05, 0.015, 0.06]):
        self.a  = sys_params[0]
        self.b  = sys_params[1]
        self.Eo = sys_params[2]
        self.c  = sys_params[3]
        self.d  = sys_params[4]
        self.n_surface = 2

    def calc_V(self, x):
        V = np.zeros([x.shape[0], 2, 2])
        V1 = np.zeros([x.shape[0], 2, 2, x.shape[1]])
        
        #Diagonal Diabatic energies
        V[:, 0, 0] = np.zeros([x.shape[0]])
        V[:, 1, 1] = -self.a * np.exp(-self.b * (x[:, 0]**2)) + self.Eo
        #Diabatic coupling
        V[:, 0, 1] = self.c*np.exp(-self.d*(x[:, 0]**2))
        V[:, 1, 0] = V[:, 0, 1]
        ##Gradients
        V1[:, 0, 0, :] = np.zeros([x.shape[0], x.shape[1]])
        V1[:, 1, 1, :] = 2.0 * self.a * self.b * x * np.exp(-self.b * (x**2))
        V1[:, 0, 1, :] = -2.0 * self. c * self.d * x * np.exp(-self.d * (x**2))
        V1[:, 1, 0, :] = V1[:, 0, 1, :]
        return V, V1

                
class Tully3(object):
    """
    Caculates energy and gradients for Tully model 3 Potentials
    """
    def __init__(self, sys_params = [6.0e-4, 0.1, 0.9]):
        self.A  = sys_params[0]
        self.b  = sys_params[1]
        self.c = sys_params[2]
    def calc_Va(self, x):
        vec = self.A*np.ones([x.size])
        return [vec, np.zeros([x.size]), np.zeros([x.size])]
    
    def calc_Vb(self, x):
        vec = self.A*np.ones([x.size])
        return [-vec, np.zeros([x.size]), np.zeros([x.size])]

    def calc_Vc(self, x):
        b = self.b
        c = self.c
        V = np.zeros([x.size],  dtype = np.float64)
        V1 = np.zeros([x.size], dtype = np.float64)
        V2 = np.zeros([x.size], dtype = np.float64)
        V3 = np.zeros([x.size], dtype = np.float64)
        V4 = np.zeros([x.size], dtype = np.float64)
        for i in range(x.size):
            if x[i] < 0.0:
                V[i]  = b*np.exp(c*x[i])
                V1[i] = b*c*np.exp(c*x[i])
                V2[i] = b*(c**2)*np.exp(c*x[i])
                V3[i] = b*(c**3)*np.exp(c*x[i])
                V4[i] = b*c**4*np.exp(c*x[i])
            if x[i] > 0.0:
                 V[i] = b*(2.0 - np.exp(-c*x[i]))
                 V1[i]= b*c*np.exp(-c*x[i])
                 V2[i]= -b*(c**2)*np.exp(-c*x[i])
                 V3[i]= b*(c**3)*np.exp(-c*x[i])
                 V4[i]= -b*c**4*np.exp(-c*x[i])
        return V, V1, V2

class Subotnik2d(object):
    """
    A 2-D model from Subotnik's paper on surface-hopping in multiple 
    dimensions...I think that's where I got it
    Subotnik, J. Phys. Chem. A 2011, 115, 12083-12096
    """
    def __init__(self, sys_params = [0.2, 0.6, 0.015, 0.3, 
                                     0.2/4.0, 0.3, 2.0]):
        self.A = sys_params[0]
        self.B = sys_params[1]
        self.C = sys_params[2]
        self.D = sys_params[3]
        self.F = sys_params[4]
        self.G = sys_params[5]
        self.W = sys_params[6]
        self.n_surface = 2

    def calc_Va(self, x):
        V1 = np.zeros(x.shape)
        V  = - self.F*np.tanh(self.B*x[:, 0]) 
        V1[:, 0] = - self.F*self.B*(1/(np.cosh(self.B*x[:, 0])**2)),
        return V, V1

    def calc_Vb(self, x):
        V1 = np.zeros(x.shape)
        z = self.B*(x[:, 0] - 1) + self.W*np.cos(self.G*x[:, 1] + sp.pi/2.0)
        V = self.A*np.tanh(z) + (3.*self.A)/4.
        z1x = self.B
        z1y = - self.G*self.W*np.sin(self.G*x[:, 1] + sp.pi/2.0)
        V1[:, 0] = self.A*(1./(np.cosh(z)**2))*z1x
        V1[:, 1] = self.A*(1./(np.cosh(z)**2))*z1y

        return V, V1

    def calc_Vc(self, x):
        V1 = np.zeros(x.shape)
        V = self.C*np.exp(-self.D*x[:, 0]**2)
        V1[:, 0] = -2.0*self.D*self.C*x[:,0]*np.exp(-self.D*x[:, 0]**2)
        return V, V1 

class Morse_two_surf(object):                
    """
    Caculates energy and gradients for two Morse potentials
    coupled by a Gaussian
    """
    
    def __init__(self, sys_params = [0.675, 1.890, 2.278e-2, 0.0000, 
                                     0.453, 3.212, 1.025e-2, 3.8e-3, 
                                     6.337e-3, 0.56, 2.744]):
        self.a1  = sys_params[0]
        self.b1  = sys_params[1]
        self.d1 = sys_params[2]
        self.e1  = sys_params[3]
        self.a2  = sys_params[4]
        self.b2  = sys_params[5]
        self.d2 = sys_params[6]
        self.e2  = sys_params[7]
        self.A  = sys_params[8]
        self.c  = sys_params[9]
        self.rx  = sys_params[10]

        self.n_surface = 2

    def calc_V(self, x):
        V = np.zeros([x.shape[0], 2, 2])
        V1 = np.zeros([x.shape[0], 2, 2, x.shape[1]])
        
        #Diagonal Diabatic energies
        V[:, 0, 0] = self.d1*(1.0 - np.exp(-self.a1*(x[:, 0] - self.b1)))**2 + self.e1
        V[:, 1, 1] = self.d2*(1.0 - np.exp(-self.a2*(x[:, 0] - self.b2)))**2 + self.e2
        #Diabatic coupling
        V[:, 0, 1] = self.A*np.exp(-self.c*(x[:, 0] - self.rx)**2)
        V[:, 1, 0] = V[:, 0, 1]

        ##Gradients
        V1[:, 0, 0, 0] = self.d1*(-2.0*self.a1*np.exp(-2.0*self.a1*(x[:, 0] - self.b1)) 
                        + 2.0*self.a1*np.exp(-self.a1*(x[:, 0] - self.b1)))
        V1[:, 1, 1, 0] = self.d2*(-2.0*self.a2*np.exp(-2.0*self.a2*(x[:, 0] - self.b2)) 
                        + 2.0*self.a2*np.exp(-self.a2*(x[:, 0] - self.b2)))
        V1[:, 0, 1, 0] = - self.A*self.c*(x[:, 0]-self.rx)*np.exp(-self.c*(x[:, 0] - self.rx)**2)
        V1[:, 1, 0, 0] = V1[:, 0, 1, 0]

        return V, V1

class Morse_three_surf(object):                
    """
    Calculates potential and gradients for 3 coupled Morse potentials
    where the coupling is a Gaussian
    E. A. Coronado, J. Xing, W. H. Miller, Chem. Phys. Let. 349 (2001) 521-529
    The parameters are
    Surfaces:
    V[i, i] = De[i]*(1-exp(-beta[i]*(x- Re[i])))**2 + c[i]
    
    Coupling:
    V[i, j] = A[i, j]*exp(-a[i, j]*(x - R[i, j])**2)
    """
    def __init__(self, sys_params):
        if sys_params == 1:
            self.xc = np.array([2.9])
            self.De = np.array([0.003, 0.004, 0.003])
            self.B = np.array([0.65, 0.6, 0.65])
            self.Re = np.array([5.0, 4.0, 6.0])
            self.c = np.array([0.00, 0.01, 0.006])
            
            self.A = 0.002*np.ones([3, 3])
            self.R = np.zeros([3, 3])
            self.R[0, 1] = 3.4 
            self.R[1, 0] = 3.4 
            self.R[1, 2] = 4.8 
            self.R[2, 1] = 4.8 
            self.a = 16.0*np.ones([3, 3])

            self.n_surface = 3

        if sys_params == 2:
            self.xc = np.array([3.3])
            self.De = np.array([0.02, 0.01, 0.003])
            self.B = np.array([0.65, 0.4, 0.65])
            self.Re = np.array([4.5, 4.0, 4.4])
            self.c = np.array([0.00, 0.01, 0.02])
            
            self.A = 0.005*np.ones([3, 3])
            self.R = np.zeros([3, 3])
            self.R[0, 1] = 3.66 
            self.R[1, 0] = 3.66 
            self.R[0, 2] = 3.34 
            self.R[2, 0] = 3.34 
            self.a = 32.0*np.ones([3, 3])

        if sys_params == 3:
            self.xc = np.array([2.1])
            self.De = np.array([0.02, 0.02, 0.003])
            self.B = np.array([0.4, 0.65, 0.65])
            self.Re = np.array([4.0, 4.5, 6.0])
            self.c = np.array([0.02, 0.00, 0.02])
            
            self.A = 0.005*np.ones([3, 3])
            self.R = np.zeros([3, 3])
            self.R[0, 1] = 3.4
            self.R[1, 0] = 3.4 
            self.R[0, 2] = 4.97 
            self.R[2, 0] = 4.97 
            self.a = 32.0*np.ones([3, 3])

    def calc_V(self, x):
        V = np.zeros([x.shape[0], 3, 3])
        V1 = np.zeros([x.shape[0], 3, 3, x.shape[1]])
        #Diagonal PES's
        V[:, 0, 0] = self.De[0]*(1.0- np.exp(-self.B[0]*(x[:, 0] - self.Re[0])))**2 + self.c[0]
        V[:, 1, 1] = self.De[1]*(1.0- np.exp(-self.B[1]*(x[:, 0] - self.Re[1])))**2 + self.c[1]
        V[:, 2, 2] = self.De[2]*(1.0- np.exp(-self.B[2]*(x[:, 0] - self.Re[2])))**2 + self.c[2]
        #Coupling

        V[:, 0, 1] = self.A[0, 1]*np.exp(-self.a[0, 1]*(x[:, 0] - self.R[0, 1])**2) #Vab
        V[:, 0, 2] = self.A[0, 2]*np.exp(-self.a[0, 2]*(x[:, 0] - self.R[0, 2])**2) #Vac
        V[:, 1, 0] = self.A[1, 0]*np.exp(-self.a[1, 0]*(x[:, 0] - self.R[1, 0])**2) #Vba
        V[:, 2, 0] = self.A[2, 0]*np.exp(-self.a[2, 0]*(x[:, 0] - self.R[2, 0])**2) #Vca
        V[:, 1, 2] = self.A[1, 2]*np.exp(-self.a[1, 2]*(x[:, 0] - self.R[1, 2])**2) #Vbc
        V[:, 2, 1] = self.A[2, 1]*np.exp(-self.a[2, 1]*(x[:, 0] - self.R[2, 1])**2) #Vcb
##    def calc_V(self, x):
##        return self.D*(np.exp(-2.0*self.a*x[:,0]) - 2.0*np.exp(-self.a*x[:,0]))
##    
##    def calc_V1(self, x):
##        return self.D*(-2.0*self.a*np.exp(-2.0*self.a*x) + 2.0*self.a*np.exp(-self.a*x))

        #Diagonal gradients
        V1[:, 0, 0, 0] = self.De[0]*(- 2.0*self.B[0]*np.exp(-2.*self.B[0]*(x[:, 0] - self.Re[0]))
                                    + 2.0*self.B[0]*np.exp(-self.B[0]*(x[:, 0] - self.Re[0])))
        V1[:, 1, 1, 0] = self.De[1]*(- 2.0*self.B[1]*np.exp(-2.*self.B[1]*(x[:, 0] - self.Re[1]))
                                    + 2.0*self.B[1]*np.exp(-self.B[1]*(x[:, 0] - self.Re[1])))
        V1[:, 2, 2, 0] = self.De[2]*(- 2.0*self.B[2]*np.exp(-2.*self.B[2]*(x[:, 0] - self.Re[2]))
                                    + 2.0*self.B[2]*np.exp(-self.B[2]*(x[:, 0] - self.Re[2])))
        #Coupling gradients
        V1[:, 0, 1, 0] = -self.a[0, 1]*(x[:, 0] - self.R[0, 1])*self.A[0, 1]*np.exp(-self.a[0, 1]*(x[:, 0] - self.R[0, 1])**2) #Vab
        V1[:, 0, 2, 0] = -self.a[0, 2]*(x[:, 0] - self.R[0, 2])*self.A[0, 2]*np.exp(-self.a[0, 2]*(x[:, 0] - self.R[0, 2])**2) #Vac
        V1[:, 1, 0, 0] = -self.a[1, 0]*(x[:, 0] - self.R[1, 0])*self.A[1, 0]*np.exp(-self.a[1, 0]*(x[:, 0] - self.R[1, 0])**2) #Vba
        V1[:, 2, 0, 0] = -self.a[2, 0]*(x[:, 0] - self.R[2, 0])*self.A[2, 0]*np.exp(-self.a[2, 0]*(x[:, 0] - self.R[2, 0])**2) #Vca
        V1[:, 1, 2, 0] = -self.a[1, 2]*(x[:, 0] - self.R[1, 2])*self.A[1, 2]*np.exp(-self.a[1, 2]*(x[:, 0] - self.R[1, 2])**2) #Vbc
        V1[:, 2, 1, 0] = -self.a[2, 1]*(x[:, 0] - self.R[2, 1])*self.A[2, 1]*np.exp(-self.a[2, 1]*(x[:, 0] - self.R[2, 1])**2) #Vcb

        return V, V1


def calc_dab(a, b, c, a1, b1, c1):
    """
    1-dimensional derivative coupling from diabatic states
    ...thought I had N-dimensions but don't know where that is
    """
    Nbasis = a.shape[0]
    Ndim = a1.shape[1]
    dab_temp = np.zeros([Nbasis, 2, 2, Ndim])
    del_V  = a - b
    del_V1 = a1[:, 0] - b1[:, 0]
    part1 = (-(2*c*(del_V1))/((del_V)**2))+((2*c1[:, 0])/(del_V))
    part2 = 1 + ((4*c**2)/((del_V)**2))
    dab_temp[:, 0, 1, 0] =  -(0.5*part1)/part2
    dab_temp[:, 1, 0, 0] =   (0.5*part1)/part2

    return dab_temp 


def calc_W(a, b, c, a1, b1, c1):
    """
    1-dimensional adiabatic surfaces from diabatic states
    ...thought I had N-dimensions but don't know where that is

    Going to hack up the 1-D case for N-D but need to test it later
    """

    Np = a1.shape[0]
    Ndim = a1.shape[1]
    W_temp = np.zeros([Np, 2])
    W1_temp = np.zeros([Np, 2, Ndim])
    del_V = a  - b
    #del_V1 = a1[:, 0] - b1[:, 0]
    del_V1 = a1 - b1
    
    w_sqrt_a = np.sqrt((del_V)**2 + 4*(c**2))

    W_temp[:, 0] = 0.5*(a+b - w_sqrt_a) 
    W_temp[:, 1] = 0.5*(a+b + w_sqrt_a) 
    
    #W1_temp[:, 0, 0] = 0.5*(a1[:, 0] + b1[:, 0] - ((2*del_V*del_V1 + 8*c*c1[:, 0])/(2*w_sqrt_a)))
    #W1_temp[:, 1, 0] = 0.5*(a1[:, 0] + b1[:, 0] + ((2*del_V*del_V1 + 8*c*c1[:, 0])/(2*w_sqrt_a)))
    W1_temp[:, 0, :] = 0.5*(a1 + b1 - ((2*del_V[:,None]*del_V1 + 8*c[:,None]*c1)/(2*w_sqrt_a[:, None])))
    W1_temp[:, 1, :] = 0.5*(a1 + b1 + ((2*del_V[:,None]*del_V1 + 8*c[:,None]*c1)/(2*w_sqrt_a[:, None])))


    return W_temp, W1_temp

