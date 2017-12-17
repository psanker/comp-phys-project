import numpy as np

from .abstractpf import AbstractPupilFunction

#### Globals ####
PI     = np.pi
TWO_PI = 2. * PI

class ModelPupilFunction(AbstractPupilFunction):
    '''
    Model pupil function

    This is our **real** model

    - P(x, y) = Same as the SimplePupilFunction, but with a hole with radius 'b'
    - W(x, y) = 0 for no phase shifting
    '''

    strut_width = 0.04

    def pFunc(self, x, y):
        pass1 = np.where((x)**2 + (y)**2 <= (self.radius())**2, 1., 0.)
        pass2 = np.where((x)**2 + (y)**2 > ((self.opts['b'] / 2.)**2), pass1, 0.)
        pass3 = np.where(np.abs(x) < ModelPupilFunction.strut_width*self.radius(), 0., pass2)
        pass4 = np.where(np.abs(y) < ModelPupilFunction.strut_width*self.radius(), 0., pass3)
        return pass4

    def wFunc(self, x, y):
        return np.zeros((self.samples, self.samples))

    def atm_Pk(self, kx, ky):
        '''
        Models atmospheric turbulence according to Von Karman spectrum

        http://community.dur.ac.uk/james.osborn/thesis/thesisse3.html
        '''
        l_min = 1e-3  # meters
        l_max = 1e2   # meters
        r0    = 20e-2 # meters

        k_max = 5.92 / l_min
        k_min = TWO_PI / l_max

        k2 = kx**2. + ky**2.

        k2 = np.where(k2 < k_min**2, 0., k2)
        k2 = np.where(k2 > k_max**2, 0., k2)

        a  = 0.023

        num = (np.abs(k2 + k_min**2))**(-11./6.)
        dem = r0**(5./3.)
        expfac = np.exp(-1. * (k2 / k_max**2))

        return a * (num / dem) * expfac

    # def kolmogorov_Pk(self, kx, ky):
    #    a  = 0.0023
    #    r0 = 20e-2 # meters
    #
    #    k = np.sqrt(kx**2 + ky**2)
    #
    #    return a * (r0**(-5./3.)) * (k**(-11./3.))


