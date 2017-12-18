import numpy as np

from .abstractpf import AbstractPupilFunction
from .gaussianrndf import GaussianRandomField

#### Globals ####
PI     = np.pi
TWO_PI = 2. * PI

class ModelPupilFunction(AbstractPupilFunction):
    '''
    Model pupil function

    This is our **real** model

    - P(x, y) = Same as the SimplePupilFunction, but with a hole with radius 'b'
    - W(x, y) is a Gaussian random field based on the von Karman turbulence model
    '''

    strut_width = 0.04

    def pFunc(self, x, y):
        pass1 = np.where((x)**2 + (y)**2 <= (self.radius())**2, 1., 0.)
        pass2 = np.where((x)**2 + (y)**2 > ((self.opts['b'] / 2.)**2), pass1, 0.)
        pass3 = np.where(np.abs(x) < ModelPupilFunction.strut_width*self.radius(), 0., pass2)
        pass4 = np.where(np.abs(y) < ModelPupilFunction.strut_width*self.radius(), 0., pass3)
        return pass4

    def wFunc(self, x, y):
        if self.hasOption('turbulence') and self.opts['turbulence'] is False:
            return np.zeros((self.samples, self.samples))
        else:
            # Init the field
            if not hasattr(self, 'gaussfield'):
                self.gaussfield = GaussianRandomField(self)

            # In case the generated field is not rendered
            # We want to preserve the same field for a single instance to simulate
            # a particular mirror. Also good for consistent and comparable images
            if not hasattr(self, 'renderedField'):
                self.renderedField = self.gaussfield.randomfield(ModelPupilFunction.atm_Pk(self.gaussfield.KX, self.gaussfield.KY))

            return self.renderedField
 
    @staticmethod
    def atm_Pk(kx, ky):
        '''
        Models atmospheric turbulence according to Von Karman spectrum

        http://community.dur.ac.uk/james.osborn/thesis/thesisse3.html
        '''
        l_min = 10e-3  # meters
        l_max = 20.   # meters
        r0    = 20e-2 # meters

        k_M = 5.92 / l_min
        k_0 = TWO_PI / l_max

        k2 = kx**2. + ky**2.

        # k2 = np.where(k2 < k_0**2, 0., k2)
        # k2 = np.where(k2 > k_M**2, 0., k2)

        a  = 0.023

        num = (np.abs(k2 + k_0**2))**(-11./6.)
        dem = r0**(5./3.)
        expfac = np.exp(-1. * (k2 / k_M**2))

        return a * (num / dem) * expfac
