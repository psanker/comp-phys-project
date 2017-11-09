import numpy as np

from .cassegrainpf import CassegrainPupilFunction
from .gaussianrndf import GaussianRandomField

class DirtyCassegrainPupilFunction(CassegrainPupilFunction):
    '''
    Same as the SimplePupilFunction but with a Gaussian random field
    to simulate deviations in the optics.
    '''

    def __init__(self, **opts):
        super().__init__(**opts)

        # Private variable because yeah.
        self.__pk2 = lambda k : np.where(k > 1e-15, (1. / k)**(2.), 0.)

    def wFunc(self, x, y):

        # Init the field
        if not hasattr(self, 'gaussfield'):
            self.gaussfield = GaussianRandomField(self, self.__pk2)

        # In case the generated field is not rendered
        # We want to preserve the same field for a single instance to simulate
        # a particular mirror. Also good for consistent and comparable images
        if not hasattr(self, 'renderedField'):
            self.renderedField = self.gaussfield.randomfield()

        return self.renderedField
