import numpy as np

from .simplepf import SimplePupilFunction
from .gaussianrndf import GaussianRandomField

class DirtySimplePupilFunction(SimplePupilFunction):
    '''
    Same as the SimplePupilFunction but with a Gaussian random field
    to simulate deviations in the optics.
    '''

    def __init__(self, **opts):
        super(DirtySimplePupilFunction, self).__init__(**opts)
        
    def wFunc(self, x, y):

        # Init the field
        if not hasattr(self, 'gaussfield'):
            self.gaussfield = GaussianRandomField(self)

        # In case the generated field is not rendered
        # We want to preserve the same field for a single instance to simulate
        # a particular mirror. Also good for consistent and comparable images
        if not hasattr(self, 'renderedField'):
            self.renderedField = self.gaussfield.randomfield()

        return self.renderedField
