class AbstractPointSpreadFunction(object):
    '''
    An interface for the many implementations that we will need for this project
    Note: The PSF is defined in Fourier space

    Base units for length are in **centimeters**
    '''

    def __init__(self, **opts):

        # Assign defaults for all children
        self.diameter = 2.

        # Interpret any options passed to the constructor
        self.applySettings(opts)

    def applySettings(self, vals):
        for k, v in vals.iteritems():
            if k == 'diameter':
                self.diameter = v

    def convolute(self):
        '''
        Convolute the PSF, given its parameters, and produce an image
        '''
        raise Exception('Not implemented yet')

    def pFunc(self, r, phi):
        '''
        The P(r, φ) function for the PSF
        '''
        raise Exception('Not implemented yet')

    def wFunc(self, r, phi):
        '''
        The W(r, φ) function to simulate mirror imperfections
        '''
