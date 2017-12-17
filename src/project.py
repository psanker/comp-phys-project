# -*- coding: utf-8 -*-

# Main document
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fftshift, fftfreq

from .simplepf import SimplePupilFunction
from .simplegausspf import DirtySimplePupilFunction
from .cassegrainpf import CassegrainPupilFunction
from .cassegausspf import DirtyCassegrainPupilFunction
from .squarepf import SquarePupilFunction
from .gaussianrndf import GaussianRandomField
from .modelpf import ModelPupilFunction

#### Globals ####
PI     = np.pi
TWO_PI = 2. * PI
ps     = 2.  # padding scale factor
N_samples = 2048

k_blue  = TWO_PI / (400e-9)
k_red   = TWO_PI / (700e-9)
k_green = TWO_PI / (550e-9)

# DIAMETERS ARE IN METERS
pupil  = SimplePupilFunction(diameter=6.5, samples=N_samples, padscale=ps)
dirty  = DirtySimplePupilFunction(diameter=6.5, samples=N_samples, padscale=ps)
caspup = CassegrainPupilFunction(diameter=6.5, b=1.5, samples=N_samples, padscale=ps)
dcaspf = DirtyCassegrainPupilFunction(diameter=6.5, b=1.5, samples=N_samples, padscale=ps)
square = SquarePupilFunction(diameter=6.5, samples=N_samples, padscale=ps)
model  = ModelPupilFunction(diameter=.250, b=.110, samples=N_samples, padscale=ps)
gauss  = GaussianRandomField(pupil)

#### Memory management ####
def terminate():
    global pupil
    pupil = None

    global caspup
    caspup = None

    global square
    square = None

    global dirty
    dirty = None

    global dcaspf
    dcaspf = None

#### Drawing logic ####
def render_pupil(pupilFunc, k=TWO_PI, color=None, filtering=True):
    '''
    pupilFunc: The complex Pupil Function to analyze
    k: Wavenumber of light (2π / λ)
    '''
    s     = pupilFunc.padscale * pupilFunc.diameter   # Scaling based on padding and diameter
    s_map = [-s, s, s, -s]                            # Because imshow is oriented top-left, remap s extrema

    fig, ax = plt.subplots()

    render = pupilFunc.render(k, filtering=filtering) # Raw render information, after applying Gaussian filter
    amp2   = render.real**2 + render.imag**2          # Purely amplitude information for image output

    if color is not None:
        ax.imshow(np.sqrt(amp2), cmap=plt.get_cmap(color), extent=s_map)
    else:
        ax.imshow(np.sqrt(amp2), extent=s_map)

    ax.set_aspect('equal')
    ax.set_xlabel('$x$ ($m$)')
    ax.set_ylabel('$y$ ($m$)')

def render_psf(pupilFunc, k=TWO_PI, color=None, noshift=False, filtering=True):
    '''
    pupilFunc: The complex Pupil Function to analyze
    k: Wavenumber of light (2π / λ)
    '''
    # PSF
    psf = pupilFunc.psf(k=k, filtering=filtering, noshift=noshift)  # Generate PSF
    psf = psf / np.amax(psf)                                   # Rescale output so max value is 1. (luminance)
    psf = np.where(psf > 1e-15, psf, 0.)                       # Filter very low values

    # Relevant k's
    spacing = 1. / (2. * pupilFunc.nyqfreq())                  # Spacing from Nyquist frequency
    k       = fftshift(fftfreq(pupilFunc.samples, d=spacing))  # The actual k's
    k_map   = [k[0], k[-1], k[-1], k[0]]                       # Because imshow is oriented top-left, remap k extrema

    # Draw
    fig, ax = plt.subplots()

    if color is not None:
        ax.imshow(psf, cmap=plt.get_cmap(color), extent=k_map)
    else:
        ax.imshow(psf, extent=k_map)

    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$ ($m^{-1}$)')
    ax.set_ylabel('$k_y$ ($m^{-1}$)')

def render_psf_range(pupilFunc, k, color=None, noshift=False, filtering=True):
    '''
    pupilFunc: The complex Pupil Function to analyze
    k: Wavenumber of light (2π / λ)
    '''

    # PSF
    psf = None

    for kval in k:
        if psf is None:
            psf = pupilFunc.psf(k=kval, filtering=filtering, noshift=noshift) # Generate PSF
            psf = psf / np.amax(psf)                                          # Rescale output so max value is 1. (luminance)
            psf = np.where(psf > 1e-15, psf, 0.)                              # Filter very low values
        else:
            s   = pupilFunc.psf(k=kval, filtering=filtering, noshift=noshift) # Generate PSF
            s   = s / np.amax(s)
            s   = np.where(psf > 1e-15, psf, 0.)
            psf += s

    psf = psf / np.amax(psf)

    # Relevant k's
    spacing = 1. / (2. * pupilFunc.nyqfreq())                  # Spacing from Nyquist frequency
    k       = fftshift(fftfreq(pupilFunc.samples, d=spacing))  # The actual k's
    k_map   = [k[0], k[-1], k[-1], k[0]]                       # Because imshow is oriented top-left, remap k extrema

    # Draw
    fig, ax = plt.subplots()

    if color is not None:
        ax.imshow(psf, cmap=plt.get_cmap(color), extent=k_map)
    else:
        ax.imshow(psf, extent=k_map)

    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$ ($m^{-1}$)')
    ax.set_ylabel('$k_y$ ($m^{-1}$)')

#### Interactivity ####
# Pupils
def plot_simplepupil():
    render_pupil(pupil, color='gray')

def plot_cassepupil():
    render_pupil(caspup, color='gray')

def plot_squarepupil():
    render_pupil(square, color='gray')

def plot_gsimplepupil():
    render_pupil(dirty, k=k_green, color='gray')

def plot_modelpupil():
    render_pupil(model, k=k_green, color='gray')

# PSFs
def plot_simplepsf():
    render_psf(pupil, color='magma')

def plot_gsimplepsf():
    render_psf(dirty, color='magma')

def plot_cassepsf():
    render_psf(caspup, color='magma')

def plot_gcassepsf():
    render_psf(dcaspf)

def plot_squarepsf():
    render_psf(square)

def plot_modelpsfr():
    render_psf(model, color='magma', k=k_red)

def plot_modelpsfg():
    render_psf(model, color='magma', k=k_green)

def plot_modelpsfb():
    render_psf(model, color='magma', k=k_blue)

def plot_modelpsfcomp():
    render_psf_range(model, [k_red, k_green, k_blue], color='magma')

# Misc
def plot_gauss ():
    # check the random field is working
    alpha = -2.

    def test_power_spec(kx, ky):
        k = np.sqrt(kx**2. + ky**2.)

        ret = np.zeros((N_samples, N_samples))
        ret = np.where(k > 1e-15, k**(alpha), 1)

        ret = TWO_PI*(ret / np.amax(ret))

        return ret

    field = gauss.randomfield(test_power_spec)
    field = field / np.amax(field)

    plt.figure()
    plt.imshow(np.sqrt(field.real**2 + field.imag**2), interpolation='none', cmap=plt.get_cmap('bone'))
    plt.xlabel('$m^{\\alpha}$')
    plt.ylabel('$m^{\\alpha}$')
    plt.title('Gaussian Random Field ($\\alpha=%d$)' % (alpha))
# It do, but the divide by zero is weird. Even filtered out the 0 vals in the lambda exp

def plot_gaussatm():
    # check the random field is working
    field = gauss.randomfield(ModelPupilFunction.atm_Pk)
    field = field / np.amax(field)

    plt.figure()
    plt.imshow(np.sqrt(field.real**2 + field.imag**2), interpolation='none', cmap=plt.get_cmap('bone'))
    plt.xlabel('$m^{\\alpha}$')
    plt.ylabel('$m^{\\alpha}$')
    plt.title('Gaussian Random Field')

def test_power_spec(kx, ky, power=-2.):
    k = np.sqrt(kx**2. + ky**2.)

    ret = np.zeros((N_samples, N_samples))
    ret = np.where(k > 1e-15, k**(power), 1)

    ret = TWO_PI*(ret / np.amax(ret))

    return ret
