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

#### Globals ####
PI     = np.pi
TWO_PI = 2.*PI

pupil  = SimplePupilFunction(diameter=6., samples=128, padscale=1.1)
dirty  = DirtySimplePupilFunction(diameter=50, samples=256, padscale=1.1)
caspup = CassegrainPupilFunction(diameter=6.5, b=2., samples=256, padscale=1.1)
dcaspf = DirtyCassegrainPupilFunction(diameter=50, b=20, samples=256, padscale=1.1)
square = SquarePupilFunction(diameter=50, samples=256, padscale=1.1)
gauss  = GaussianRandomField(pupil, lambda s : np.where(s > 1e-15, (1. / s)**(3.0), 0.))

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
def render_pupil(pupilFunc, k = TWO_PI, color=None):
    '''
    pupilFunc: The complex Pupil Function to analyze
    k: Wavenumber of light (2π / λ)
    '''
    s     = pupilFunc.padscale * pupilFunc.diameter
    s_map = [-s, s, s, -s]

    fig, ax = plt.subplots()

    if color is not None:
        ax.imshow(pupilFunc.render(k, filtering=True).real, cmap=plt.get_cmap(color), extent=s_map)
    else:
        ax.imshow(pupilFunc.render(k, filtering=True).real, extent=s_map)

    ax.set_aspect('equal')

def render_psf(pupilFunc, k = TWO_PI, color=None):
    '''
    pupilFunc: The complex Pupil Function to analyze
    k: Wavenumber of light (2π / λ)
    '''
    # PSF
    psf = pupilFunc.psf(filtering=True)  # Generate PSF
    psf = psf / np.amax(psf)             # Rescale output so max value is 1. (luminance)
    psf = np.where(psf > 1e-15, psf, 0.) # Filter very low values

    # Relevant k's
    spacing = 1. / (2. * pupilFunc.nyqfreq())
    k       = fftshift(fftfreq(pupilFunc.samples, d=spacing))
    k_map   = [k[0], k[-1], k[-1], k[0]]

    # Draw
    fig, ax = plt.subplots()

    if color is not None:
        ax.imshow(psf, cmap=plt.get_cmap(color), extent=k_map)
    else:
        ax.imshow(psf, extent=k_map)

    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

#### Interactivity ####
# Pupils
def plot_simplepupil():
    render_pupil(pupil)

def plot_cassepupil():
    render_pupil(caspup)

def plot_squarepupil():
    render_pupil(square)

def plot_gsimplepupil():
    render_pupil(dirty, k = TWO_PI / 20e-2, color='gray')

# PSFs
def plot_simplepsf():
    render_psf(pupil)

def plot_gsimplepsf():
    render_psf(dirty)

def plot_cassepsf():
    render_psf(caspup)

def plot_gcassepsf():
    render_psf(dcaspf)

def plot_squarepsf():
    render_psf(square)

# Misc
def plot_gauss():
    # check the random field is working
    plt.figure()
    plt.imshow(gauss.randomfield().real, interpolation='none', cmap=plt.get_cmap('bone'))
    # It do, but the divide by zero is weird. Even filtered out the 0 vals in the lambda exp
