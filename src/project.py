# -*- coding: utf-8 -*-

# Main document
import matplotlib.pyplot as plt
import numpy as np

from .simplepf import SimplePupilFunction
from .simplegausspf import DirtySimplePupilFunction
from .cassegrainpf import CassegrainPupilFunction
from .cassegausspf import DirtyCassegrainPupilFunction
from .squarepf import SquarePupilFunction
from .gaussianrndf import GaussianRandomField

#### Globals ####
PI     = np.pi
TWO_PI = 2.*PI

pupil  = SimplePupilFunction(diameter=1, samples=256, padscale=1.)
dirty  = DirtySimplePupilFunction(diameter=50, samples=256, padscale=1.)
caspup = CassegrainPupilFunction(diameter=50, b=20, samples=256, padscale=1.)
dcaspf = DirtyCassegrainPupilFunction(diameter=50, b=20, samples=256, padscale=1.)
square = SquarePupilFunction(diameter=50, samples=256, padscale=1.)
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
def render_pupil(pupilFunc, k = TWO_PI):
    '''
    pupilFunc: The complex Pupil Function to analyze
    k: Wavenumber of light (2π / λ)
    '''
    X, Y = pupilFunc.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, pupilFunc.render(k, filtering=True))
    ax.set_aspect('equal')

def render_psf(pupilFunc, k = TWO_PI):
    '''
    pupilFunc: The complex Pupil Function to analyze
    k: Wavenumber of light (2π / λ)
    '''
    # Generate PSF
    psf = pupilFunc.psf(filtering=True)

    # Rescale output so max value is 1. (luminance)
    psf = psf / np.amax(psf)

    # Filter very low values
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf)
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
    render_pupil(dirty, k = TWO_PI / 20e-2)

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
