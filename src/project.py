# -*- coding: utf-8 -*-

# Main document
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import ifft2

from .simplepf import SimplePupilFunction

# Globals
PI     = np.pi
TWO_PI = 2.*PI

pupil = SimplePupilFunction(diameter=50, samples=250)
red   = TWO_PI / 500e-7

# Interactivity
def terminate():
    global pupil
    pupil = None

def plot_simplepupil():
    X, Y = pupil.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, pupil.render(red))
    ax.set_aspect('equal')

def plot_simplepsf():
    kx = np.linspace(-pupil.nyqfreq() / 2., pupil.nyqfreq() / 2., pupil.samples)
    ky = np.linspace(-pupil.nyqfreq() / 2., pupil.nyqfreq() / 2., pupil.samples)

    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    # Filter and rescale properly
    psf = pupil.psf(red)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.contourf(KX, KY, psf, levels=np.linspace(0, 1, 50))
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
