# -*- coding: utf-8 -*-

# Main document
import matplotlib.pyplot as plt
import numpy as np

from .simplepf import SimplePupilFunction
from .cassegrainpf import CassegrainPupilFunction

# Globals
PI     = np.pi
TWO_PI = 2.*PI

pupil  = SimplePupilFunction(diameter=50, samples=250)
caspup = CassegrainPupilFunction(diameter=50, b=21, samples=500)
red    = TWO_PI / 500e-7

# Memory management
def terminate():
    global pupil
    pupil = None

    global caspup
    caspup = None

# Interactivity
def plot_simplepupil():
    X, Y = pupil.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, pupil.render(red))
    ax.set_aspect('equal')

def plot_cassepupil():
    X, Y = caspup.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, caspup.render(red))
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

def plot_cassepsf():
    kx = np.linspace(-caspup.nyqfreq() / 2., caspup.nyqfreq() / 2., caspup.samples)
    ky = np.linspace(-caspup.nyqfreq() / 2., caspup.nyqfreq() / 2., caspup.samples)

    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    # Filter and rescale properly
    psf = caspup.psf(red)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.contourf(KX, KY, psf, levels=np.linspace(0, 1, 50))
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
