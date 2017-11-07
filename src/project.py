# -*- coding: utf-8 -*-

# Main document
import matplotlib.pyplot as plt
import numpy as np

from .simplepf import SimplePupilFunction
from .cassegrainpf import CassegrainPupilFunction
from .squarepf import SquarePupilFunction
from .gaussianrndf import GaussianRandomField

# Globals
PI     = np.pi
TWO_PI = 2.*PI

pupil  = SimplePupilFunction(diameter=50, samples=256)
caspup = CassegrainPupilFunction(diameter=50, b=21, samples=256)
square = SquarePupilFunction(diameter=50, samples=256)
red    = TWO_PI / 500e-7
Pk2 = lambda s : s**(-3.0)
gauss = GaussianRandomField(pupil, Pk2)

# Memory management
def terminate():
    global pupil
    pupil = None

    global caspup
    caspup = None

    global square
    square = None

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

def plot_squarepupil():
    X, Y = square.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, square.render(red))
    ax.set_aspect('equal')

def plot_simplepsf():
    kx = np.linspace(-pupil.nyqfreq() / 2., pupil.nyqfreq() / 2., pupil.samples)
    ky = np.linspace(-pupil.nyqfreq() / 2., pupil.nyqfreq() / 2., pupil.samples)

    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    # Filter and rescale properly
    psf = pupil.psf(red, padding=1.6, force=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.contourf(KX, KY, psf, levels=np.linspace(0, 1, 50), interpolation='nearest')
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

def plot_squarepsf():
    # Filter and rescale properly
    psf = square.psf(red, force=True, padding=2.)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    kx = np.linspace(-square.nyqfreq() / 2., square.nyqfreq() / 2., square.samples)
    ky = np.linspace(-square.nyqfreq() / 2., square.nyqfreq() / 2., square.samples)

    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    fig, ax = plt.subplots()
    ax.contourf(KX, KY, psf, levels=np.linspace(0, 1, 50))
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_gauss():
    # check the random field is working
    plt.figure()
    plt.imshow(gauss.randomfield().real, interpolation='none')
