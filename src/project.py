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

pupil  = SimplePupilFunction(diameter=50, samples=256, padscale=1.2)
caspup = CassegrainPupilFunction(diameter=50, b=20, samples=256, padscale=1.2)
square = SquarePupilFunction(diameter=50, samples=256, padscale=1.2)
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
    ax.contourf(X, Y, pupil.render(red, filtering=True))
    ax.set_aspect('equal')

def plot_cassepupil():
    X, Y = caspup.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, caspup.render(red, filtering=True))
    ax.set_aspect('equal')

def plot_squarepupil():
    X, Y = square.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, square.render(red, filtering=True))
    ax.set_aspect('equal')

def plot_simplepsf():
    # Filter and rescale properly
    psf = pupil.psf(red, filtering=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf, interpolation='bessel')
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_cassepsf():
    # Filter and rescale properly
    psf = caspup.psf(red, filtering=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf, interpolation='bessel')
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_squarepsf():
    # Filter and rescale properly
    psf = square.psf(red, filtering=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf)
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_gauss():
    # check the random field is working
    plt.figure()
    plt.imshow(gauss.randomfield().real, interpolation='none')
