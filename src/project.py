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

# Globals
PI     = np.pi
TWO_PI = 2.*PI

pupil  = SimplePupilFunction(diameter=50, samples=256, padscale=1.)
dirty  = DirtySimplePupilFunction(diameter=50, samples=256, padscale=1.)
caspup = CassegrainPupilFunction(diameter=50, b=20, samples=256, padscale=1.)
dcaspf = DirtyCassegrainPupilFunction(diameter=50, b=20, samples=256, padscale=1.)
square = SquarePupilFunction(diameter=50, samples=256, padscale=1.)
gauss  = GaussianRandomField(pupil, lambda s : np.where(s > 1e-15, (1. / s)**(3.0), 0.))

# Memory management
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

# Interactivity
def plot_simplepupil():
    X, Y = pupil.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, pupil.render(filtering=True))
    ax.set_aspect('equal')

def plot_cassepupil():
    X, Y = caspup.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, caspup.render(filtering=True))
    ax.set_aspect('equal')

def plot_squarepupil():
    X, Y = square.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, square.render(filtering=True))
    ax.set_aspect('equal')

def plot_simplepsf():
    # Filter and rescale properly
    psf = pupil.psf(filtering=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf, interpolation='bessel')
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_gsimplepsf():
    # Filter and rescale properly
    psf = dirty.psf(filtering=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf, interpolation='bessel')
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_cassepsf():
    # Filter and rescale properly
    psf = caspup.psf(filtering=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf, interpolation='bessel')
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_gcassepsf():
    # Filter and rescale properly
    psf = dcaspf.psf(k = TWO_PI / 20e-2, filtering=True)
    psf = psf / np.amax(psf)
    psf = np.where(psf > 1e-15, psf, 0.)

    fig, ax = plt.subplots()
    ax.imshow(psf, interpolation='bessel', cmap=plt.get_cmap('bone'))
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def plot_squarepsf():
    # Filter and rescale properly
    psf = square.psf(filtering=True)
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
    plt.imshow(gauss.randomfield().real, interpolation='none', cmap=plt.get_cmap('bone'))
    # It do, but the divide by zero is weird. Even filtered out the 0 vals in the lambda exp
