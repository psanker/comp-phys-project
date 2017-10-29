# -*- coding: utf-8 -*-

# Main document

import matplotlib.pyplot as plt
import numpy as np

from .simplepf import SimplePupilFunction

pupil = SimplePupilFunction(diameter=20)

PI     = np.pi
TWO_PI = 2.*PI

def plot_simplepupil():
    red = TWO_PI / 500e-7
    X, Y = pupil.configurationMesh()

    fig, ax = plt.subplots()
    ax.contourf(X, Y, pupil.render(red))
    ax.set_aspect('equal')

def plot_simplepsf():
    red = TWO_PI / 500e-7

    kx = np.linspace(0, pupil.nyqfreq, pupil.samples)
    ky = np.linspace(0, pupil.nyqfreq, pupil.samples)

    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    psf = pupil.psf(red)

    fig, ax = plt.subplots()
    ax.contourf(KX, KY, psf, levels=np.linspace(0, np.amax(psf), 100))
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
