"""Hessian properties."""

from collections.abc import Sequence

import numpy as numpy
from qcelemental import constants as qcc

from phydat import ptab

MatrixLike = Sequence[Sequence[float]] | numpy.ndarray


def vibrations(hess: MatrixLike, symbs: Sequence[str]):
    """Get vibrational modes and frequencies from the Hessian matrix.

    :param hess: The Hessian matrix
    :param symbs: The atomic symbols
    :return: The vibrational frequencies (in cm-1) and normal modes
    """
    # 1. build mass-weighted Hessian matrix
    mw_vec = numpy.repeat(list(map(ptab.to_mass, symbs)), 3) ** -0.5
    hess_mw = mw_vec[:, numpy.newaxis] * hess * mw_vec[numpy.newaxis, :]

    # 2. compute eigenvalues and eigenvectors of the mass-weighted Hessian matrix
    eig_vals, eig_vecs = numpy.linalg.eigh(hess_mw)

    # 3. un-mass-weight the normal coordinates
    norm_coos = mw_vec[:, numpy.newaxis] * eig_vecs

    # 4. get wavenumbers from a.u. force constants
    har2J = qcc.conversion_factor("hartree", "J")
    amu2kg = qcc.conversion_factor("atomic_mass_unit", "kg")
    bohr2m = qcc.conversion_factor("bohr", "meter")
    sol = qcc.get("speed of light in vacuum") * 100  # in cm / s
    to_inv_cm = numpy.sqrt(har2J / (amu2kg * bohr2m * bohr2m)) / (sol * 2 * numpy.pi)
    freqs = numpy.sqrt(numpy.complex_(eig_vals)) * to_inv_cm
    freqs = tuple(map(float, numpy.real(freqs) - numpy.imag(freqs)))

    return freqs, norm_coos
