import pymoc.io.fits
import healpy as hp
import pymoc.util.catalog
import numpy as np
##################################################


def coords_to_hpidx(ra, dec, order):
    """Convert coordinates to HEALPix indexes
    Given to list of right ascension and declination, this function computes
    the HEALPix index (in nested scheme) at each position, at the given order.
    Parameters
    ----------
    ra: array or list of floats
        The right ascensions of the sources.
    dec: array or list of floats
        The declinations of the sources.
    order: int
        HEALPix order.
    Returns
    -------
    array of int
        The HEALPix index at each position.
    """
    ra, dec = np.array(ra), np.array(dec)

    theta = 0.5 * np.pi - np.radians(dec)
    phi = np.radians(ra)
    healpix_idx = hp.ang2pix(2**order, theta, phi, nest=True)

    return healpix_idx


def inMoc(ra, dec, moc):
    """Find source position in a MOC
    Given a list of positions and a Multi Order Coverage (MOC) map, this
    function return a boolean mask with True for sources that fall inside the
    MOC and False elsewhere.
    Parameters
    ----------
    ra: array or list of floats
        The right ascensions of the sources.
    dec: array or list of floats
        The declinations of the sources.
    moc: pymoc.MOC
        The MOC read by pymoc
    Returns
    -------
    array of booleans
        The boolean mask with True for sources that fall inside the MOC.
    """
    source_healpix_cells = coords_to_hpidx(np.array(ra), np.array(dec), moc.order)

    # Array of all the HEALpix cell ids of the MOC at its maximum order.
    moc_healpix_cells = np.array(list(moc.flattened()))

    # We look for sources that are in the MOC and return the mask
    return np.in1d(source_healpix_cells, moc_healpix_cells)

"""
Useage for filtering catalogue using a MOC

import pymoc
import pymoc.io.fits
# Read in the MOC
moc_opt = pymoc.MOC()
pymoc.io.fits.read_moc(moc_opt, "/path/to/optical/moc.fits")

# Filter catalogue using
bool_inmoc = inMoc(array_of_ra, array_of_dec, moc_opt)

# bool_inmoc is a boolean array of len(array_of_ra) indicating if a source is within moc_opt or not
"""
