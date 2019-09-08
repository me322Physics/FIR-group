# Commands to select directory based on hostname
#from socket import gethostname

#if gethostname() == 'colonsay':
#    path_start = '/disk1/rohitk/ELN1_project/'
#elif gethostname() == 'rohitk-elitebook':
#    path_start = '/home/rohitk/Documents/PhD/Year1/ELN1_project/'

#################################################
# Add the path of useful functions at the start
import sys
#sys.path.append(path_start+'basic_functions')
from useful_functions import return_hist_par, varstat, latest_dir, jytoabmag, field_filter
from plot_func import rc_def, make_fig, make_fig_multi
rc_def()
##################################################

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob

# For catalog matching
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import search_around_sky

from astropy.table import Table
import os
import pickle
from scipy.stats import binned_statistic_2d

#sys.path.append("/disk1/rohitk/ELN1_project/OCT17_ELAIS_im/maxl_test/elaisn1_lr/")
from mlrun1_bootes import get_n_m_kde, survey_area
from useful_functions import get_center

import seaborn as sns
##################################################


def aper_corr(magnitude_base, aperture_magnitudes):
    """
    Compute the median aperture magnitude corrections given base magnitudes and aperture magnitudes
    """
    correction_factors = 10 ** ((magnitude_base - aperture_magnitudes.T) / 2.5)
    med_cfs = np.nanmedian(correction_factors, axis=1)
    std_cfs = np.nanstd(correction_factors, axis=1)

    return med_cfs, std_cfs


def indx_to_bool(array_of_indices, array_length):
    """
    Convert an array_of_indices into a boolean array of length array_length
    """
    bool_array = np.zeros(array_length, dtype=bool)
    bool_array[array_of_indices] = True
    return bool_array


# Load in the aperture corrected catalogue
MASTER_PATH = "data/edited_cats/optical/LH_MASTER_opt_spitzer_merged_cedit_apcorr.fits"
cata_phot = Table.read(MASTER_PATH)

# Load int he GAIA catalogue and select stars
# Read in the GAIA DR2 catalogue
cata_gaia = Table.read("data/gaia_dr2_lh.fit")
imag_col = "Gmag"

# Cross-match with GAIA stars that are reasonably bright (but won't be saturated in chi2 catalogue)
mag_range = (cata_gaia[imag_col] > 18) & (cata_gaia[imag_col] < 20.)
cata_gaia = cata_gaia[mag_range]

# Cross match the two coordinates
gaia_coords = SkyCoord(cata_gaia["_RAJ2000"], cata_gaia["_DEJ2000"], unit='deg', frame='icrs')
K_coord_all = SkyCoord(cata_phot["ALPHA_J2000"], cata_phot["DELTA_J2000"], unit='deg', frame='icrs')

# Cross match the two catalogues - for each Bootes source, find the nearest Gaia match
ind_gaia, sep_g, _ = match_coordinates_sky(K_coord_all, gaia_coords, nthneighbor=1)

slim = 0.5

# Do a NN selection on the chi2 catalogue
# ind_chi2, sep_chi2, _ = match_coordinates_sky(K_coord_all, K_coord_all, nthneighbor=2)
cal_ind = (sep_g.arcsec <= slim)  # & (sep_chi2.arcsec > 10.)  # & (cata_phot["FLAGS"] != 4)
print("No. of stars used for plots: {0}".format(np.sum(cal_ind)))

# Reduce the chi2 catalogue to only stars
cata_phot = cata_phot[cal_ind]

se_coords = SkyCoord(cata_phot["ALPHA_J2000"], cata_phot["DELTA_J2000"], unit='deg', frame='icrs')

# Load in the SDSS catalogue
PATH_SDSS = "data/sdss_dr12_lh_psfmag.fit"
cata_sdss = Table.read(PATH_SDSS)
sdss_coords = SkyCoord(cata_sdss["_RAJ2000"], cata_sdss["_DEJ2000"], unit='deg', frame='icrs')

# Do the cross match
ind_sdss, ind_se, sep2d, _ = search_around_sky(sdss_coords, se_coords, seplimit=0.2*u.arcsec)

"""
p_ra_down = 241.5
p_dec_down = 53.8
p_ra_up = 244.
p_dec_up = 56.3

en1_area = survey_area(p_ra_up, p_dec_up, p_ra_down, p_dec_down)
print("chi2 catalogue area: {0} sq. deg.".format(en1_area / 3600**2))

phot_rect = field_filter(p_ra_up, p_dec_up, p_ra_down, p_dec_down, cata_phot["ALPHA_J2000"], cata_phot["DELTA_J2000"])

# With the psf-mag catalogue
p_ra_down = 240.04
p_dec_down = 51.07675900
p_ra_up = 244.5
p_dec_up = 53.98

sdss_area = survey_area(p_ra_up, p_dec_up, p_ra_down, p_dec_down)
print("SDSS area: {0} sq. deg.".format(sdss_area / 3600**2))
sdss_rect = field_filter(p_ra_up, p_dec_up, p_ra_down, p_dec_down, cata_sdss["RA_ICRS"], cata_sdss["DE_ICRS"])
"""

plot_ind = np.arange(0, len(ind_sdss), 50)

# Vega-AB correction factors
mvega = dict()
mvega["K"] = 1.9
mvega["J"] = 0.938
mvega["i"] = 0
mvega["g"] = 0
mvega["r"] = 0
mvega["z"] = 0
mvega["u"] = 0

ap_size = np.array([1., 2., 3., 4., 5., 6., 7., 10.])
ap_ind = np.arange(len(ap_size))

# The photometry to compare
phot_band = str(input("Filter to make the plots for: "))
phot_filter = phot_band.lower()


# Define some constants here
ap_size = np.array([1., 2., 3., 4., 5., 6., 7., 10.])
ap_ind = np.arange(len(ap_size))

# Aperture to use for plots
ap_to_use = 3  # 3'' aperture magnitude
ap_ind_to_use = int(ap_ind[np.where(ap_size == ap_to_use)])
ap = str(int(ap_size[np.where(ap_size == ap_to_use)]))
ap = ap_to_use

null_mag = -99


for phot_band in phot_filter:
    print("##### " + phot_band + " #####")

    # Aperture column to use from the SDSS catalogue
    """
    # ******* Note *******
    # Use either
    # 	1. cata_col = phot_band + "mag" for model magnitudes or
    # 	2. cata_col = phot_band + "pmag" for PSF magnitudes (better for comparing stars)
    """
    cata_col =  phot_band[0] + 'mag'
    se_mcol = "MAG_APER_{0}_{1}".format(phot_band, ap)

    # Make a couple of plots of the aperture corrected magnitudes and the median differences
    phot_mag = cata_phot[se_mcol][ind_se]
    sdss_mag = np.copy(cata_sdss[cata_col][ind_sdss] + mvega[phot_band])

    print("## Statistics for SDSS ({0}) - SExtractor ({1}) for {2}'' aperture ##".format(phot_band, phot_band, ap))
    no_99 = (cata_phot[se_mcol][ind_se] != null_mag) & (~np.isnan(cata_sdss[ind_sdss][cata_col]))
    mdiff = (sdss_mag[no_99] - phot_mag[no_99])
    # mdiff = phot_mag - sdss_mag
    varstat(mdiff)

    ra_xmatch = cata_phot["ALPHA_J2000"][ind_se]
    dec_xmatch = cata_phot["DELTA_J2000"][ind_se]

    fig, ax = make_fig()
    plt.scatter(mdiff, ra_xmatch[no_99], s=2, alpha=0.1)
    plt.xlabel("SDSS - SE {0}-band".format(phot_band))
    plt.ylabel(r"$RA$")
    plt.xlim([-0.5, 0.5])
    
    fig, ax = make_fig()
    plt.scatter(mdiff, dec_xmatch[no_99], s=2, alpha=0.1)
    plt.xlabel("SDSS - SE {0}-band".format(phot_band))
    plt.ylabel(r"$DEC$")
    plt.xlim([-0.5, 0.5])
    

    #plot the colour difference on their positions
    fig, ax = make_fig()
    mask = (mdiff < 0.75) & (mdiff>-0.75)
    plt.scatter(ra_xmatch[no_99][mask], dec_xmatch[no_99][mask],c=mdiff[mask],norm=LogNorm(), s=20)
    
    plt.xlabel('RA')
    plt.ylabel("dec")
    plt.title("SDSS - SE {0}-band".format(phot_band))
    
    fig, ax = make_fig()
    ra_bin = np.arange(157,164.5,0.1)
    dec_bin = np.arange(56.5,60.5,0.1)
    mean_mdiff,_,_,_ = binned_statistic_2d(ra_xmatch[no_99][mask], dec_xmatch[no_99][mask],mdiff[mask],statistic='mean',bins=[ra_bin,dec_bin])
    X,Y = np.meshgrid(get_center(ra_bin),get_center(dec_bin))
    #plt.pcolormesh(get_center(ra_bin),get_center(dec_bin),mean_mdiff)
    plt.imshow(mean_mdiff)
    plt.xlabel('RA')
    plt.ylabel("dec")
    plt.title("SDSS - SE {0}-band".format(phot_band))
    
    

    # del ra_xmatch

    # Make a plot of the one-to-one aperture corrected magnitudes 
    fig, ax = make_fig()
    plt.scatter(phot_mag[plot_ind], sdss_mag[plot_ind], s=5, alpha=0.2, color='magenta')
    plt.plot([15., 30.], [15., 30.], '-', color='black', lw=1., ls=':')
    plt.xlabel(r"$mag_{SE}$")
    plt.ylabel(r"$mag_{SDSS}$")
    plt.xlim([15., 20.])
    plt.ylim([15., 20.])
    plt.title(phot_band)
    plt.tight_layout()

    # Make a plot of the magniture difference binned by magnitudes
    m_n, m_e, m_c = return_hist_par(0.1, phot_mag[phot_mag <= 35.])

    mdiff_med = []
    col_term_med = []
    for k in range(len(m_c)):
        within_bin = (phot_mag[no_99] >= m_e[k]) & (phot_mag[no_99] < m_e[k+1])
        mdiff_med.append(np.nanmedian(np.copy(mdiff[within_bin])))

    mdiff_med = np.array(mdiff_med)

    fig, ax = make_fig()
    plt.plot(m_c, mdiff_med, 'r.', markersize=2.5)
    plt.axhline(color='black', linestyle='--', lw=1.)
    plt.xlabel(r"$mag_{SE}$")
    plt.ylabel(r"$MED[mag_{SDSS}\ -\ mag_{SE}]$")
    plt.ylim([-1, 1])
    plt.title(phot_band)
    plt.tight_layout()

    # Bin the magnitude difference
    _, md_e, _ = return_hist_par(0.005, mdiff[sdss_mag[no_99] > 0])

    fig, ax = make_fig()
    plt.hist(mdiff, bins=md_e, histtype='step', lw=0.8)
    plt.xlabel(r"$mag_{SDSS}\ -\ mag_{SE}$")
    plt.xlim([-1, 1])
    plt.title(phot_band)
    plt.tight_layout()

    # Plot the magnitude difference vs magnitude

    fig, ax = make_fig()
    # plt.scatter(phot_mag[no_99], mdiff, 'o', s=5, alpha=0.2, markerfacecolor='None', markeredgecolor='blue')
    plt.scatter(phot_mag[no_99], mdiff, marker='o', s=5, alpha=0.2, facecolor='None', edgecolor='blue')
    plt.axhline(color='red', linestyle='--', lw=1.)
    plt.xlabel(r"$mag_{SE}$")
    plt.ylabel(r"$mag_{SDSS}\ -\ mag_{SE}$")
    plt.ylim([-1, 1])
    plt.title(phot_band)
    # plt.tight_layout()

    # Plot a hexbin magnitude difference vs magnitude
    """
    phot_mag.name = "None"
    mdiff.name = "None"
    h = (sns.jointplot(x=phot_mag[no_99], y=mdiff, kind='hex', joint_kws=dict(gridsize=(150,200)), xlim=(15,25), ylim=(-1.5,1.5))
         .set_axis_labels(r"$mag_{SE}$", r"$mag_{SDSS}\ -\ mag_{SE}$"))
    """

plt.show()
