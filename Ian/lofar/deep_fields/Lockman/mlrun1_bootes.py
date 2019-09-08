from socket import gethostname

if gethostname() == 'colonsay':
    PATH_START = '/disk1/rohitk/'

elif gethostname() == 'rohitk-elitebook':
    PATH_START = '/home/rohitk/Documents/PhD/Year1/ELAIS-N1/'

# Path start for Max_L analysis
#PATH_START = PATH_START + "OCT17_ELAIS_im/maxl_test/"
    
#################################################
# Add the path of useful functions at the start

import sys
#sys.path.append(PATH_START)
# Import some important functions
#sys.path.append(PATH_START+'../../basic_functions')

# Import coordinate converstion functions
from useful_functions import (coord_matching, nearest_neigh_match)
#from overlapping_area import (isinpan, isinukidss, isinSWIRE, isinSERVS)

from astropy.table import Table
from matplotlib import pyplot as plt

import numpy as np
from astropy.coordinates import search_around_sky
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky

#from moc_util import coords_to_hpidx, inMoc
from moc_util import coords_to_hpidx, inMoc

from sklearn.neighbors import KernelDensity

from numpy.linalg import det
###############################
DEG_TO_ARCSEC = 3600.


# Function to display basic statistics about an array/variable
def varstat(distribution):
    """
    Print basic properties about a variable to stdout
    
    
    Parameters:
    ------------
    
    distribution : The input variable/distribution to display the statistic of to stdout
    
    Returns:
    -----------
    """
    
    stat_to_print = [np.nanmean(distribution),np.nanmedian(distribution),
                     np.nanstd(distribution),len(distribution),np.nanmin(distribution),
                     np.nanmax(distribution),len(distribution[distribution==0.])]
    
    var_to_print = ["Mean", "Median", "Std. Dev.", "Length", "Min", "Max", "Len_Zeros"]
    var_to_print = [aa.ljust(10) for aa in var_to_print]
    print(" ".join(var_to_print))
    
    #stat_to_print = map(str,stat_to_print)
    #print(stat_to_print)
    stat_to_print = [str(np.round(bb,6)).ljust(10) for bb in stat_to_print]
    stat_to_print[4] = str(np.nanmin(distribution))
    print(" ".join(stat_to_print))
    # print(stat_to_print)
    return


# Function to calculate the survey area based on two sets of coordinates

def survey_area(ra_up, dec_up, ra_down, dec_down):
    """
    Compute the area of the survey (rectangular area)
    
    Parameters:
    -----------
    ra_up : Upper corner RA
    dec_up : Upper corner Dec
    ra_down : Lower corner RA
    dec_down : Lower corner Dec
    
    Returns:
    --------
    area : Survey area in arcsec
    """
    
    return ((np.deg2rad(ra_up) - np.deg2rad(ra_down)) 
            * (np.sin(np.deg2rad(dec_up)) - np.sin(np.deg2rad(dec_down)))
            * np.rad2deg(DEG_TO_ARCSEC)**2)


################################################################
#                    Error Functions                           #
################################################################

# Function to compute and average sigma and sigma components in the major and minor axis

def get_sigma_all_old(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra, opt_dec, opt_ra_err, opt_dec_err, 
              additonal_error=0.6):
    """
    Get the combined error and the axes components between an elongated 
    radio source and an optical source. Decompositions of the errors from get_sigma function
    
    Input:
    * maj_error: error in the major axis of the radio Gaussian in arsecs
    * min_error: error in the minor axis of the radio Gaussian in arsecs
    * pos_angle: position angle of the radio Gaussian in degrees
    * radio_ra: Right ascension of the radio source in degrees
    * radio_dec: Declination of the radio source in degrees
    * opt_ra: Right ascension of the optical source in degrees
    * opt_dec: Declination of the optical source in degrees
    * opt_ra_err: Error in right ascension of the optical source in degrees
    * opt_dec_err: Error in declination of the optical source in degrees
    * additonal_error: Additonal term to add to the error. By default
        it adds an astrometic error of 0.6 arcsecs.
    
    Output:
    * sigma: Combined error
    * sigma_maj: Error in the major axis direction
    * sigma_min: Error in the minor axis direction
    """
    factor = 0.60056120439322491 # sqrt(2.0) / sqrt(8.0 * log(2)); see Condon(1997) for derivation of adjustment factor
    majerr = factor * maj_error * 3600.
    minerr = factor * min_error * 3600.
    cosadj = np.cos(np.deg2rad(0.5*(radio_dec + opt_dec)))
    phi = np.arctan2((opt_dec - radio_dec), ((opt_ra - radio_ra)*cosadj))
    # angle from direction of major axis to vector joining LOFAR source and optical source
    sigma = np.pi/2.0 - phi - np.deg2rad(pos_angle) 
    
    # Convert the additional error froma arcsec to degrees
    # additonal_error = additonal_error / 3600.
    
    maj_squared = ((majerr * np.cos(sigma))**2 + 
                   (opt_ra_err * np.cos(phi))**2 +
                   additonal_error**2/2.
                   )
    min_squared = ((minerr * np.sin(sigma))**2 + 
                   (opt_dec_err * np.sin(phi))**2 +
                   additonal_error**2/2.
                   )
    return np.sqrt(maj_squared + min_squared), np.sqrt(maj_squared), np.sqrt(min_squared)


# Function to compute f(r)

##########################################################
#     Functions to comput n(m) and q(m) - generally      #
##########################################################

# Get a list of bin edges using a given bin width

def get_bin_list(magnitude, bin_width):
    """
    Get a list og bin edges based on specified bin with
    for a magnitude distribution
    
    Parameters:
    -----------
    
    magnitude : Magnitude distribution
    bin_width : Bin width
    
    Returns:
    --------
    
    bin_edges : List of bin edges of equal width
    """
    
    # Get number of bins
    nbins = int(np.ceil((np.max(magnitude) - np.min(magnitude)) / bin_width))
    
    # Get bin edges
    
    return np.linspace(np.min(magnitude), np.max(magnitude)+bin_width, nbins+1)


# Cumulative number density

def get_n_m(magnitude, bin_list, area):
    """Compute n(m)
    Density of sources per unit of area (cumulative form)
    
    Parameters:
    -----------
    
    magnitude : Magnitude distribution
    bin_list : List of bin edges to compute n(m)
    area : Survey area of the optical/NIR survey
    
    Returns:
    --------
    n(m) : Cumulatie number distribution, normalised to area
    """
    n_hist, _ = np.histogram(magnitude, bin_list)
    return np.cumsum(n_hist)/area


# Get q(m) also in a cumulative fashion

def get_q_m(lofar_coords, opt_coords, rmax, magnitude, bin_list, area, n_m):
    """
    Get q(m)/Q_0 - computes the real(m)/np.cumsum(real(m))
    
    Parameters:
    -----------
    
    lofar_coords : LOFAR coordinates (SkyCoord object)
    opt_coords : Optical/NIR coordinates (SkyCoord object)
    rmax : rmax within which to search for optical sources of a LOFAR source
    magnitude : magnitude distribution
    bin_list : List of bin edges - THE SAME BINS USED TO COMPUTE n(m)
    area : Survey area of the optical/NIR band
    n_m : output of get_n_m --- n(m) distribution
    
    Returns:
    --------
    q(m): Strictly, it returns q(m)/Q_0
    """
    
    # Search for all opt/NIR sources within rmax of LOFAR sources
    ind_l, ind_o, s2d, d3d = search_around_sky(lofar_coords, opt_coords, rmax*u.arcsec)
    
    # Get the unique indices into opt_coords that match
    ind_o_un = np.unique(ind_o)
    ind_l_un = np.unique(ind_l)
    print("Unique opt_ind: ", len(ind_o_un))
    print("Unique LOFAR ind: ", len(ind_l_un))
    
    
    # Get the magnitude of the optical/NIR sources that have a match - defined as total(m)
    total_m, _ = np.histogram(magnitude[ind_o_un], bin_list)

    # Compute real(m)
    real_m = np.cumsum(total_m) - len(lofar_coords)* n_m * np.pi * rmax**2
    
    # Remove any possible small negative values?
    real_m[real_m <= 0.] = 0.
    
    # Cumulative sum of the real_m
    # real_m_cumsum = np.cumsum(real_m)
    
    # Return the q(m)/Q_0 value - where Q_0 has yet to be determined (using an iterative approach)
    return real_m/real_m[-1], np.cumsum(total_m), real_m
    
# Function to generate random positions

def generate_rand_pos_bootes(ra_up,dec_up,ra_down,dec_down,n_random):
    """
    Generate n_random number of uniformly distributed random sources 
    within the sky-projected area specified by the ra dec positions
    
    Parameters:
    -----------
    ra_up : Upper corner RA
    dec_up : Upper corner Dec
    ra_down : Lower corner RA
    dec_down : Lower corner Dec
    n_random : Number of random sources to generate
    
    Returns:
    --------
    rand_coords : Random catalogue coordinates (SkyCoord object)
    """
    
    rand_ra = np.random.uniform(ra_down, ra_up, 2*n_random)
    rand_dec = np.random.uniform(dec_down, dec_up, 2*n_random)
    
    # Only select the objects within overlapping area
    random_ra = []
    random_dec = []
    
    # Counter for number of objects within the overlapping area
    count = 0
    i = -1  # Initial index
    
    # To take into account the non-uniform survey area shape
    while count < n_random:
        i = i + 1
        if ((rand_ra[i] > ra_down) & (rand_ra[i] < ra_up) &
            (rand_dec[i] > dec_down) & (rand_dec[i] < ra_up)):   # accept it if it is inside the overlapping area
            random_ra.append(rand_ra[i])
            random_dec.append(rand_dec[i])
            # Iterate the loop counter
            count = count + 1
            
    # End of while loop
    
    # Convert to SkyCoord object
    return SkyCoord(random_ra, random_dec, unit=(u.deg,u.deg), frame='icrs')


def generate_rand_pos_servs(ra_up,dec_up,ra_down,dec_down,n_random):
    """
    generate_rand_pos version for only SERVS area
    Generate n_random number of uniformly distributed random sources 
    within the sky-projected area specified by the ra dec positions
    
    Parameters:
    -----------
    ra_up : Upper corner RA
    dec_up : Upper corner Dec
    ra_down : Lower corner RA
    dec_down : Lower corner Dec
    n_random : Number of random sources to generate
    
    Returns:
    --------
    rand_coords : Random catalogue coordinates (SkyCoord object)
    """
    
    rand_ra = np.random.uniform(239.5, 246.0, 8*n_random)
    rand_dec = np.random.uniform(53.3, 56.7, 8*n_random)
    
    # Only select the objects within overlapping area
    random_ra = []
    random_dec = []
    
    # Counter for number of objects within the overlapping area
    count = 0
    i = -1  # Initial index
    
    # To take into account the non-uniform survey area shape
    while count < n_random:
        i = i + 1
        if isinSERVS(rand_ra[i], rand_dec[i]):   # accept it if it is inside the overlapping area
            random_ra.append(rand_ra[i])
            random_dec.append(rand_dec[i])
            # Iterate the loop counter
            count = count + 1
            
    # End of while loop
    
    # Convert to SkyCoord object
    return SkyCoord(random_ra, random_dec, unit=(u.deg,u.deg), frame='icrs')


# Function to compute Q_0 value for a given search radius

def get_Q0(lofar_coords, opt_coords, random_coords, radius):
    """
    Compute the Q_0 parameter for a given radius
    
    Parameters:
    -----------
    
    lofar_coords : LOFAR coordinates
    opt_coords : Optical/NIR coordinates
    n_random : Number of random sources to generate
    radius : Radius to search for blanks in the random or random blank catalogue
    
    Returns:
    --------
    Q_0 value : Strictly returns Q_0/F(r) from the above equation
    """
    
    # Match the LOFAR catalogue to the optical/NIR catalogue
    ind_1l, ind_2o, d2d, d3d = search_around_sky(lofar_coords, opt_coords, radius*u.arcsec)
    
    # Number of LOFAR sources without a optical/NIR counterpart within radius
    nl_no_match = len(lofar_coords) - len(np.unique(ind_1l))
    
    # Match this random catalogue to theoptical/NIR catalogue
    ind_r, ind_2o, d2d, d3d = search_around_sky(random_coords, opt_coords, radius*u.arcsec)
    
    # Number of random sources without an optical/NIR counterpart within radius
    nr_no_match = len(random_coords) - len(np.unique(ind_r))
    
    q0_value = 1 - (float(nl_no_match)/nr_no_match * (float(len(random_coords))/len(lofar_coords)))
    
    return q0_value, nl_no_match, nr_no_match


def get_fr_old(r, sigma, sigma_maj, sigma_min):
    """Get the probability related to the spatial distribution.
    
    Parameters:
    -----------
    
    r : Offset r between the LOFAR and potential optical counterpart
    sigma : Combined positional error between LOFAR source and the 
            potential optical counterpart (get_sigma_all[0])
    sigma_mag : Same positional error but along the major axis (get_sigma_all[1])
    sigma_min : Same positional error but along the minor axis (get_sigma_all[2])
    
    Returns:
    --------
    
    f(r) : The probability distribution function (pdf) between of the offset
            between LOFAR and optical counterpart
    """
    #print('radius is: {}'.format(r))
    #print('sigma is: {}'.format(sigma))
    #print('sigma_maj is: {}'.format(sigma_maj))
    #print('sigma_min is: {}'.format(sigma_min))
    return (0.5*np.exp(-0.5*r**2/(sigma**2)))/(np.pi*sigma_maj*sigma_min)


# Compute F(r) for Q_0 calculation

def compute_Fr(radius, average_sigma):
    """
    Compute F(r) as defined above
    
    Parameters:
    -----------
    
    radius : Radius to search for blanks in the random or random blank catalogue
    average_sigma : Avergae (combined) sigma value from get_sigma function
    """
    #print(radius)
    #print(average_sigma)
    return 1 - np.exp(- (radius**2)/(2 * average_sigma**2))


# Function to interpolate n(m) to actually get n(<c_mag)

def get_nm_interp(magnitude, centers, n_m_dist):
    """
    Interpolate the full n_m distribution to find n(mag)
    
    Parameters:
    -----------
    
    magnitude : Magnitude of the counterpart
    centers : Bin centres used to compute n_m distribution
    n_m_dist : Output of get_n_m
    
    Returns:
    --------
    n(mag) : Value of n_m at mag
    """
    
    return np.interp(magnitude, centers, n_m_dist)

# Function to interpolate q(m) to actually get q(<c_mag) while folding in the Q0 and the F(R) distribution

def get_qm_interp(magnitude, centers, q_m_dist, q0_value, Fr_dist):
    """
    Interpolate the full q_m distribution to find q(mag)
    AND fold in the Q_0 and F(r) parameters
    
    Parameters:
    -----------
    
    magnitude : Magnitude of the counterpart
    centers : Bin centres used to compute n_m distribution
    q_m_value : Output of get_q_m
    q0_value : Output of get_Q0 function
    Fr_dist : Output of the compute_Fr function
    
    Returns:
    --------
    q(mag) : Value of q_m at mag
    """
    # Fold in the Q_0 and the F(r) values
    return np.interp(magnitude, centers, q_m_dist*q0_value)/Fr_dist


# Define a function to compute the LR value

def get_lr(mag, q0, n_m, q_m, sigma, sigma_maj, sigma_min, offset, center, opt_search_rad, F_of_r):
    """
    Compute LR based on already computed values: Q0, n_m, q_m
    And computes f(r) and F(r) before computing LR
    
    Parameters:
    -----------
    
    mag : Magnitude of the couterpart
    q0 : Output of get_Q0 function
    n_m : Output of get_n_m function
    q_m: Output of get_q_m function
    sigma : Combined error
    sigma_maj: Error in the major axis direction
    sigma_min: Error in the minor axis direction
    offset : Separation between LOFAR source and possible counterpart
    center : Bin centres used to compute the n_m distribution
    opt_search_rad : Search radius used for the match
    F_of_r : bool, compute F(r) or not? True if computing Q0 from using blanks, otherwise, False
    
    Returns:
    --------
    lr_list : array of LR values (ONLY the max of this value should be taken)
    """
    
    # Compute f(r) - probability distribution of offset (from output of matching)
    # fr = get_fr(offset, sigma, sigma_maj, sigma_min)    # Old Guassian code without correct angles
    
    # fr = fr_u(offset, sigma_0_0, det_sigma)
    
    fr = get_fr_old(offset, sigma, sigma_maj, sigma_min)
    
    # Compute F(r) using search radius from matching and the combined sigma
    if F_of_r == True:
        # When running this for magnitude only run
        Fr = compute_Fr(opt_search_rad, sigma)
    elif F_of_r == False:
        # When running this for colour calibration, set F_of_r to False
        Fr = 1
    else:
        raise Exception("Computation of F(r) (for q(m)/q(m,c) not defined)")
    
    # Get the interpolated n(m)
    nm_interp = get_nm_interp(mag, center, n_m)
    
    # Get the interpolated q(m) distribution while folding in the 
    # Q_0 value and F(r) distribution
    qm_interp = get_qm_interp(mag, center, q_m, q0, Fr)
    
    #print('f(r) is: {}'.format(fr))
    #print('n(m) is: {}'.format(nm_interp))
    #print('q(m) is: {}'.format(qm_interp))
    
    return qm_interp * fr / nm_interp

####################################################
#                    RUN 2                         #
####################################################

def get_giK_bin_indices_old(iK_bin_list):
    """
    Get the indices of objects in each giK colour bins 
    10 i-K colour bins --> further divided into 2 equal g-i colour halves
    
    Parameters:
    -----------
    
    iK_bin_list : Bin list of i-K colours
    
    Returns:
    --------
    [iK_ind, : Indices in each i-K bin
    gi_upper, : Indices in upper g-i bin for each jth i-K bin
    gi_lower] : Indices in lower g-i bin for each jth i-K bin
    """
    
    iK_digitize = np.digitize(master["iK_col"], bins=iK_bin_list)

    # Prints the number of objects within the entire colour range
    #print(len(master[(iK_col_bins[0] <= master["iK_col"]) & (master["iK_col"] <= iK_col_bins[-1])]))

    # All the colours but as separate bins
    iK_all_ind = []

    # Gives indices of objects in each bin
    for k in range(1,len(np.unique(iK_digitize))-1):
        
        iK_all_ind.append(iK_digitize == k)
    
    # Now split each iK bin into equal halves in gi colour
    gi_lower = []
    gi_upper = []

    for aa in range(len(iK_all_ind)):
    
        # Find the median g-i colour
        gi_median = np.nanmedian(master["gi_col"][iK_all_ind[aa]])
    
        # Split into higher or lower than the median - this should not select any nan values (the less/greater than)
        gi_lower.append(master["gi_col"][iK_all_ind[aa]] <= gi_median)
        gi_upper.append(master["gi_col"][iK_all_ind[aa]] > gi_median)
        
    return [iK_all_ind, gi_upper, gi_lower]

# Get the giK bool array for 20 categories

def get_giK_bin_indices(iK_bin_list, iK_colours, gi_colours, all_master_indices):
    """
    Get the indices of objects in each giK colour bins 
    10 i-K colour bins --> further divided into 2 equal g-i colour halves
    
    Parameters:
    -----------
    
    iK_bin_list : Bin list of i-K colours
    iK_colours : i-K colour of the sources to be binned
    gi_colours : g-i colour of the sources to be binned
    all_master_indices: Has length = len(iK_colours)
    
    Returns:
    --------
    [iK_ind, : Indices in each i-K bin
    gi_upper, : Indices in upper g-i bin for each jth i-K bin
    gi_lower] : Indices in lower g-i bin for each jth i-K bin
    """
    
    # List of boolean arrays (each of length=len(iK_digitize))
    iK_all_ind = []
    # List of boolean arrays  - BUT each array has length = len(master) - so it an be used for indexing
    iK_full_ind = []

    # Loop counter
    count = 0
    
    # Gives indices of objects in each bin
    for k in range(len(iK_bin_list)-1):
        
        # Bool array of colours within the jth and jth+1 bin
        iK_all_ind.append((iK_colours >= iK_bin_list[k]) & (iK_colours < iK_bin_list[k+1]))
        
        subset_indices = all_master_indices[iK_all_ind[count]]
        
        # Store the full boolean array of length = len(master)
        iK_full_ind.append(np.isin(all_master_indices, subset_indices, assume_unique=True))
        count = count + 1
    
    # Now split each iK bin into equal halves in gi colour
    gi_lower = []
    gi_upper = []

    for aa in range(len(iK_all_ind)):
        
        # Subset of master indices within the jth i-K bin
        master_ind_subset_bin = all_master_indices[iK_all_ind[aa]]
    
        # Find the median g-i colour
        gi_median = np.nanmedian(gi_colours[iK_all_ind[aa]])
        
        # Get indices of objects within the upper and lower half of the g-i bin
        # that index into the FULL master catalogue
        gi_low_master_ind = master_ind_subset_bin[gi_colours[iK_all_ind[aa]] <= gi_median]
        gi_upp_master_ind = master_ind_subset_bin[gi_colours[iK_all_ind[aa]] > gi_median]
        
        # Split into higher or lower than the median - this should not select any nan values (the less/greater than)
        gi_lower.append(np.isin(all_master_indices, gi_low_master_ind, assume_unique=True))
        gi_upper.append(np.isin(all_master_indices, gi_upp_master_ind, assume_unique=True))
        
    return [iK_full_ind, gi_upper, gi_lower] # Don't actually need to return iK_all_ind

# Function to compute bin the K and Ks only sources in iK bins

def get_iK_bin_indices(iK_bin_list, iK_colours, all_master_indices, full_master_indices):
    """
    Bin the K-only and Ks-only sources into the iK bins
    
    Parameters:
    -----------
    
    iK_bin_list : Bin list of i-K colours
    iK_colours : i-K colour of the sources to be binned
    all_master_indices: Has length = len(iK_colours)
    full_master_indices : Has length = len(master)
    
    Returns:
    --------
    iK_ind : Indices in each i-K bin
    """

    # List of boolean arrays (each of length=len(iK_digitize))
    iK_all_ind = []
    # List of boolean arrays  - BUT each array has length = len(master) - so it an be used for indexing
    iK_full_ind = []

    # Loop counter
    count = 0
    
    # Gives indices of objects in each bin
    for k in range(len(iK_bin_list)-1):
        
        # Bool array of colours within the jth and jth+1 bin
        iK_all_ind.append((iK_colours >= iK_bin_list[k]) & (iK_colours < iK_bin_list[k+1]))

        # Corresponding indices of these objects
        subset_indices = all_master_indices[iK_all_ind[count]]
        
        # Store the full boolean array of length = len(master)
        iK_full_ind.append(np.isin(full_master_indices, subset_indices, assume_unique=True))
        count = count + 1
        
    return iK_full_ind

def get_qm_c(magnitude, bin_list):
    """
    Function to compute q(m) for a given category using same bin_list for all categories
    
    Parameters:
    -----------
    magnitude: Magnitude distribution
    bin_list: bin list of magnitude distribution
    
    Returns:
    --------
    qm_c
    """
    
    n_bins, _ = np.histogram(magnitude, bins=bin_list, range=(5.,38.))
    cumul_n_bins = np.cumsum(n_bins)
    return cumul_n_bins / float(cumul_n_bins[-1])

def gen_binc_binl(min_value, max_value, bin_width):
    """
    Function to generate bin centres and bin list between two magnitude limits with bin_width
    
    Parameters:
    -----------
    min_value: Minimum value of the bin_list
    max_value: Maximum value of the bin_list
    bin_width: bin_width
    
    Returns:
    --------
    bin_list: List of bin edges
    bin_centres: List of bin centres
    """
    
    bin_list = np.arange(min_value, max_value, bin_width)
    
    bin_centres = bin_list[:-1] + bin_width
    
    return bin_list, bin_centres

def get_nm_c(magnitude, bin_list, area):
    """Compute n(m)
    Density of sources per unit of area (cumulative form)
    
    Parameters:
    -----------
    
    magnitude : Magnitude distribution
    bin_list : List of bin edges to compute n(m)
    area : Survey area of the optical/NIR survey
    
    Returns:
    --------
    n(m) : Cumulatie number distribution, normalised to area
    """
    n_hist, _ = np.histogram(magnitude, bin_list, range=(5., 38.))
    return np.cumsum(n_hist)/area

#########################################################
# Functions for ML full run


def get_lr_full(mag, n_m, q_m, sigma, sigma_maj, sigma_min, offset, center, opt_search_rad):
    """
    Compute LR based on already computed values: Q0, n_m, q_m
    And computes f(r) and F(r) before computing LR
    
    Parameters:
    -----------
    
    mag : Magnitude of the couterpart
    n_m : Output of get_n_m function
    q_m: Output of get_q_m function
    sigma : Combined error
    sigma_maj: Error in the major axis direction
    sigma_min: Error in the minor axis direction
    offset : Separation between LOFAR source and possible counterpart
    center : Bin centres used to compute the n_m distribution
    opt_search_rad : Search radius used for the match
    
    Returns:
    --------
    lr_list : array of LR values (ONLY the max of this value should be taken)
    """
    
    # Compute f(r) - probability distribution of offset (from output of matching)
    fr = get_fr(offset, sigma, sigma_maj, sigma_min)
    print('f(r) is: {}'.format(fr))
    
    # Get the interpolated n(m)
    nm_interp = get_nm_interp(mag, center, n_m)
    print('n(m) is: {}'.format(nm_interp))
        
    # Get the interpolated q(m) distribution while folding in the 
    # Q_0 value and F(r) distribution
    qm_interp = get_qm_interp_full(mag, center, q_m)
    print('q(m) is: {}'.format(qm_interp))
        
    return qm_interp * fr / nm_interp


def get_qm_interp_full(magnitude, centers, q_m_dist):
    """
    Interpolate the full q_m distribution to find q(mag)
    AND fold in the Q_0 and F(r) parameters
    
    Parameters:
    -----------
    
    magnitude : Magnitude of the counterpart
    centers : Bin centres used to compute n_m distribution
    q_m_value : Output of get_q_m
    Fr_dist : Output of the compute_Fr function
    
    Returns:
    --------
    q(mag) : Value of q_m at mag
    """
    # Fold in the Q_0 and the F(r) values
    return np.interp(magnitude, centers, q_m_dist)


def get_q_m_kde(magnitude, bin_centre, radius=5, bandwidth=0.2):
    """Compute q(m)
    Normalized probability of a real match in a 
    non-cumulative fashion using a KDE.
    For this function we need the centre of the bins instead 
    of the edges.
    **Note that the output is non-cumulative**
    """
    # Get real(m)
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(magnitude[:, np.newaxis])
    pdf_q_m = np.exp(kde_skl.score_samples(bin_centre[:, np.newaxis]))
    real_m = pdf_q_m*len(magnitude)/np.sum(pdf_q_m)
    # Correct probability if there are no sources
    if len(magnitude) == 0:
        real_m = np.ones_like(n_hist_total)*0.5
    # Remove small negative numbers
    real_m[real_m <= 0.] = 0.
    return real_m/np.sum(real_m)

def get_n_m_kde(magnitude, bin_centre, area, bandwidth=0.2):
    """Compute n(m)
    Density of sources per unit of area in a non-cumulative
    fashion using a KDE.
    For this function we need the centre of the bins instead 
    of the edges.
    **Note that the output is non-cumulative**
    """
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(magnitude[:, np.newaxis])
    pdf = np.exp(kde_skl.score_samples(bin_centre[:, np.newaxis]))
    return pdf/area*len(magnitude)/np.sum(pdf)


def estimate_q_m_kde(magnitude, bin_centre, n_m, coords_small, coords_big, radius=5, bandwidth=0.2):
    """Compute q(m)
    Estimation of the distribution of real matched sources with respect 
    to a magnitude (normalized to 1). As explained in Fleuren et al. in a 
    non-cumulative fashion using a KDE.
    For this function we need the centre of the bins instead 
    of the edges.
    """
    assert len(magnitude) == len(coords_big)
    # Cross match
    idx_small, idx_big, d2d, d3d = search_around_sky(
        coords_small, coords_big, radius*u.arcsec)
    n_small = len(coords_small)
    idx = np.unique(idx_big)
    # Get the distribution of matched sources
    kde_skl_q_m = KernelDensity(bandwidth=bandwidth)
    kde_skl_q_m.fit(magnitude[idx][:, np.newaxis])
    pdf_q_m = np.exp(kde_skl_q_m.score_samples(bin_centre[:, np.newaxis]))
    n_hist_total = pdf_q_m*len(magnitude[idx])/np.sum(pdf_q_m)
    # Correct probability if there are no sources ## CHECK
    if len(magnitude[idx]) == 0:
        n_hist_total = np.ones_like(n_hist_total)*0.5
    # Estimate real(m)
    real_m = n_hist_total - n_small*n_m*np.pi*radius**2
    # Remove small negative numbers ## CHECK
    real_m[real_m <= 0.] = 0.
    return real_m/np.sum(real_m)


def gen_rand_cat_inMOC(n, ra_up, ra_down, dec_up, dec_down, catmoc):

    ''' 
    generate a random catalogue of positions within the given MOC
    the positions are generated within the RA and DEC defined and
    then removed if not within the given MOC. This is repeated 
    until n points lie within the MOC
    
    
    '''
    
    moc_area = catmoc.area_sq_deg
    area = (abs(ra_up-ra_down)) * (abs(dec_up-dec_down))
    ratio = area/moc_area
    if ratio<1:
        print('area ratio is less than 1 so something is wrong')
    print('ratio of the random generation area compared to the moc area is: {}'.format(area/moc_area))
    ra = []
    dec = []
    #randcoords = generate_random_catalogue(int(n*5*ratio),ra_down,ra_max,dec_down,dec_up)
    randra = np.random.uniform(ra_down, ra_up, int(2*n*ratio))
    randdec = np.random.uniform(dec_down, dec_up, int(2*n*ratio))
    mask = inMoc(randra,randdec,catmoc)
    if np.sum(mask)>=n:
        coords = SkyCoord(randra[mask][:n],randdec[mask][:n],unit='deg')
        assert len(coords)==n,print('the number of coordinates generated is not equal to the number of radio sources in the moc. Something is wrong')
            
        return(coords)
    if np.sum(mask)<n:
        while len(ra)<n:
            ra = np.append(ra,randra[mask])
            dec = np.append(dec,randdec[mask])
            randcoords = generate_random_catalogue(n*5*ratio,RAmin,RAmax,DECmin,DECmax)
            randra  = randcoords.ra.value
            randdec = randcoords.dec.value
            mask = inMoc(randra,randdec,catmoc)
            ra = np.append(ra,randra[mask])
            dec = np.append(dec,randdec[mask])

        coords = SkyCoord(ra[:n],dec[:n],unit='deg')
        return(coords)


# New eroor functions to calculate f(r)

def get_sigma_all(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra, opt_dec, opt_ra_err, opt_dec_err, 
              additional_error=0.6):
    """Apply the get_sigma function in parallel and return the determinant of 
    the covariance matrix and its [1,1] term (or [0,0] in Python)
    """
    n = len(opt_ra)
    det_sigma = np.empty(n)
    sigma_0_0 = np.empty(n)
    for i in range(n):
        sigma = get_sigma(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra[i], opt_dec[i], opt_ra_err[i], opt_dec_err[i], 
              additional_error=additional_error)
        det_sigma[i] = det(sigma)
        sigma_0_0[i] = sigma[0,0]
    return sigma_0_0, det_sigma


def get_sigma(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra, opt_dec, opt_ra_err, opt_dec_err, 
              additional_error=0.6):
    """
    Get the covariance matrix between an elongated 
    radio source and an optical source.
    
    Input:
    * maj_error: error in the major axis of the radio Gaussian in arsecs
    * min_error: error in the minor axis of the radio Gaussian in arsecs
    * pos_angle: position angle of the radio Gaussian in degrees
    * radio_ra: Right ascension of the radio source in degrees
    * radio_dec: Declination of the radio source in degrees
    * opt_ra: Right ascension of the optical source in degrees
    * opt_dec: Declination of the optical source in degrees
    * opt_ra_err: Error in right ascension of the optical source in degrees
    * opt_dec_err: Error in declination of the optical source in degrees
    * additonal_error: Additonal term to add to the error. By default
        it adds an astrometic error of 0.6 arcsecs.
    
    Output:
    * sigma: Combined covariance matrix
    """
    factor = 0.60056120439322491 # sqrt(2.0) / sqrt(8.0 * log(2)); see Condon(1997) for derivation of adjustment factor
    majerr = factor * maj_error * 3600
    minerr = factor * min_error * 3600
    
    # angle between the radio and the optical sources
    cosadj = np.cos(np.deg2rad(0.5*(radio_dec + opt_dec)))
    phi = np.arctan2(((opt_ra - radio_ra)*cosadj), (opt_dec - radio_dec))
    
    # angle from direction of major axis to vector joining LOFAR source and optical source
    alpha = phi - np.deg2rad(pos_angle)
    
    # Covariance matrices
    sigma_radio_nr = np.array([[majerr**2, 0], [0, minerr**2]])
    sigma_optical_nr = np.array([[opt_dec_err**2, 0], [0, opt_ra_err**2]])
    
    # Rotate the covariance matrices
    R_radio = R(alpha)
    sigma_radio = R_radio @ sigma_radio_nr @ R_radio.T
    R_optical = R(phi)
    sigma_optical = R_optical @ sigma_optical_nr @ R_optical.T
    
    # Additional error
    sigma_additonal_error = np.array([[additional_error**2, 0], [0, additional_error**2]])
    sigma = sigma_radio + sigma_optical + sigma_additonal_error
    
    return sigma


def R(theta):
    """Rotation matrix.
    Input:
      - theta: angle in degrees
    """
    theta_rad = np.deg2rad(theta)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])



def fr_u(r, sigma_0_0, det_sigma):
    """Get the probability related to the spatial distribution"""
    return 0.5/np.pi/det_sigma*np.exp(-0.5*r**2/sigma_0_0)