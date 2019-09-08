#
from astropy.coordinates import search_around_sky
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky

import numpy as np
from operator import itemgetter
import glob

from astropy.utils.console import human_file_size

DEG_TO_ARCSEC = 3600.

def get_center(bins):
    """
    Get the central positions for an array defining bins
    """
    return (bins[:-1] + bins[1:]) / 2

def return_hist_par(bin_width,distribution):
    """
    Function which given a bin_width and the distribution will compute the histogram of distribution of the variable
    and return the binned distribution and the bin edges computed and the bin centres

    Input:
            bin_width
            distribution

    Output: 
            binned_distribution
            bin_edges
            bin_centres
    """

    # Number of bins based on bin_width
    nbins = (np.max(distribution) - np.min(distribution)) / bin_width

    # Bin edges
    bine = np.linspace(np.nanmin(distribution),np.nanmax(distribution)+bin_width,nbins+1)
    #print(bine,len(bine))

    # Returns the number of objects within each (i-K) bin
    binned_distribution, bin_edges_np = np.histogram(distribution,bins=bine,normed=False)
    #print(bin_edges_np,len(bin_edges_np))

    # Compute the bin centres of the (i-K) colour
    bin_centres = bin_edges_np[:-1] + bin_width

    return binned_distribution, bin_edges_np, bin_centres


def roundup_hundred(number):
    """
    Round up a number to the next hundred
    """

    return np.ceil(number/100.) * 100


def roundup_thousand(number):
    """
    Round up a number to the next hundred
    """

    return np.ceil(number/1000.) * 1000 


def logspace_bins(start, stop, log_bin_width):
    """
    Return an array that is spaced equally in log10

    Inputs:
        start:          Minimum of the distribution
        stop:           Maximum of the distribution
        log_bin_width:  Bin spacing in log units (i.e. bin spacing after taking log)

    Output:
        log_bin_edges:  Return an array of bin edges
    """
    # Number of bins
    bin_number = (np.log10(stop) - np.log10(start)) / log_bin_width
    # Equally space the values after converting to log 
    log_bin_edges = np.linspace(np.log10(start), np.log10(stop), int(np.round(bin_number)))

    return log_bin_edges


def logspace_bins_n(start, stop, bin_number):
    """
    Return an array that is spaced equally in log10

    Inputs:
        start:          Minimum of the distribution
        stop:           Maximum of the distribution
        nbins:  Number of bins

    Output:
        log_bin_edges:  Return an array of bin edges
    """
    # Number of bins
    # bin_number = (np.log10(stop) - np.log10(start)) / log_bin_width
    # Equally space the values after converting to log 
    log_bin_edges = np.linspace(np.log10(start), np.log10(stop), int(np.round(bin_number)))

    return log_bin_edges

#################################################################


def coord_matching(first_coords, second_coords, match_sep):
    """
    Performs a coordinates cross-match of the positions between two SkyCoord objects
    coord_matching(tomatch_coords, catalog_coords, match_sep)

    Input:
            first_coords:  First set of coordinates - a SkyCoord object
            second_coords: Second set of coordinates - a SkyCoord object
            match_sep:     Separation to search within - in arcseconds

    Output:
            indx1:         Indices into first_coords that have matched with coords2 - Array
            indx2:         Indices into second_coords that have matched with coords1 - Array
            sep2d:         On-sky separation between the coordinates - (Angle object)
    """
    # Perform the catalog match
    indx1, indx2, sep2d, dist3d = search_around_sky(first_coords, second_coords, seplimit=match_sep*u.arcsec)

    return indx1, indx2, sep2d


# Find the nth nearest neighbours

def nearest_neigh_match(match_coord, catalog_coord, nth_neighbour):
    """
    Finds the nth nearest neighbours in catalog_coord, that match in the match_coord catalogue

    The 2D separation is in the units of the input catalogues? u.deg
    """

    idx, sep2d, dist3d = match_coordinates_sky(match_coord, catalog_coord, nthneighbor=nth_neighbour)

    return idx, sep2d


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


def jytoabmag(flux_in_Jy):
    """
    Convert flux in Jy to AB magnitudes
    """

    return 2.5 * (23 - np.log10(flux_in_Jy)) - 48.6


def latest_dir(dir_pattern):
    """
    Function to take dir_pattern and sort and return the latest directory
    """

    # Get all the directories
    all_dirs = glob.glob(dir_pattern)
    tup_dirs = [tuple(xx.split('_')) for xx in all_dirs]
    
    # Directories first sorted by month, then date, then the number of directory on nth date
    tup_dirs_s = sorted(tup_dirs, key=itemgetter(-3,-4,-1))
    
    # List of directories sorted
    dirs_out = ['_'.join(yy) for yy in tup_dirs_s]

    return dirs_out[-1]


def field_filter(ra_up, dec_up, ra_down, dec_down, ra_coord, dec_coord):
    """
    Function to select sources within the specified rectangular area.
    Returns a bool array with the same length as the ra_coord/dec_coord

    Parameters:
    -----------
    ra_up : The higher RA coordinate
    dec_up : The higher Dec coordinate
    ra_down : The lower RA coordinate
    dec_down : The lower Dec coordinate
    ra_coord : The RA coordinates of the survey
    dec_coord : The Dec coordinates of the survey

    Returns:
    --------
    bool_within : Boolean array of indices into ra/dec_coord
    """

    bool_within = ((ra_coord >= ra_down) & (ra_coord < ra_up) &
                   (dec_coord >= dec_down) & (dec_coord < dec_up))

    return bool_within


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


def hm_file_size(array_shape):
    """
    Take the array shape and compute the size of the array
    """

    size_bytes = (np.product(array_shape, dtype=np.int64) * np.dtype(complex).itemsize)*u.byte

    return human_file_size(size_bytes)


def emag_to_snr(magnitude_error):
    """
    Convert an error in magnitude to a SNR
    """

    return 1 / (10**(magnitude_error / 2.5) - 1)


def get_overlap_sources(source_ra, source_dec):
    """
    Return indices of sources within overlapping area of multiwavelength coverage
    """
    overlapping_bool = []
    for i in range(len(source_ra)):
        if ((isinpan(source_ra[i],source_dec[i])) and
            (isinukidss(source_ra[i],source_dec[i])) and
            (isinSWIRE(source_ra[i], source_dec[i]))):
            # overlapping_sources.append(i)
            overlapping_bool.append(True)
        else:
            overlapping_bool.append(False)

    return overlapping_bool
