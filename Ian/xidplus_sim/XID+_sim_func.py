import numpy as np
import matplotlib.pyplot as plt
import astropy
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.io import ascii, fits
from astropy.table import Table
from astropy import units as u
from scipy.signal import convolve2d
from random import *

#x = separation
#y = flux ratio
#z = noise ratio

def gauss_1(x,s):
    return np.exp(-x**2/(2*s**2))

def data_map(seperation, flux_ratio, snr, wavelength):
    
    dat_map = np.zeros((288,288))
    
    t = np.arange(-36, 36, 1)
    gaus = gauss_1(t,7.64)
    g = gaus[:, np.newaxis] * gaus[np.newaxis, :]
    
    y1 = 10
    y2 = y1*flux_ratio 
    
    dat_map[int(144-seperation/2),144]=y1
    dat_map[int(144+seperation/2),144]=y2
    
    dat_conv = convolve2d(dat_map, g, mode='same')
    
    if wavelength==250: #pixsize=6 arcsec
        sim_length = 126 #box length of the simulated data in arcseconds
        pixsize = 6
        dat_map = np.zeros((sim_length,sim_length))
    
        t = np.arange(-36, 36, 1)
        gaus = gauss_1(t,18.15/2.355) #create gaussian beam for SPIRE at 250um
        g = gaus[:, np.newaxis] * gaus[np.newaxis, :]
    
        y1 = 10
        y2 = y1*flux_ratio 
    
        dat_map[int(sim_length/2-seperation/2),sim_length/2]=y1
        dat_map[int(sim_length/2+seperation/2),sim_length/2]=y2
    
        dat_conv = convolve2d(dat_map, g, mode='same')
        
        final_map = np.zeros((sim_length/pixsize,sim_length/pixsize))
        
        n = 0
        m = 0
        
        
        for n in range(len(final_map)):
            for m in range(len(final_map)):
                final_map[n,m] = np.mean(dat_conv[pixsize*n:pixsize*n+pixsize-1,pixsize*m:pixsize*m+pixsize-1])

        #seed(1)
        noise = abs(np.random.normal(0, y1/snr, pixsize**2).reshape(sim_length/pixsize,sim_length/pixsize))
        
        f_n_map = final_map+noise
        
        a = np.ones((sim_length/pixsize,sim_length/pixsize))
        hdu1 = fits.PrimaryHDU()
        hdu2 = fits.ImageHDU(f_n_map)
        hdu3 = fits.ImageHDU(noise)    
    
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        
#        plt.imshow(f_n_map)
#        plt.show()

        image_file = fits.open('../../../HELP/dmu_products/dmu19/dmu19_HELP-SPIRE-maps/data/COSMOS_SPIRE250_v1.0.fits')
        

    
        
    if wavelength==350: #pixsize=8 arcsec
        
        final_map = np.zeros((36,36))
        
        n = 0
        m = 0
        
        
        for n in range(36):
            for m in range(36):
                final_map[n,m] = np.mean(dat_conv[8*n:8*n+7,8*m:8*m+7])
                
        seed(1)
        noise = abs(np.random.normal(0, y1/snr, 1296).reshape(36,36))
        
        f_n_map = final_map+noise
        
        a = np.ones((36,36))
        
#        plt.imshow(f_n_map)
#        plt.show()

        image_file = fits.open('../../../HELP/dmu_products/dmu19/dmu19_HELP-SPIRE-maps/data/COSMOS_SPIRE350_v1.0.fits')
    
    if wavelength==500: #pixsize=12 arcsec
        
        final_map = np.zeros((24,24))
        
        n = 0
        m = 0
        
        
        for n in range(24):
            for m in range(24):
                final_map[n,m] = np.mean(dat_conv[12*n:12*n+11,12*m:12*m+11])
                
        seed(1)
        noise = abs(np.random.normal(0, y1/snr, 576).reshape(24,24))
        
        f_n_map = final_map+noise
        
        a = np.ones((24,24))
        
#        plt.imshow(f_n_map)
#        plt.show()

        
        image_file = fits.open('../../../HELP/dmu_products/dmu19/dmu19_HELP-SPIRE-maps/data/COSMOS_SPIRE500_v1.0.fits')
    
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(f_n_map)
    hdu3 = fits.ImageHDU(a)    
    
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    
#    plt.title('Map')
#    plt.imshow(final_map+noise)    
#    plt.colorbar()
#    plt.show()

    #Open FITS file
    
    im250phdu=image_file[0].header
    im250hdu=image_file[1].header
    
    #Load in the data and the error maps
    image = image_file[1].data
    image_error = image_file[3].data
    wcs_h =  (wcs.WCS(image_file[1].header)).celestial
    
    #Check positions of files
#    image_file.info()
    image_file.close()
    
    #Create random ra and dec values
    ra = 150.1
    dec = 2.2
    
    #Give length of box sides
    box_length = 288./3600.
    
    c = SkyCoord(ra*u.degree,dec*u.degree,unit='deg')
    
    imgcut = Cutout2D(image, c, size=[box_length*u.degree, box_length*u.degree], wcs=wcs_h)
    wcscut = imgcut.wcs
    imgcut = imgcut.data
    
    errcut = Cutout2D(image_error, c, size=[box_length*u.degree, box_length*u.degree], wcs=wcs_h)
#    wcs_err_cut = imgcut.wcs
#   img_err_cut = imgcut.data
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection=wcscut)
   
#    ax.imshow(imgcut, interpolation='nearest', origin='lower')
#    plt.show()
    
    #replace the image cutout with your data here
    
    imgcut = hdulist[1].data
    errcut = hdulist[2].data
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcscut)
    
    ax.imshow(imgcut, interpolation='nearest', origin='lower')
    plt.show()
    
    prior_cat = Table([[1,2],[ra, ra],[dec+seperation/7200., dec-seperation/7200.]], names=('id', 'ra', 'dec'), meta={'name': 'first table'})
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcscut)
    
    ax.imshow(imgcut, interpolation='nearest', origin='lower')
    
    x = prior_cat['ra']
    y = prior_cat['dec']
    ax.scatter(x,y,transform=ax.get_transform('world'),c='red')
    plt.show()
    
    return imgcut, prior_cat, wcs_h, errcut, c, im250phdu, im250hdu

              
def data_map_250_wcs(seperation, flux_ratio, snr,plot=False):
    
    sim_length = 120 #box length of the simulated data in arcseconds
    pixsize = 6
    dat_map = np.zeros((sim_length,sim_length))
    
    t = np.arange(-36, 37, 1)
    gaus = gauss_1(t,18.15/2.355) #create gaussian beam for SPIRE at 250um
    g = gaus[:, np.newaxis] * gaus[np.newaxis, :]
    
    y1 = 10
    y2 = y1*flux_ratio 
    
    dat_map[int(sim_length/2-seperation/2),int(sim_length/2)]=y1
    dat_map[int(sim_length/2+seperation/2),int(sim_length/2)]=y2
    x = np.array([int(sim_length/2),int(sim_length/2)])
    y = np.array([int(sim_length/2-seperation/2),int(sim_length/2+seperation/2)])

    
    dat_conv = convolve2d(dat_map, g, mode='same')
    
    #create a wcs for the high resolution map
    w_high_res = wcs.WCS(naxis=2)
    #assign coordinates to the bottom left hand corner of the data
    w_high_res.wcs.crval = [150.,2.]
    w_high_res.wcs.crpix = [-0.5,-0.5]
    w_high_res.wcs.cdelt = [1/3600,1/3600]
    #convert the pixel coordinates of the sources to ra and dec
    #the last parameter should be 1 because it is a fits file. 
    #i'm not so sure about the reason it is a one but that is what it should be
    source_ra,source_dec = w_high_res.wcs_pix2world(x,y,1)
    source_x_coord,source_y_coord = w_high_res.wcs_world2pix(source_ra,source_dec,1)
    
    if plot==True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=w_high_res)
        ax.imshow(dat_conv, interpolation='nearest', origin='lower')
        x_test,y_test = w_high_res.wcs_pix2world(-0.5,-0.5,0)
        ax.scatter(x_test,y_test,transform=ax.get_transform('world'),c='green',s=5)
        ax.scatter(150.,2.,transform=ax.get_transform('world'),c='green',s=5)
        ax.scatter(source_ra,source_dec,transform=ax.get_transform('world'),c='blue',s=1)
        plt.show()
        
    final_map = np.zeros((int(sim_length/pixsize),int(sim_length/pixsize)))

    #create a low resolution             
    for n in range(len(final_map)):
        for m in range(len(final_map)):
            #print(dat_conv[pixsize*n:pixsize*n+pixsize-1,pixsize*m:pixsize*m+pixsize-1])
            #print(np.mean(dat_conv[pixsize*n:pixsize*n+pixsize-1,pixsize*m:pixsize*m+pixsize-1]))
            final_map[n,m] = np.mean(dat_conv[pixsize*n:pixsize*n+pixsize-1,pixsize*m:pixsize*m+pixsize-1])
    
    noise = abs(np.random.normal(0, y1/snr, int(sim_length/pixsize)**2).reshape(int(sim_length/pixsize),int(sim_length/pixsize)))
        
    f_n_map = final_map+noise
        
    #create a fits file with the low resolution map and the noise map
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(f_n_map)
    hdu3 = fits.ImageHDU(noise)    
    
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    
    #create a wcs for the low resolution map
    w_low_res = wcs.WCS(naxis=2)
    w_low_res.wcs.ctype = ['RA---TAN','DEC--TAN']
    w_low_res.wcs.crval = [150.,2.]
    w_low_res.wcs.crpix = [-0.5,-0.5]
    w_low_res.wcs.cdelt = [pixsize/3600,pixsize/3600]
    w_low_res.pixel_shape = [len(f_n_map),len(f_n_map)]

    if plot==True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=w_low_res)
        ax.imshow(f_n_map, interpolation='nearest', origin='lower')
        ax.scatter(150.,2.,transform=ax.get_transform('world'),c='blue',s=1)
        ax.scatter(source_ra,source_dec,transform=ax.get_transform('world'),c='blue',s=1)
        plt.show()
    
    c = SkyCoord(np.mean(source_ra)*u.degree,np.mean(source_dec*u.degree))
    
    #create the header that xid+ will use to create it's own wcs
    hdu2.header['CTYPE1'] = 'RA---TAN'
    hdu2.header['CTYPE2'] = 'DEC--TAN'
    hdu2.header['CRVAL1'] = 150.
    hdu2.header['CRVAL2'] = 2.
    #These need to be 0.5 instead of -0.5 for some reason that i don't understand
    hdu2.header['CRPIX1'] = 0.5
    hdu2.header['CRPIX2'] = 0.5
    hdu2.header['CD1_1'] = pixsize/3600
    hdu2.header['CD1_2'] = 0.
    hdu2.header['CD2_1'] = 0.
    hdu2.header['CD2_2'] = pixsize/3600
    hdu2.header['NAXIS1'] = len(f_n_map)
    hdu2.header['NAXIS2'] = len(f_n_map)
    imhdu = ''
    
    return(f_n_map,[source_ra,source_dec],w_low_res,noise,c,hdu1,hdu2.header)
              
              
def data_map_350(seperation, flux_ratio, snr):
    
    sim_length = 120 #box length of the simulated data in arcseconds
    pixsize = 8
    dat_map = np.zeros((sim_length,sim_length))
    
    t = np.arange(-50, 50, 1)
    gaus = gauss_1(t,25.15*8/8.33/2.355) #create gaussian beam for SPIRE at 250um
    g = gaus[:, np.newaxis] * gaus[np.newaxis, :]
    
    y1 = 10
    y2 = y1*flux_ratio 
    
    dat_map[int(sim_length/2-seperation/2),int(sim_length/2)]=y1
    dat_map[int(sim_length/2+seperation/2),int(sim_length/2)]=y2
    
    dat_conv = convolve2d(dat_map, g, mode='same')
        
    final_map = np.zeros((int(sim_length/pixsize),int(sim_length/pixsize)))
        
    n = 0
    m = 0
        
        
    for n in range(len(final_map)):
        for m in range(len(final_map)):
            final_map[n,m] = np.mean(dat_conv[pixsize*n:pixsize*n+pixsize-1,pixsize*m:pixsize*m+pixsize-1])

    noise = abs(np.random.normal(0, y1/snr, int(sim_length/pixsize)**2).reshape(int(sim_length/pixsize),int(sim_length/pixsize)))
        
    f_n_map = final_map+noise
        
        #a = np.ones((sim_length/pixsize,sim_length/pixsize))
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(f_n_map)
    hdu3 = fits.ImageHDU(noise)    
    
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        
    image_file = fits.open('../../../HELP/dmu_products/dmu19/dmu19_HELP-SPIRE-maps/data/COSMOS_SPIRE350_v1.0.fits')
    imphdu=image_file[0].header
    imhdu=image_file[1].header
    image = image_file[1].data
    image_error = image_file[3].data
    wcs_h =  (wcs.WCS(image_file[1].header)).celestial
    image_file.close()
        
    ra = 150.
    dec = 2.
    pix_y,pix_x = wcs_h.wcs_world2pix(ra,dec,0)
    inra= [ra,ra]
    indec = [dec-seperation/3600/2,dec+seperation/3600/2]
    prior_cat = [inra,indec]
    c = SkyCoord(ra*u.degree,dec*u.degree)
        
    xmin = int(pix_x-(len(f_n_map))/2)+1
    xmax = int(pix_x+(len(f_n_map))/2)+1
    ymin = int(pix_y-(len(f_n_map))/2)+1
    ymax = int(pix_y+(len(f_n_map))/2)+1
    print(xmin,xmax,ymin,ymax)
    image[xmin:xmax,ymin:ymax] = f_n_map
    image_error[xmin:xmax,ymin:ymax] = noise
    return(image,prior_cat,wcs_h,image_error,c,imphdu,imhdu)

def data_map_500(seperation, flux_ratio, snr):
    
    sim_length = 180 #box length of the simulated data in arcseconds
    pixsize = 12 #pixel size of the herschel map at the wavelength
    dat_map = np.zeros((sim_length,sim_length))
    
    t = np.arange(-60, 60, 1)
    gaus = gauss_1(t,36.3/2.355) #create gaussian beam for SPIRE at 250um
    g = gaus[:, np.newaxis] * gaus[np.newaxis, :]
    
    y1 = 10 #flux of the main source
    y2 = y1*flux_ratio  #flux of the secondary source
    
    #put these two sources nto the map at the centre and offset by their seperation
    dat_map[int(sim_length/2-seperation/2),int(sim_length/2)]=y1
    dat_map[int(sim_length/2+seperation/2),int(sim_length/2)]=y2
    
    #convole the delta function map with the guassian beam to simulate the herschel map
    dat_conv = convolve2d(dat_map, g, mode='same')
    
    #create the empty hercshel map
    final_map = np.zeros((int(sim_length/pixsize),int(sim_length/pixsize)))
        
    n = 0
    m = 0
        
    #loop over the empty herschel map and average the smaller pixels inside each larger pixel    
    for n in range(len(final_map)):
        for m in range(len(final_map)):
            final_map[n,m] = np.mean(dat_conv[pixsize*n:pixsize*n+pixsize-1,pixsize*m:pixsize*m+pixsize-1])
    
    #create a noise map with mean decided by the SNR given
    noise = abs(np.random.normal(0, y1/snr, int(sim_length/pixsize)**2).reshape(int(sim_length/pixsize),int(sim_length/pixsize)))
    
    #add the noise and signal map together to get the final simulated herschel map
    f_n_map = final_map+noise
    
    #read in a real herschel map so that the simulated data can be inserted into it in while getting access to correct 
    #header information
    image_file = fits.open('../../../HELP/dmu_products/dmu19/dmu19_HELP-SPIRE-maps/data/COSMOS_SPIRE500_v1.0.fits')
    imphdu=image_file[0].header
    imhdu=image_file[1].header
    image = image_file[1].data
    image_error = image_file[3].data
    wcs_h =  (wcs.WCS(image_file[1].header)).celestial
    image_file.close()
    
    #the coordinates that the simulated data is centered on
    ra = 150.
    dec = 2.
    #find the pixel in the herschel map that this correspods to
    pix_y,pix_x = wcs_h.wcs_world2pix(ra,dec,0)
    #create the coordinates of the two sources in the simulated data
    inra= [ra,ra]
    indec = [dec-seperation/3600/2,dec+seperation/3600/2]
    prior_cat = [inra,indec]
    c = SkyCoord(ra*u.degree,dec*u.degree)
    
    #find the range of pixels that you need to look at 
    #NOTE THAT THE CODE IS CORRECT BUT THE LABELS OF X AND Y ARE SWITCHED AROUND
    xmin = int(pix_x-(len(f_n_map))/2)+1
    xmax = int(pix_x+(len(f_n_map))/2)+1
    ymin = int(pix_y-(len(f_n_map))/2)+1
    ymax = int(pix_y+(len(f_n_map))/2)+1
    print(xmin,xmax,ymin,ymax)
    image[xmin:xmax,ymin:ymax] = f_n_map
    image_error[xmin:xmax,ymin:ymax] = noise
    return(image,prior_cat,wcs_h,image_error,c,imphdu,imhdu)

