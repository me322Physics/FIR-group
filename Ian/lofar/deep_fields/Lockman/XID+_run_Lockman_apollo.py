print('started running python script')
import astropy
from astropy.io import ascii, fits
import pylab as plt
from astropy import wcs
from astropy.table import Table,Column,join,hstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import pymoc
import glob
from time import sleep
import os
import sys

print('importing modules.....half way')

import numpy as np
import xidplus
from xidplus import moc_routines
import pickle
import xidplus.catalogue as cat

import sys
from herschelhelp_internal.utils import inMoc,flux_to_mag
from xidplus.stan_fit import SPIRE

import aplpy
import seaborn as sns
#sns.set(color_codes=True)
import pandas as pd
#sns.set_style("white")
import xidplus.posterior_maps as postmaps
from herschelhelp_internal.masterlist import merge_catalogues, nb_merge_dist_plot, specz_merge
import pyvo as vo
from herschelhelp_internal.utils import inMoc

print('finished importing modules')


taskid = np.int(os.environ['SGE_TASK_ID'])-1
#taskid=3
print('taskid is: {}'.format(taskid))


#Read in the prepared catalogue
lofar = Table.read('data/xid+_prepared_cat.fits')
lofar = lofar[lofar['xid+_rerun']]

if taskid*10>len(lofar):
    print('Task id is too high. Trying to run code on more sources than exist')
    sys.exit()
ind_low = taskid*10
if taskid*10+9>len(lofar):
    ind_up = len(lofar)-1
else:
    ind_up = taskid*10+9
ra = lofar['ra'][[ind_low:ind_up]]
dec = lofar['dec'][[ind_low:ind_up]]

columns = 'ra','dec','help_id','f_spire_250','ferr_spire_250','flag_spire_250','f_spire_350','ferr_spire_350','flag_spire_350','f_spire_500','ferr_spire_500','flag_spire_500','flag_optnir_det','f_mips_24'
masterlist = Table.read('../../../../../HELP/dmu_products/dmu32/dmu32_Lockman-SWIRE/data/Lockman-SWIRE_20180219.fits')
help_masterlist = masterlist[columns]
help_masterlist['help_id'] = help_masterlist['help_id'].astype(str)
masterlist = 0

#Read in the Ldust predictions needed to increase the coerage of the prior list
ldust = Table.read('../../../../../HELP/dmu_products/dmu28/dmu28_Lockman-SWIRE/data/Lockman-SWIRE_Ldust_prediction_results.fits') 
ldust.rename_column('id','help_id')
ldust['help_id'] = ldust['help_id'].astype(str)
#join Ldust predictions with the help table
ldust_id = [name[:-6] for name in list(ldust['help_id'])]
ldust['help_id'] = np.array(ldust_id)
ldust_cols = ['help_id','bayes.dust.luminosity','bayes.dust.luminosity_err','best.universe.redshift']
ldust = ldust[ldust_cols]
help_masterlist_ldust = join(help_masterlist,ldust,keys='help_id',join_type='outer')

#create a moc so that only radio sources in the prior area are selected
c = SkyCoord(ra=help_masterlist_ldust['ra'], dec=help_masterlist_ldust['dec'])
prior_moc=pymoc.util.catalog.catalog_to_moc(c,60,15)

radio_in_moc = inMoc(ra,dec,prior_moc)
#if radio_in_moc==False:
#    sys.exit()

lofar_coords = SkyCoord(ra,dec,unit='deg')
help_coords = SkyCoord(help_masterlist['ra'],help_masterlist['dec'],unit='deg')
radius = 2
idx_help, idx_lofar, d2d, d3d = lofar_coords.search_around_sky(
    help_coords, radius*u.arcsec)

#compute the predicted flux from the dust and usethis to construct the prior
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
f_pred=help_masterlist_ldust['bayes.dust.luminosity']/(4*np.pi*cosmo.luminosity_distance(help_masterlist_ldust['best.universe.redshift']).to(u.cm))
mask = np.isfinite(f_pred)
ldust_mask = (np.log10(f_pred)>8.5) & (np.isfinite(f_pred))
mips_mask = (help_masterlist['flag_optnir_det']>=5) & (help_masterlist['f_mips_24']>20)

prior_cat = help_masterlist_ldust[ldust_mask | mips_mask]

#if there is only one crossmatch within the search radius then match them if the source is in the prior list
#otherwise add the ra and dec of the lofar optical counterpart to the prior list
XID_rerun = []
source_type = []
ldust_mask = (np.log10(f_pred)>8.5) & (np.isfinite(f_pred))
mips_mask = (help_masterlist['flag_optnir_det']>=5) & (help_masterlist['f_mips_24']>20)
mask = [ldust_mask[idx_help] | mips_mask[idx_help]]
idx_true = idx_help[mask]





#Read in the herschel images
imfolder='../../../../../HELP/dmu_products/dmu19/dmu19_HELP-SPIRE-maps/data/'

pswfits=imfolder+'ELAIS-N1_SPIRE250_v1.0.fits'#SPIRE 250 map
pmwfits=imfolder+'ELAIS-N1_SPIRE350_v1.0.fits'#SPIRE 350 map
plwfits=imfolder+'ELAIS-N1_SPIRE500_v1.0.fits'#SPIRE 500 map

#-----250-------------
hdulist = fits.open(pswfits)
im250phdu=hdulist[0].header
im250hdu=hdulist[1].header

im250=hdulist[1].data*1.0E3 #convert to mJy
nim250=hdulist[2].data*1.0E3 #convert to mJy
w_250 = wcs.WCS(hdulist[1].header)
pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----350-------------
hdulist = fits.open(pmwfits)
im350phdu=hdulist[0].header
im350hdu=hdulist[1].header

im350=hdulist[1].data*1.0E3 #convert to mJy
nim350=hdulist[2].data*1.0E3 #convert to mJy
w_350 = wcs.WCS(hdulist[1].header)
pixsize350=3600.0*w_350.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----500-------------
hdulist = fits.open(plwfits)
im500phdu=hdulist[0].header
im500hdu=hdulist[1].header 
im500=hdulist[1].data*1.0E3 #convert to mJy
nim500=hdulist[2].data*1.0E3 #convert to mJy
w_500 = wcs.WCS(hdulist[1].header)
pixsize500=3600.0*w_500.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()

from astropy.coordinates import SkyCoord
from astropy import units as u
c = SkyCoord(ra=ra, dec=dec)  
import pymoc
moc=pymoc.util.catalog.catalog_to_moc(c,60,15)

#---prior250--------
prior250=xidplus.prior(im250,nim250,im250phdu,im250hdu, moc=moc)#Initialise with map, uncertianty map, wcs info and primary header
prior250.prior_cat(prior_cat['ra'],prior_cat['dec'],'prior_cat',ID=prior_cat['help_id'])#Set input catalogue
prior250.prior_bkg(-5.0,5)#Set prior on background (assumes Gaussian pdf with mu and sigma)
#---prior350--------
prior350=xidplus.prior(im350,nim350,im350phdu,im350hdu, moc=moc)
prior350.prior_cat(prior_cat['ra'],prior_cat['dec'],'prior_cat',ID=prior_cat['help_id'])
prior350.prior_bkg(-5.0,5)

#---prior500--------
prior500=xidplus.prior(im500,nim500,im500phdu,im500hdu, moc=moc)
prior500.prior_cat(prior_cat['ra'],prior_cat['dec'],'prior_cat',ID=prior_cat['help_id'])
prior500.prior_bkg(-5.0,5)

#pixsize array (size of pixels in arcseconds)
pixsize=np.array([pixsize250,pixsize350,pixsize500])
#point response function for the three bands
prfsize=np.array([18.15,25.15,36.3])
#use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
from astropy.convolution import Gaussian2DKernel

##---------fit using Gaussian beam-----------------------
prf250=Gaussian2DKernel(prfsize[0]/2.355,x_size=101,y_size=101)
prf250.normalize(mode='peak')
prf350=Gaussian2DKernel(prfsize[1]/2.355,x_size=101,y_size=101)
prf350.normalize(mode='peak')
prf500=Gaussian2DKernel(prfsize[2]/2.355,x_size=101,y_size=101)
prf500.normalize(mode='peak')

pind250=np.arange(0,101,1)*1.0/pixsize[0] #get 250 scale in terms of pixel scale of map
pind350=np.arange(0,101,1)*1.0/pixsize[1] #get 350 scale in terms of pixel scale of map
pind500=np.arange(0,101,1)*1.0/pixsize[2] #get 500 scale in terms of pixel scale of map

prior250.set_prf(prf250.array,pind250,pind250)#requires PRF as 2d grid, and x and y bins for grid (in pixel scale)
prior350.set_prf(prf350.array,pind350,pind350)
prior500.set_prf(prf500.array,pind500,pind500)

prior250.get_pointing_matrix()
prior350.get_pointing_matrix()
prior500.get_pointing_matrix()

prior250.upper_lim_map()
prior350.upper_lim_map()
prior500.upper_lim_map()

from xidplus.stan_fit import SPIRE
fit=SPIRE.all_bands(prior250,prior350,prior500,iter=1000)

posterior=xidplus.posterior_stan(fit,[prior250,prior350,prior500])
priors = [prior250,prior350,prior500]
import xidplus.catalogue as cat
SPIRE_cat=cat.create_SPIRE_cat(posterior,priors[0],priors[1],priors[2])
SPIRE_cat = Table.read(SPIRE_cat)

mask = SPIRE_cat['HELP_ID']=='lofar'
mask_lofar = lofar['ra']==ra
mask_pcat = prior_cat['help_id']=='lofar'
SPIRE_cat = SPIRE_cat[mask]
SPIRE_cat.add_columns([prior_cat['f_mips_24'][mask_pcat],prior_cat['flag_optnir_det'][mask_pcat]])
lofar_fir = hstack([lofar[mask_lofar],SPIRE_cat])
XID_rerun_col = Column(data=XID_rerun,name='XID_rerun',dtype=bool)
source_type_col = Column(data=source_type,name='source_type',dtype=str)
lofar_fir.add_columns([XID_rerun_col,source_type_col])
   
if os.path.exists('data/fir/xidplus_run_{}'.format(taskid))==True:()
else:
    os.mkdir('data/fir/xidplus_run_{}'.format(taskid))
Table.write(lofar_fir,'data/fir/xidplus_run_{}/lofar_xidplus_fir_{}.fits'.format(taskid,taskid),overwrite=True)


