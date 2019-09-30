#!/usr/bin/env python
# coding: utf-8

# In[1]:


import astropy
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import os


# In[2]:


def pop_list(ls,indices):
    for n,ind in enumerate(indices):
        ls.pop(ind)
    return(ls)


# In[6]:


# In[3]:


photz = Table.read('data/photz/LH_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits')
#photz =Table.read('/beegfs/lofar/deepfields/science_ready_catalogs/LH_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits')

# In[4]:


# In[5]:


cigale_input = photz.copy()


# In[7]:


filters_file = open('data/photz/filters/Lockman-SWIRE_filters.res.info','r')
#filters_file = open('/beegfs/lofar/deepfields/science_ready_catalogs/filter_information/Lockman-SWIRE_filters.res.info','r')
filters = []
for line in filters_file:
    temp = line.split(' ')
    filters.append(temp)
filters_file.close()

for n,line in enumerate(filters):
    if n!=32:
        filters[n] = line[0:-1]
    else:
        filters[n][-1] = filters[n][-1].replace('\n','')
    filters[n] = pop_list(filters[n],[1 for m in range(len(filters[n])-2)])
    filters[n][-1] = filters[n][-1].replace('.filter','')
filters = filters[2:]

# In[8]:


#translate_file = open('data/photz/filters/LH.filter.translate','r')
translate_file = open('/beegfs/lofar/deepfields/science_ready_catalogs/filter_information/LH.filter.translate','r')
translate = []
for line in translate_file:
    if '#' in line:
        continue
    temp = line.split(' ')
    translate.append(temp)
translate_file.close()

translate = translate[:len(translate)-1]

for n,line in enumerate(translate):
    translate[n][1] = line[1][1:-1]


# In[9]:


for n,line in enumerate(translate):
    colname = line[0]
    filt_num = int(line[1])
    filt_name = ''
    for m,filt in enumerate(filters):
        if int(filt[0])==filt_num:
            filt_name = filt[1]
    if filt_name=='':
        continue
    if 'err' in colname:
        filt_name = filt_name+'_err'

    cigale_input.rename_column(colname,filt_name)


# In[10]:


cigale_input.rename_column('z1_median','redshift')


# In[12]:


Table.write(cigale_input,'cigale_input_Lockman.fits',format='fits',overwrite=True)


# In[ ]:





# In[ ]:




