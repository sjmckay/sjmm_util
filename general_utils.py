import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord as SC
from astropy.io import fits
from astropy import table
from astropy.table import Table, QTable

from astropy.wcs import wcs
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from reproject import reproject_interp,reproject_exact

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 



def make_regions(coords, name = None, tags=None, r=2, color='green',width=1):
    """generate a region file"""
    
    ra = coords.ra.deg
    dec = coords.dec.deg
    
    ra[np.isnan(ra)] = 0
    dec[np.isnan(dec)]=0
    
    width = str(width)
    print('Generating region file...')    
    n = len(ra)
    text = np.empty([2+n*2], dtype=object)

    text[0] = ('global color='+color+' dashlist=8 3 width='+width+' font="helvetica 10 normal roman"'+
             ' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
    text[1] = 'icrs\n'
    
    if tags is not None:
        if isinstance(tags,str) and tags == 'none':
            tags = []
            for i in range(len(coords)):
                tags.append('')
        else:    
            assert len(tags) == n
    else:
        tags = [str(i) for i in range(n)]
    
    for i in range(n):
        text[i*2+2] = 'circle('+str(ra[i])+','+str(dec[i])+','+str(r)+'")\n'
        text[1+i*2+2] = '# text('+str(ra[i])+','+str(dec[i]+(r*2.)/3600.)+')'+' text={'+tags[i]+'}\n'
    try:
        target = open(name, 'w')
        target.writelines(text)
        target.close()
        print('Done.\n')
    except:
        raise IOError(f'Could not write to file {name}')



def compute_radial_profile(im):
    sz = im.shape[0]
    r = int(sz/2)
    improfile=np.zeros(r+1,float)  
    radial=np.zeros(r+1,float)
    dist=np.zeros(im.shape,float)

    #compute distance array
    for i in range(sz):
        for j in range(sz):
            dist[i,j] = np.sqrt((i-r)**2.0 + (j-r)**2.0) #distance from center to each point

    #average flux in each radius bin (image, model, residual)
    for i in range(r+1):
        filter = (dist >= -0.5+i) & (dist < 0.5+i)
        improfile[i]=np.mean(im[filter]) # compute average profile, binned into increments of [i-0.5, i+0.5]
        radial[i]=np.mean(dist[filter])   # average radius of each bin

    return radial, improfile


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def calc_medians(x, y, nbins=30, bins = None):
    """
    Divide the x axis into sections and return groups of y based on its x value
    """
    if bins is None:
        bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins)

    bin_space = bins[1:]-bins[0:-1]
    mbins = bins[0:-1] + bin_space/2.

    indices = np.digitize(x, bins)
    medians = []
    lb = []
    ub = []
    bserr= []
    for i in range(1,len(bins)):
        if np.sum(indices==i) < 2:
            medians.append(np.nan)
            lb.append(np.nan)
            ub.append(np.nan)
            bserr.append(np.nan)
        else:
            medians.append(np.nanmedian(y[indices==i]))
            lb.append(np.nanpercentile(y[indices==i],16))
            ub.append(np.nanpercentile(y[indices==i],84))
            err = bootstrap((y[indices==i],),statistic=np.nanmedian, vectorized=True, n_resamples=1000).standard_error
            if err is None: bserr.append(np.nan)
            else: bserr.append(err)
    return bins, mbins, medians, lb, ub, bserr