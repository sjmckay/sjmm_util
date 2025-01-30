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

from scipy.stats import bootstrap
import scipy.stats as st

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
    else: bins = np.asarray(bins)
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


def kde2d(x,y,xlim,ylim):
    xmin,xmax = xlim
    ymin,ymax = ylim
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    func = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, func

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y