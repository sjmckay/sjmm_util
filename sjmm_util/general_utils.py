import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord as SC
from astropy.table import Table

from astropy.wcs import wcs
from astropy.nddata import Cutout2D

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

def coords_from_3col(tab):
    """get skycoords from table RAh, RAm, RAs etc columns"""
    coords = []
    if 'rah' in tab.colnames:
        for i in range(len(tab['rah'])):
            coords.append(SC(str(tab['rah'][i])+'h'+str(tab['ram'][i])+'m'+str(tab['ras'][i])+'s' \
                           +' '+str(tab['decd'][i])+'d'+str(tab['decm'][i])+'m'+str(tab['decs'][i])+'s',
                           unit=['hourangle','deg']))
    else:
        if 'DE-' in tab.colnames:
                sgn = np.array(tab['DE-'],dtype=object)
        else:
            sgn = []
            for i in range(len(tab['DEm'])):
                sgn.append('')
        for i in range(len(tab['RAh'])):
            coords.append(SC(str(tab['RAh'][i])+'h'+str(tab['RAm'][i])+'m'+str(tab['RAs'][i])+'s' \
                       +' '+sgn[i]+str(tab['DEd'][i])+'d'+str(tab['DEm'][i])+'m'+str(tab['DEs'][i])+'s',
                       unit=['hourangle','deg']))
    return SC(coords)

def add_ra_dec_cols(tab, coords, index):
    '''add ra and dec in degrees to table from coords'''
    ra = [float(c.to_string(precision=6).split(' ')[0]) for c in coords]
    dec = [float(c.to_string(precision=6).split(' ')[1]) for c in coords]
    tab.add_column(ra, index = index, name = 'ra')
    tab.add_column(dec,index = index + 1, name = 'dec')
    return tab


def compute_radial_profile(im):
    """azimuthal averaging to produce a radial profile from 2D data."""
    sz = im.shape[0]
    r = int(sz/2)
    improfile=np.zeros(r+1,float)  
    radial=np.zeros(r+1,float)
    dist=np.zeros(im.shape,float)

    #compute distance array
    for i in range(sz):
        for j in range(sz):
            dist[i,j] = np.sqrt((i-r)**2.0 + (j-r)**2.0) #distance from center to each point

    #average value in each radius bin
    for i in range(r+1):
        filter = (dist >= -0.5+i) & (dist < 0.5+i)
        improfile[i]=np.mean(im[filter]) # compute average profile, binned into increments of [i-0.5, i+0.5]
        radial[i]=np.mean(dist[filter])   # average radius of each bin

    return radial, improfile


def find_map_peak(map, mask=None):
    '''find peak pixel in a 2D cutout. Returns pixel coords in row, col (y, x) format (i.e., numpy format)'''
    nmap = map.copy()
    if mask is not None:
        nmap[mask] = np.nan
        if np.all(np.isnan(nmap)):
            warnings.warn("All nan map for peak finding. Returning center...")
            return nmap.shape[0]//2, nmap.shape[1]//2
    return np.unravel_index(np.nanargmax(nmap,),nmap.shape)

def rebin(arr, factor=None,new_len=None,function='mean'):
    """Rebin 2D array arr to shape new_shape by averaging/summing."""
    
    if function =='sum': func = np.sum
    elif function == 'quad': 
        func = lambda x, axis: np.sqrt(np.sum(x**2,axis=axis))/x.shape[axis]
    else: func = np.mean
    
    if factor is not None:
        shape = (len(arr)//factor,factor)   
        max_index = len(arr)-len(arr)%factor
        newarr = arr[:max_index].copy()
        return func(newarr.reshape(shape),axis=-1)
    if new_len is not None:
        shape = (new_len,len(arr)//new_len)   
        max_index = (len(arr)//new_len)*new_len
        newarr = arr[:max_index].copy()
        return func(newarr.reshape(shape), axis=-1)

def calc_medians(x, y, nbins=30, bins = None):
    """
    Divide the x axis into sections and return groups of y based on its x value.
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
    """produce a Gaussian kernel density estimation of a given dataset's density distribution."""
    xmin,xmax = xlim
    ymin,ymax = ylim
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    func = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, func
