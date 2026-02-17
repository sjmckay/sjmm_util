import numpy as np

from skimage import measure
import warnings
from scipy import ndimage
from scipy.stats import bootstrap
import scipy.stats as st

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 



def get_image_footprint(data, wcs_input=None, fill_holes=True):
    """Get the footprint polygon(s) of valid data in a 2D image.

    Parameters
    ----------
    data : 2D numpy.ndarray
        Image array (NaNs or zeros treated as invalid).
    wcs_input : astropy.wcs.WCS, optional
        If provided, footprints are also returned in sky coordinates.
    fill_holes : bool, default=True
        If True, fill internal holes so only the outer boundary is kept.
        If False, returns the outer polygon along with any holes.

    Returns
    -------
    footprint_pix : list of (N, 2) numpy.ndarray
        List of polygon vertices in pixel coordinates (x, y).
        The first polygon is the outer boundary; subsequent ones (if any) are holes.
    footprint_world : list of (N, 2) numpy.ndarray or None
        Same polygons in world coordinates (RA, Dec),
        or None if no WCS is provided.
    """
    # Mask valid pixels
    mask = np.isfinite(data)&(data!=0)
    if fill_holes:
        # Fill holes in data
        mask = ndimage.binary_fill_holes(mask)
        contours = measure.find_contours(mask, 0.5)
        if not contours:
            return None, None
        # largest contour
        contour = max(contours, key=len)
        footprint_pix = [np.fliplr(contour)]  # ensure (x, y)
    else:
        # Keep holes in data
        contours = measure.find_contours(mask, 0.5)
        if not contours:
            return None, None
        # Sort contours by area (so we return the outer boundary first)
        areas = [np.abs(np.cross(c[:-1], c[1:]).sum()) for c in contours]
        idx_sorted = np.argsort(areas)[::-1]
        footprint_pix = [np.fliplr(contours[i]) for i in idx_sorted]

    #Also return world coordinates if WCS is provided
    if wcs_input is not None:
        footprint_world = [wcs_input.all_pix2world(poly, 0) for poly in footprint_pix]
    else:
        footprint_world = None

    return footprint_pix, footprint_world


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
    """Rebin 2D array arr to shape new_shape by averaging/summing. If the factor by which to rebin does not
    fit evenly into the old array, extra elements are discarded.
    
    Parameters
    ----------
    arr (array): array to rebin
    
    factor(int or float): optional factor (number of channels) by which to rebin
    
    new_len (int): optional new length of array (in case you don't want to input as a factor to downsize by)
    
    function (str): either 'mean', 'quad', or 'sum'. If mean or sum, applies those numpy functions. 
        If quad, sums in quadrature and divides by sqrt(factor) to get reduced error.
        
        
    Returns
    -------
    (array): rebinned array with function applied.
    """
    
    if function =='sum': func = np.nansum
    elif function == 'quad': 
        func = lambda x, axis: np.sqrt(np.nansum(x**2,axis=axis))/x.shape[axis]
    else: func = np.nanmean
    
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


def distance_array(map, center):
    '''create array of euclidean distance between array elements and "center" pixel (in numpy format, i.e., y, x)
    
    center=(px,py)
    map[py,px] = center_val
    '''
    x = np.arange(map.shape[0]) 
    y = np.arange(map.shape[1])
    Y,X = np.meshgrid(y,x)
    dist = np.sqrt((X-center[1])**2 + (Y-center[0])**2)
    return dist


def calc_medians(x, y, nbins=30, bins = None):
    """
    Compute the median of binned y values as a function of x, along with 16th and 84th percentiles and bootstrap error on the median.

    Parameters
    ----------
    x (array-like): Independent variable values.
    y (array-like): Dependent variable values.
    nbins (int, optional): Number of bins to use if bins are not provided.
    bins (array-like, optional): Bin edges. If None, bins are computed from the range of x. 

    Returns
    -------
    bins (numpy.ndarray): Bin edges used for the calculation.
    mbins (numpy.ndarray): Bin centers corresponding to the medians.
    medians (numpy.ndarray): Median of y values in each bin.
    lb (numpy.ndarray): 16th percentile of y values in each bin.
    ub (numpy.ndarray): 84th percentile of y values in each bin.
    bserr (numpy.ndarray): Bootstrap standard error of the median in each bin.
    """
    if bins is None:
        bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins)
    else: bins = np.asarray(bins)
    bin_space = bins[1:]-bins[0:-1]
    mbins = bins[0:-1] + bin_space/2.

    indices = np.digitize(x, bins)
    medians, lb, ub, bserr = [], [], [], []
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


def calc_bins(x, y, nbins=30, bins = None, func=np.nanmedian):
    """
    Compute a function of binned y values as a function of x as well as bootstrap error on the calculation.
    Generalization of calc_medians.

    Parameters
    ----------
    x (array-like): Independent variable values.
    y (array-like): Dependent variable values.
    nbins (int, optional): Number of bins to use if bins are not provided.
    bins (array-like, optional): Bin edges. If None, bins are computed from the range of x. 
    func (callable, optional): Function to apply to y values in each bin. Default is np.nanmedian.

    Returns
    -------
    bins (numpy.ndarray): Bin edges used for the calculation.
    mbins (numpy.ndarray): Bin centers corresponding to the medians.
    stat (numpy.ndarray): Function values in each bin.
    bserr (numpy.ndarray): Bootstrap standard error of the function in each bin.
    """
    if bins is None:
        bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins)
    else: bins = np.asarray(bins)
    bin_space = bins[1:]-bins[0:-1]
    mbins = bins[0:-1] + bin_space/2.

    indices = np.digitize(x, bins)
    stat, bserr = [], [], 
    for i in range(1,len(bins)):
        if np.sum(indices==i) < 2:
            stat.append(np.nan)
            bserr.append(np.nan)
        else:
            stat.append(func(y[indices==i]))
            err = bootstrap((y[indices==i],),statistic=func, vectorized=True, n_resamples=1000).standard_error
            if err is None: bserr.append(np.nan)
            else: bserr.append(err)
    return bins, np.array(mbins), np.array(stat), np.array(bserr)


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