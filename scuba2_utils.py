from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from astropy.nddata import Cutout2D
from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord as SC
from astropy.io import fits
from astropy import table
from astropy.table import Table, QTable

from astropy.wcs import wcs
from astropy.nddata import Cutout2D

def diff_Gauss(x, c1, sigma1, sigma2):
    """define 1D difference of Gaussian function with scalable parameters"""
    c2 = c1 - 1.0
    return c1 * np.exp(-0.5*x**2/sigma1**2) - c2 * np.exp(-0.5*x**2/sigma2**2)


def plot_map(data, wcs_copy, s_list=None, r_min=3.75):
    """plot image with wcs projected axes"""
    
    fig = plt.figure(figsize=(5,5),dpi=180)
    ax1 = fig.add_subplot(projection=wcs_copy)
    im = ax1.imshow(data)
    plt.colorbar(im)
    ax1.set_xlabel('RA')
    ax1.set_ylabel('Dec')
    
    if s_list is not None:
        ra = s_list['ra']
        dec = s_list['dec']
        x,y = wcs_copy.wcs_world2pix(ra,dec,0)
        for i in range(len(x)):
            circle = plt.Circle((x[i],y[i]),r_min, ec='lime', fill=False)
            fig.gca().add_patch(circle)
    return fig, ax1



def load_scuba2_image_data(image_filename, noise_filename, proj_name):
    """open image and noise files and return squeezed data"""
    print("Opening files...")
    im_hdul = fits.open(image_filename)
    ns_hdul = fits.open(noise_filename)
    print("done.")
    
    im = np.squeeze(im_hdul[0].data) # squeeze since scuba2 fits files are 3D with one axis just 1 entry long
    ns = np.squeeze(ns_hdul[0].data)
    
    hdr = im_hdul[0].header
    wcs_copy = wcs.WCS(hdr,naxis=2)
    
    print("Closing files.")
    im_hdul.close()
    ns_hdul.close()
  
    ns[np.where(ns==0.)] = 100000 #850 noise map has zeros outside --> Nans in S/N map
    
    # create S/N map
    print("Generating signal-to-noise map.\n")
    snr = im/ns

    # fill in NaNs
    print("Removing bad data...")
    NANs = np.isnan(im)
    
    im[NANs] = 0
    ns[NANs] = 100000
    snr[NANs] = 0
    #snr[np.isinf(snr)] = 0
    print("done.\n")
    
    return im, ns, snr, hdr, wcs_copy   #, px_size


def create_psf(name, im, ns, snr, thresh, ns_factor, sz_psf): 
    """ define a psf based on the stacked sources in the image (not s/n map)
            thresh = threshold for detection in S/n map
            ns_factor = multiple of minimum noise within which we can detect sources
            
            psf is a 2*sz_psf+1, 2*sz_psf+1 array
            
            Adapted from scuba2.mkpsf by Li-Yen Hsu"""
    sz = 2*sz_psf+1 #size of psf, for readability
    
    snr[ns > ns_factor*ns.min()] = 0 # remove snr data outside source detection area
    print("Minimum rms noise:",ns.min())
    
    print("Creating PSF:")
    print('Stacking sources...')
    stack = np.zeros((sz, sz), float)
    peak = peak_local_max(snr, threshold_abs=thresh)
    # stack these detected sources
    for i in range(len(peak)):
        stack += im[peak[i,0]-sz_psf:peak[i,0]+sz_psf+1,peak[i,1]-sz_psf:peak[i,1]+sz_psf+1]
    
    print('done.')
   
    stack = stack/stack.max() # normalize the stacked image
    print('# of sources in psf fit:',len(peak))
    
    # compute the radial profile of the stacked image
    profile=np.zeros(sz_psf+1,float)
    radial=np.zeros(sz_psf+1,float)
    dist=np.zeros((sz,sz),float)
    
    print("\nComputing radial profile...")
    for i in range(sz):
        for j in range(sz):
            dist[i,j] = np.sqrt((i-sz_psf)**2.0 + (j-sz_psf)**2.0) #distance from center to each point
    
    for i in range(sz_psf+1):
        filter = (dist >= -0.5+i) & (dist < 0.5+i)
        profile[i]=np.mean(stack[filter]) # compute average profile, binned into increments of [i-0.5, i+0.5]
        radial[i]=np.mean(dist[filter])   # average radius of each bin
    print("done.\n")    

    print("Fitting PSF to stacked sources...")
    # fit a diff-of-Gaussion model to the radial profile
    p_best, covar = curve_fit(diff_Gauss, radial, profile, bounds=(0, [10., 15., 15.]))
        #note: bounds take lower, upper bounds on fittable paramters (c1, sigma1, sigma2)

    # make a plot
    fig = plt.figure()
    plt.scatter(radial,profile)
    plt.plot(radial, diff_Gauss(radial, *p_best), 'r-')
    plt.xlabel('arcsec')
    plt.ylabel('normalized flux')
    plt.grid(True)
    plt.show(block=False)
    
    print('Best-fit parameters:')
    print('c1, sigma1, sigma2 =', p_best)
    
    psf = np.zeros((sz,sz),float)
    
    for i in range(sz):
        for j in range(sz):
            psf[i,j] = diff_Gauss(dist[i,j], *p_best)
    
    return psf
    

def extract_sources(proj_name, hdr, im_arg, ns_arg, snr_arg, psf, wcs_copy, thresh, ns_factor, r_min, r_psf):
    """ extract sources from image map by finding the peak s/n pixel, subtracting a scaled PSF centered on that
    pixel, and saving the flux and location of that source.
    
    thresh = minimum s/n ratio for a detection
    ns_factor = multiple of min noise that determines boundary of our source detection area
    r_min = min allowed distance to a source we have already found
    r_psf = radius of psf
            
            Adapted from scuba2.extract by Li-Yen Hsu"""
    
    print("Source extraction:")
    #so we don't change the original arrays
    im = im_arg.copy()
    snr = snr_arg.copy()
    ns = ns_arg.copy() 
    
    print("Trimming psf...")
    if psf.shape[0] < r_psf:
        print("ERROR: r_psf too large. Results will be affected.")
    
    # remove the outer radii of the PSF (defined by r_psf) (SJM: from Hsu code)
    ncol = psf.shape[1]
    center = (ncol-1)/2
    psf=psf[int(center-r_psf) : int(center+r_psf+1), int(center-r_psf) : int(center+r_psf+1)]
    psfdist=np.zeros((r_psf*2+1,r_psf*2+1),float)

    for i in range(r_psf*2+1):
        for j in range(r_psf*2+1):
            psfdist[i,j]=np.sqrt( (i-r_psf)**2.0 + (j-r_psf)**2.0 )    

    psf[psfdist > r_psf]=0
    print("done.\n")

    
    # set up arrays to fill with source information
    ra = np.array([])
    dec = np.array([])
    x = np.array([])
    y = np.array([])
    noiselevel = np.array([])
    flux_out = np.array([])
    snr_out = np.array([])

    i=0
    
    print("finding sources...")
    while snr.max() >= thresh and i < 400:
        
        peak = peak_local_max(snr, num_peaks=1)
        xi = peak[0,1] #indices of source in image pixels (note y is row and x is column)
        yi = peak[0,0]

        # compute the minimum among the distances between the current detection and previous detections
        if i > 0:
            mindist = np.sqrt( (xi-x)**2.0 + (yi-y)**2.0 ).min()
        else: mindist = 1000

        # ignore a detection if it's within "r_min" from a previous detection
        if mindist <= r_min:  
            snr[yi,xi]=0
            print('mindist error:',mindist,r_min)
            
        else:
            #append source info to final arrays
            flux_detected = im[yi,xi]
            flux_out = np.append(flux_out, round(flux_detected,6) )
            snr_out = np.append(snr_out, round(snr[yi,xi],6) )
            noiselevel = np.append(noiselevel, ns[yi,xi])

            ra_i,dec_i = wcs_copy.wcs_pix2world(xi,yi,0) #note reversal of x and y
            ra  = np.append(ra , round(float(ra_i),6) )
            dec  = np.append(dec , round(float(dec_i),6) )
            x = np.append(x, xi)
            y = np.append(y, yi)

            #remove source from image and snr
            im[yi-r_psf :yi+r_psf+1, xi-r_psf :xi+r_psf+1] -= psf*flux_detected
            snr[yi-r_psf :yi+r_psf+1, xi-r_psf :xi+r_psf+1] -= psf*flux_detected/ns[yi-r_psf :yi+r_psf+1, xi-r_psf :xi+r_psf+1]
            print(i, snr_out[i], np.round(flux_detected,3), np.round(ns[yi,xi],3))
            i=i+1
    
    print("done.\n")
    
    #remove sources outside source detection area
    filter = noiselevel <= ns_factor*ns.min()
    
    ra = ra[filter]
    dec=dec[filter]
    flux_out=flux_out[filter]
    snr_out=snr_out[filter]
    
    #sort sources by flux
    order = np.argsort(flux_out) #order of indices from smallest flux to largest
    ra = np.flipud(ra[order])
    dec = np.flipud(dec[order])
    snr_out = np.round(np.flipud(snr_out[order]),3)
    flux_out = np.round(np.flipud(flux_out[order]),3)
    err_out = np.round(flux_out/snr_out,3)

    source_list = {'ID':range(1,len(ra)+1), 'ra':ra, 'dec':dec, 'flux(mJy/beam)':flux_out, \
                   'error(mJy/beam)':err_out, 'S/N':snr_out}
    source_list = Table(source_list)
   
    return source_list
    
    