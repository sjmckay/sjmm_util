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
    
def get_scuba2_prior(coords, s2_name, im, ns, snr, wcsi, band='850'):

    if band == '450':
        snr_thresh = 3.5
        stack_thresh = 4.0
        r_min = 3.75
        r_psf = 30 
        sz_psf = 80 
        ns_factor = 4.5 
    else:
        snr_thresh = 3.5
        stack_thresh = 4.0
        r_min = 7.25
        r_psf = 50 
        sz_psf = 70
        ns_factor = 4.5 


    psf = create_psf(s2_name, im, ns, snr, stack_thresh, ns_factor, sz_psf)

    center=int(psf.shape[0]/2.0)
    print(center,psf.shape)
    psf=psf[int(center-r_psf) : int(center+r_psf+1), int(center-r_psf) : int(center+r_psf+1)]

    imresid = im.copy()
    snrresid = snr.copy()

    xo,yo = np.round(wcsi.world_to_pixel(coords),0).astype(int)

    mask = ~((xo<0)|(xo>=im.shape[1])|(yo<0)|(yo>=im.shape[0]))

    s2flux= np.zeros(len(coords))
    s2err= np.zeros(len(coords))
    s2snr= np.zeros(len(coords))

    s2flux[mask] = im[yo[mask],xo[mask]].copy()
    s2err[mask] = ns[yo[mask],xo[mask]].copy()
    s2snr[mask] = snr[yo[mask],xo[mask]].copy()
    s2flux[~mask] = np.nan
    s2err[~mask] = np.nan
    s2snr[~mask] = np.nan

    ra = np.array([])
    dec = np.array([])
    x = np.array([])
    y = np.array([])
    err_out = np.array([])
    flux_out = np.array([])
    snr_out = np.array([])

    inds = np.argsort(s2flux)
    sortedcds = coords[np.flip(inds)].copy()
    sortedmask = mask[np.flip(inds)]
    for i,cd in enumerate(sortedcds):
        try:         
            cut = Cutout2D(imresid, cd, wcs=wcsi,size=8*u.arcsec,copy=True,mode='partial',fill_value=np.nan)
            peak = peak_local_max(cut.data, num_peaks=1)
            if len(peak) > 0:
                yp=peak[0,0]
                xp=peak[0,1]
        #         plt.imshow(cut.data)
        #         plt.colorbar()
        #         plt.plot(xp,yp,'bo')
        #         plt.show()

            else:
                yp,xp = np.unravel_index(np.argmax(cut.data), cut.data.shape)

            xorig,yorig = cut.to_original_position((xp,yp))
            flux_detected = imresid[yorig,xorig]
            flux_out = np.append(flux_out, round(flux_detected,6) )
            snr_out = np.append(snr_out, round(snrresid[yorig,xorig],6) )
            err_out = np.append(err_out, ns[yorig,xorig])

            ra_i,dec_i = wcsi.wcs_pix2world(xorig,yorig,0) #note reversal of x and y
            #print(type(ra_i), ra_i)
            ra  = np.append(ra , round(float(ra_i),8) )
            dec  = np.append(dec , round(float(dec_i),8) )
            x = np.append(x, xorig)
            y = np.append(y,yorig)

            #remove source from image and snr
            imresid[yorig-r_psf :yorig+r_psf+1, xorig-r_psf :xorig+r_psf+1] -= psf*flux_detected
            snrresid[yorig-r_psf :yorig+r_psf+1, xorig-r_psf :xorig+r_psf+1] -= psf*flux_detected/ns[yorig-r_psf :yorig+r_psf+1, xorig-r_psf :xorig+r_psf+1]
        except:
            flux_out = np.append(flux_out, np.nan )
            snr_out = np.append(snr_out, np.nan )
            err_out = np.append(err_out, np.nan)
            ra  = np.append(ra , np.nan)
            dec  = np.append(dec , np.nan)
            x = np.append(x, np.nan)
            y = np.append(y,np.nan)

    ### TODO
    # fluxes = np.zeros(len(coords))
    # errors = np.zeros(len(coords))
    # snrs = np.zeros(len(coords))
    # fluxes[np.flip(inds)] = flux_out
    # errors[np.flip(inds)] = err_out
    # snrs[np.flip(inds)] = snr_out

    return flux_out,err_out, snr_out, imresid, snrresid, inds


def compute_false_positives(im, ns, wcsi, psf,r=3,r_psf=40):
    imr=im.copy()
    n = 0
    iters=0
    center=int(psf.shape[0]/2.0)
    psf=psf[int(center-r_psf) : int(center+r_psf+1), int(center-r_psf) : int(center+r_psf+1)]

    fluxes=[]
    errs = []
    ppas = int(np.abs(1/wcsi.pixel_scale_matrix[0,0]/3600))
    print(ppas)
    while (n <= 5000) and (iters < 10000):
#         print(n, iters)
        x = np.random.randint(61,imr.shape[1]-61)
        y = np.random.randint(61,imr.shape[0]-61)
        if imr[y,x] !=0:
            cut = imr[y-r*ppas:y+r*ppas+1,x-r*ppas:x+r*ppas+1]
            cuterr = ns[y-r*ppas:y+r*ppas+1,x-r*ppas:x+r*ppas+1]
            try:
                peak = peak_local_max(cut, num_peaks=1)
                if len(peak) > 0:
                    yp=peak[0,0]
                    xp=peak[0,1]
#                     plt.imshow(cut.data)
#                     plt.colorbar()
#                     plt.plot(xp,yp,'bo')
#                     plt.show()

                else:
                    yp,xp = np.unravel_index(np.argmax(cut), cut.data.shape)
        
            except:
                yp,xp = np.unravel_index(np.argmax(cut), cut.data.shape)
                print('exception!')
                
            flux = cut[yp,xp]
            err = cuterr[yp,xp]
            
            if flux!=0:
                n+=1
                fluxes.append(flux)
                errs.append(err)
                yorig = y-r*ppas+yp; xorig = x-r*ppas+xp
#                 plt.imshow(imr)
#                 plt.plot(xorig,yorig,'ro')
#                 plt.show()
#                 print(flux)
                imr[yorig-r_psf :yorig+r_psf+1, xorig-r_psf :xorig+r_psf+1] -= psf*flux
#                 plt.imshow(imr[yorig-100:yorig+101,xorig-100:xorig+101])
#                 plt.show()
   
        iters+=1
    
    return fluxes, errs

def gen_rand_cds(im, wcsi, otherwcs = None,inputshape=None, size=5000):
    from reproject import reproject_interp

    if otherwcs:
        im_to_use,fp = reproject_interp((im, wcsi), otherwcs, shape_out=inputshape, parallel=12)
    else: im_to_use = im

    yrange, xrange = im_to_use.shape
    x = np.random.randint(low=0,high=xrange,size=size)
    y = np.random.randint(low=0,high=yrange,size=size)

    ra, dec = otherwcs.all_pix2world(y,x,0)
    coords = SC(ra, dec, unit='deg')
    return coords    