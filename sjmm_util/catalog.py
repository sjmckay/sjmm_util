import numpy as np
from astropy.coordinates import SkyCoord as SC
from scipy.stats import bootstrap
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
    '''
    Create SkyCoord array from an Astropy table with separate RA/Dec columns for hours, minutes, etc.
    
    Parameters
    ----------
    tab (astropy.table.Table or QTable): table with RA/Dec columns with format "RAh", "DEm", etc.
    
    Returns
    coords (astropy.coordinates.SkyCoord): SkyCoord object derived from the table RA and Dec columns.
    
    '''
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
    '''
    Add SkyCoord column to table as single RA and Dec columns

    Parameters
    ----------
    tab (astropy.table.Table): table to add RA/Dec columns to

    coords (astropy.coordinates.SkyCoord): SkyCoord object to add to table.

    index (int): intended column index in table.

    Returns
    -------
    tab  (astropy.table.Table): the table with coord columns added.
    '''
    ra = [float(c.to_string(precision=6).split(' ')[0]) for c in coords]
    dec = [float(c.to_string(precision=6).split(' ')[1]) for c in coords]
    tab.add_column(ra, index = index, name = 'ra')
    tab.add_column(dec,index = index + 1, name = 'dec')
    return tab

def latex_pm(median, p16, p84, precision=2):
    """format median and upper/lower errors for latex table"""
    upper = p84 - median
    lower = median - p16
    
    fmt = f"{{:.{precision}f}}"
    m = fmt.format(median)
    u = fmt.format(upper)
    l = fmt.format(lower)
    
    return rf"${m}^{{+{u}}}_{{-{l}}}$"