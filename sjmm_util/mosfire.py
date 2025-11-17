import numpy as np
from astropy.table import Table
from .catalog import radec2string

# MOSFIRE approx limits/info
ha=0.656
hb=0.486
oii=0.3728

ylim=(0.972,1.124) 
hlim=(1.466,1.807)
jlim=(1.153,1.352)
klim=(1.921,2.404)


def to_magma(table, name='objfile'):
    with open(name+'.coords', 'w') as objfile:
        for obj in table:
            mag = np.round(obj['mag'], 3)
            objfile.write(f"{obj['name']} {obj['priority']} {mag} {radec2string(obj)} 2000.0 2000.0 0.0 0.0\n")
    return


def exclude_targets(table, path_to_output, high_priors):
    """Take existing MAGMA target list and remove or reassign object priorities (down by 1 dex). 
    Pass list of names as ``high_priors`` to be just reassigned priorities, sources with any other names will be removed from future masks."""
    newtab = table.copy()
    last_targets = Table.read(path_to_output,format='ascii')
    last_targets['col1'].name = 'name'
    to_remove = []
    for row in newtab:
        if (row['name'] in last_targets['name']) and ('star' not in row['name']):
            if np.any([i in row['name'] for i in high_priors]):
                if row['priority'] <= 10:
                    pass
                else:
                    row['priority'] /= 10
            else:
                to_remove.append(row.index)
    newtab.remove_rows(to_remove)
    return newtab


def line_limits(z,line,limits):
    return (z>(limits[0]/line-1))&(z<(limits[1]/line-1))


def check_zrange(z,bands=['y','j','h','k'],use_oii=False):
    z_y = line_limits(z,ha,ylim)|line_limits(z,hb,ylim)
    z_j = line_limits(z,ha,jlim)|line_limits(z,hb,jlim)
    z_h = line_limits(z,ha,hlim)|line_limits(z,hb,hlim)
    z_k = line_limits(z,ha,klim)|line_limits(z,hb,klim)

    if use_oii:
        z_y |= line_limits(z,oii,ylim)
        z_j |= line_limits(z,oii,jlim)
        z_h |= line_limits(z,oii,hlim)
        z_k |= line_limits(z,oii,klim)

    detectable = np.zeros(len(z),dtype=bool)
    if 'k' in bands:
        detectable |= z_k
    if 'h' in bands:
        detectable |= z_h
    if 'j' in bands:
        detectable |= z_j
    if 'y' in bands:
        detectable |= z_y
       
    return detectable
