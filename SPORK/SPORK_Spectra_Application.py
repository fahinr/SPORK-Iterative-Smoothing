import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy import interpolate
from astropy.io import fits

def initialize_knots(wmin, wmax, knot_spacing):
    """ Place knots evenly through """
    waverange = wmax - wmin - 2*knot_spacing
    Nknots = int(waverange // knot_spacing)
    minknot = wmin + (waverange - Nknots * knot_spacing)/2.
    xknots = np.arange(minknot, wmax, knot_spacing)
    # Make sure there the knots don't hit the edges
    while xknots[-1] >= wmax - knot_spacing: xknots = xknots[:-1]
    while xknots[0] <= wmin + knot_spacing: xknots = xknots[1:]
    
    return list(xknots)

def fit_continuum_lsq(spec, knots, exclude=[], maxiter=3, sigma_lo=2, sigma_hi=2,
                      get_edges=False, **kwargs):
    """ Fit least squares continuum through spectrum data using specified knots, return model """
    assert np.all(np.array(list(map(len, exclude))) == 2), exclude
    assert np.all(np.array(list(map(lambda x: x[0] < x[1], exclude)))), exclude
    x, y, w = np.linspace(0,len(spec),len(spec)), np.array(spec) , np.array([x/10 for x in np.array(spec)])
              
    # This is a mask marking good pixels
    mask = np.ones_like(x, dtype=bool)
    # Exclude regions
    for xmin, xmax in exclude:
        mask[(x >= xmin) & (x <= xmax)] = False
    # Get rid of bad fluxes
    mask[np.abs(y)<1e-6] = False
    mask[np.isnan(y)] = False
    if get_edges:
        left = np.where(mask)[0][0]
        right = np.where(mask)[0][-1]
    
    for iter in range(maxiter):
        # Make sure there the knots don't hit the edges
        wmin = x[mask].min()
        wmax = x[mask].max()
        while knots[-1] >= wmax: knots = knots[:-1]
        while knots[0] <= wmin: knots = knots[1:]
        
        try:
            fcont = interpolate.LSQUnivariateSpline(x[mask], y[mask], knots, w=w[mask], **kwargs)
        except ValueError:
            print("Knots:",knots)
            print("xmin, xmax = {:.4f}, {:.4f}".format(wmin, wmax))
            raise
        # Iterative rejection
        cont = fcont(x)
        sig = (cont-y) * np.sqrt(w)
        sig /= np.nanstd(sig)
        mask[sig > sigma_hi] = False
        mask[sig < -sigma_lo] = False
    if get_edges:
        return fcont, left, right
        
    return fcont

def spork(spec, N_knots, x, sigma_hi, sigma_low):
    return fit_continuum_lsq(np.array(spec), knots=np.array(initialize_knots(0,len(x),N_knots)).astype(int),
                             maxiter=5, sigma_lo=sigma_low, sigma_hi=sigma_hi,get_edges=False)(np.linspace(0,len(x),len(x)))




### Loading and plotting spectra ###

all_spectra = np.load('FILE PATH') # Inputting the file path for the flux values desired. This will give you a 60x1024 array. Each row is one spectrum. 

plt.imshow(all_spectra) # This will make a 2D plot of all the spectra previously inputted. 

one_spectrum = all_spectra[0] # Here is the 0th spectrum & it's plot.
plt.plot(one_spectrum) 


### Applying Spork ###

# Ex for Spork, sp = spork(one_spectrum, 5*, np.linspace(0,len(one_spectrum),len(one_spectrum)**), 1***, 6****)
# * the degree of fit, where too high is a bad fit, and too low is a tight fit.
# ** the wavelength array, which will usually be [0,1,2...1024].
# *** the lower sigma clipping = degree to which low outliers are ignored
# **** the upper sigma clipping = degree to which high outliers are ignored

# Changing the knot spacing (*), and the upper/lower sigma clipping values will alter the fit to the data. 


sp = spork(one_spectrum, 5, np.linspace(0,len(one_spectrum),len(one_spectrum)), 1, 5)

# Plotting real flux data against SPORK's fit of the real flux data.
plt.plot(one_spectrum)
plt.plot(j)


