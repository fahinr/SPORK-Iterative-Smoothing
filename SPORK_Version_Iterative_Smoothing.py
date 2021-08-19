import csv
import matplotlib.gridspec as gridspec
import math

import sys  

import matplotlib
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from scipy import stats
import numpy as np
import pandas as pd

from astropy.io import fits
from scipy.interpolate import splrep, splev, interp1d
from scipy.interpolate import CubicSpline
from scipy import interpolate, signal

#### Preliminary Calculations ####

rplanet=(.143 *const.R_sun) / (u.m)   # Planet radius
rstar=(1.19 * const.R_sun) / (u.m)    # Stellar radius
rRatio = (rplanet/rstar)**2           # Area ratio
rRatio = float(rRatio)
tEff = 6091                           # Stellar effective temperature
periodsec = 3.525*24*3600 * u.s
inc = 86.71 * np.pi / 180.0 * u.deg
starmass = 1.23 * const.M_sun #in kg
vOrb = np.cbrt((2.*np.pi*const.G*starmass) / periodsec) 
print('Planet orbital velocity is',vOrb)


#### FUNCTIONS ####

  ### SPORK ###
  
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


########################################################

def loadinspec(night):
    nFold = 'n'+str(night)+'/'
    alg = np.load(nFold+'algn_regr.npy')
    wlen = np.load(nFold+'wlen_regr.npy')
    data = fits.getdata(nFold+'data.fits')
    HJD, = data.field('hjd')
    rvobs, = data.field('RVOBS')
    rvsys = data.field('RVSYS')
    phaseval, = data.field('ph')
    t0 = data.field('t0') #52826.82851
    period = 3.52474859
    nf, = phaseval.shape # Counting the number of frames
    phase = (HJD - t0) / period
    # Taking the modulus to neglect integer phases
    phase %= 1 
    # Printing diagnostics
    print('Phase difference with data.fits:',np.mean(phase-phaseval))
    rvtot = rvobs + rvsys   # In km/s
    print('Systemic+barycentric RV range:',rvtot[[0,-1]])
    
    return alg,wlen,rvtot,phase

''' This functions computes the black body radiation B_nu in frequency SI units 
[W m-2 Hz-1] given temperature in K and wavelengths in meters. '''
def blackbody(T,wl):
    # Define constants used in calculation
    h = 6.626E-34
    c = 2.998E8
    k = 1.38E-23
    nu = c / wl
    c1 = 2.0 * np.pi * h / c**2    
    c2 = h / (k*T)
    val = c2 * nu
    return c1 * nu**3 / (np.exp(val) - 1.0)

''' This function injects a model spectrum - including the broadening due to
full atmospheric circulation - into the data, with parameters:
- scale: a scaling factor for the model
- wData: the wavelength solution of the data (in nanometers)
- fData: the aligned spectral series (in ADU/s)
- rvtot: barycentric + systemic velocity
- night: (1 or 2) the night to process
- kp: the planet RV semi-amplitude to use for the injection
- ph: the vector of orbital phases of length = axis 1 of fData
- rr: (Rp/R*)**2, or the area ratio between planet and star '''

def injectmodel(scale,wData,fData,rvtot,night,kp,ph,rr,tEff):
    no, nf, nx = fData.shape
    plRV = rvtot + kp * np.sin(2.0 * np.pi * ph)
    for j in range(nf):
        # Reading in model to inject
        modName = 'MODEL .DAT FILE PATH'
        wMod, fMod = np.loadtxt(modName, unpack=True)
        # Scaling of the model
        fMod *= ( rr / blackbody(tEff,wMod) )
        wMod *= 1E9   # In nanometers to match data
        # Model needs convolution to CRIRES IP. All models have the same wlen scale
        # so computation of the broadening kernel is only necessary once.
        if j == 0:
            dlam_lam_model = np.mean(2 * (wMod[1:]-wMod[0:-1]) / (wMod[1:]+wMod[0:-1]))
            dlam_lam_crires = 1.0 / 9E4
            # FWHM of CRIRES instrumental profile in model pixels
            fwhm_pix = dlam_lam_crires / dlam_lam_model
            # Conversion FWHM -> Gaussian sigma
            sigma_px = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            # Computing the convolution kernel
            xker = np.arange(61) - 30
            yker = np.exp(-0.5 * (xker/sigma_px)**2)
            yker /= yker.sum() #Normalisation
        # Convolution of model with CRIRES Gaussian IP
        fMod = np.convolve(fMod, yker, mode='same')
        # Spline interpolation of the convolved model
        cs = splrep(wMod, fMod, s=0)
        
        # Shifting the data wavelengths based on RVs
        wShift = wData * (1.0 - plRV[j] / 2.998E5)
        fShift = splev(wShift, cs, der=0)
        # Scaling the model spectrum
        fShift *= scale
        # Injecting by multiplying aligned data by (1 + Fp/Fs)
        for io in range(no):
            fData[io,j,] *= (1 + fShift[io,])
    return fData

############## TELLURIC REMOVAL WITH SPORK NORMALIZATION ################

'''Inputs:
- spec: observed spectral series (in ADU/s)
- smoothing_factor: inputted via iterative smoothing '''

def remove_tellurics_p(spec, smoothing_factor):
    #sp_wave = spec[0,1,:].copy()
    no, nf, nx = spec.shape        # Getting dimensions
    yy = np.arange(nf)             # Used to detrend in time (see loop below)
    for io in range(no):
        spec[io,:,[0,1,1022,1023]] = 1.0
        spc = spec[io,].copy()     # Slicing detector
        # Creating the time average of the spectra - robust against NaNs
        mspc = np.nanmedian(spc,axis=0)
        
        # looping over phases
        for j in range(nf):
            spc = spec[io,j,].copy()
            cs = splrep(np.linspace(0,1024,1024),mspc, s = smoothing_factor) 
            f = splev(np.linspace(0,1024,1024),cs,der=0)
            # Excluding NaNs
            iok = np.isfinite(spc)
            # Computing the 2nd order fit on the finite elements only
            spc /= spork(spc,511,np.linspace(0,1024,1024),1,5)/np.max(spc)
            
            coef = np.polyfit(f[iok],spc[iok],2)
            fit = coef[0]*f**2 + coef[1]*f + coef[2]
          
            # Divide by the fit
            spec[io,j,] /= fit
           
        # Loop over wavelfdengths to detrend in time
        for i in range(nx):
            sp_chann = spec[io,:,i].copy()
            iok = np.isfinite(sp_chann)
            coef = np.polyfit(yy[iok],sp_chann[iok],2)
            fit = coef[0]*yy**2 + coef[1]*yy + coef[2]
            
            spec[io,:,i] /= fit
        # Masking the edges (almost always bad)
    
    return spec-1.0 
            

''' More aggressive masking based on the number of standard deviations of a 
column above the average standard deviation. Takes in input the telluric
removed, prossibly already masked matrix. '''
def mask_data(spec):
    no, nf, nx = spec.shape
    # no = 4 detectors
    # nf = 59 exposures
    # nx = 1024 pixels 
    for io in range(no):
        spc = spec[io,].copy()
        sig = np.std(spc,axis=0)
        mstd = np.median(sig)
        for i in range(nx):
            if sig[i] > 2.0*mstd: spc[:,i] = 0.0
        spec[io,] = spc
    return spec

def masking(night,alg):
    if night == 1:
        alg[1,:,821:833]=0
        alg[1,:,888:901]=0
        alg[3,:,0:23]=0 
        alg[3,:,521:524]=0 
    elif night == 2:
        alg[0,:,0:3]=0
        alg[0,:,1021:1025]=0
        alg[1,:,417:420]=0
        alg[1,:,821:834]=0
        alg[1,:,889:901]=0
        alg[2,:,0:4]=0
        alg[3,:,0:24]=0
        alg[3,:,510:513]=0 
        alg[3,:,521:524]=0 
        alg[3,:,1021:1025]=0 
    
    return alg

''' Cross correlation *directly at the requested Kp and vrest*, slower than the
grid approach but it baypasses the need for shifting and co-adding. Since this 
is meant to be used to estimate chi-square statistics, the CCF is sampled on a
coarser grid to avoid issues with oversampling (= correlation between adjacent 
RV points). Inputs:
- wData: data wavelengths in nanometers
- fData: telluric-removed, masked data
- kp: requested planet RV semi-amplitude
- vrest: requested additional planet rest-frame velocity (scalar)
- ph: orbital phases
- rvtot: systemic + barycentric velocities (same dimension as ph)
- night: (1 or 2) the night being processed
- hipass: apply a high-pass filter to the individual CCFs '''
def cc_at_vrest(wData, fData, kp, ph, rvtot, night, hipass=False):
    ncc = 101
    rvlag = np.linspace(-150,150,ncc)
    no, nf, nx = fData.shape
    ccf = np.zeros((no-1,nf,ncc))   # Excluding fourth detector
    for j in range(nf):
        modName = 'MODEL .DAT FILE PATH'
        wMod, fMod = np.loadtxt(modName, unpack=True)
        wMod *= 1E9     # meters -> nm
        cs = splrep(wMod, fMod, s=0)
        #print(cs)
        lagTemp = rvlag + rvtot[j] + kp * np.sin(2*np.pi*ph[j])
        for ii in range(100):
            wShift = wData * (1.0 - lagTemp[ii]/2.998E5)
            fShift = splev(wShift,cs,der=0)
            for io in range(no-1):
                fVec = fData[io,j,].copy()
                gVec = fShift[io,].copy()
                iok = np.isfinite(fVec) * np.isfinite(gVec)
                ccf[io,j,ii] = mattcc(fVec[iok],gVec[iok])
    # Removing gradients and trends in the CCF
    if hipass:
        nbin = 10
        bins = int(ncc / nbin)
        xb = np.arange(nbin)*bins + bins/2.0
        yb = np.zeros(nbin+2)
        xb = np.append(np.append(0,xb),ncc-1)
        for io in range(no-1): 
            for j in range(nf):
                for ib in range(nbin):
                    imin = int(ib*bins)
                    imax = int(imin + bins)
                    yb[ib+1] = np.mean(ccf[io,j,imin:imax])
                cs_bin = splrep(xb,yb,s=0.0)
                fit = splev(np.arange(ncc),cs_bin,der=0)
                ccf[io,j,] -= fit
    return ccf

''' Fast cross correlation function with full formula (mean subtraction and 
normalisation by variance). Takes in input the two vectors to cross correlate, 
the data (fVec) and the model (gVec). '''

def mattcc(fVec, gVec):
    N, = fVec.shape
    Id = np.ones(N)
    fVec -= (fVec @ Id) / N
    gVec -= (gVec @ Id) / N
    sf2 = (fVec @ fVec)
    sg2 = (gVec @ gVec)
    
    return (fVec @ gVec) / np.sqrt(sf2*sg2)
  
  
#### LOADING IN NIGHT ####

algn1,wlen1,rvel1,ph1 = loadinspec(1)
algn2,wlen2,rvel2,ph2 = loadinspec(2)
for io in range(4):
    plt.figure(figsize=(12,3))
    plt.imshow(algn2[io,])
    plt.show()
    

# Kp is the orbital velocity of the PLANET
# vrest is the total velocity of the SYSTEM

# don't change these numbers yet. we just want to test our code at the maximum detection value which has 
# already been determined to be 149,0.

kpVec = np.arange(149,150,1) #124,179,3
vrestVec = np.arange(0,1,1) #-21.,21.5,3
nkp = kpVec.shape
nvrest, = vrestVec.shape
print('K_P:',kpVec)
print('V_rest:',vrestVec)



# Output RV vector - enough to cover sections to estimate the noise and the area around the peak
ncc = 101                       # MUST MATCH rvlag in code.cc_at_vrest()
xx = np.linspace(-150,150,ncc)  # MUST MATCH rvlag in code.cc_at_vrest()
# Selecting RV values to compute the variance of the data (iout) and evaluate the chi square (iin)
iout = np.abs(xx) >= 50         # boolean 
nout = iout.sum()               # how many 'trues'
iin = np.abs(xx) < 50           # boolean
nin = iin.sum()                 # how many 'falses'
print(nin, nout)
# Initialising the matrices that will contain the delta(sigma) values
dSigma = np.zeros((nkp,nvrest))
hikey = True 




#### ITERATE OVER KP LOOP ####

# All iterations will be inputted into a csv file with 
# each smoothing value's respective size and dsigma.

fields = ['SmoothingSize', 'Dsig']
filename = "Smoothing_Values.csv"


maxsum = 0
maxindex = 0

with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
    #Iterate over logarithmic spaces
    for i in np.geomspace(0.01, 1, num=1000):

        telRem1 = remove_tellurics_p(algn1.copy(), i)
        telRem2 = remove_tellurics_p(algn2.copy(), i)

        telRem1 = mask_data(masking(1,telRem1))
        telRem2 = mask_data(masking(2,telRem2))
        for ik in range(nkp):

            kpvalue = kpVec[ik].copy()
            # Cross correlating the real data
            ccfReal1 = cc_at_vrest(wlen1, telRem1, kpvalue, ph1, rvel1, 1, hipass=hikey)
            ccfReal2 = cc_at_vrest(wlen2, telRem2, kpvalue, ph2, rvel2, 2, hipass=hikey)
            print('ccfReal complete')
            ccfReal = np.append(ccfReal1, ccfReal2, axis=1)
            no, nf, nx = ccfReal.shape
            stdReal = np.sum(np.sum(np.std(ccfReal, axis=2),axis=0),axis=0) / np.sqrt(nf*no)
            ccfReal = ccfReal.sum(axis=0)   # Summing over detectors
            ccfReal = ccfReal.sum(axis=0)   # Summing over time
            # Doing statistical tests on real data
            varReal = np.std(ccfReal[iout])**2    # CCF variance (away from peak)
            chi2Real = (ccfReal[iin]**2).sum() / varReal   # Chi-square of peak
            sigmaReal = stats.norm.isf(stats.chi2.sf(chi2Real,nin-1)/2)   # Chi-square -> sigma
            print('Kp={:3.1f}, chi2={:3.1f}, dof={:2}, sigma={:2.1f}'.format(kpvalue, chi2Real, nin-1, sigmaReal))
            # Looping over rest-frame velocities and retrieving the injected signal
            for iv in range(nvrest):
                # Injecting the model spectrum at the (Vrest, Kp) to test
                algnInj1 = injectmodel(3, wlen1, algn1.copy(), rvel1+vrestVec[iv], 1, kpvalue, ph1, rRatio, tEff)
                algnInj2 = injectmodel(3, wlen2, algn2.copy(), rvel2+vrestVec[iv], 2, kpvalue, ph2, rRatio, tEff)

                # Removing tellurics from the injected matrices


                telRemInj1 = mask_data(masking(1,remove_tellurics_p(algnInj1, i)))
                telRemInj2 = mask_data(masking(2,remove_tellurics_p(algnInj2, i)))

                # Cross correlating the injected spectra
                ccfInj1 = cc_at_vrest(wlen1, telRemInj1, kpvalue, ph1, rvel1, 1, hipass=hikey)
                ccfInj2 = cc_at_vrest(wlen2, telRemInj2, kpvalue, ph2, rvel2, 2, hipass=hikey)
                ccfInj = np.append(ccfInj1, ccfInj2, axis=1)
                ccfInj = ccfInj.sum(axis=0)
                ccfInj = ccfInj.sum(axis=0)
                # Subtract the real CCF to get the noiseless model CCF
                ccfModel = ccfInj - ccfReal
                # Scale the noiseless model CCF to the real CCF - Need to impose slope > 0 (correlation)
                # so the absolute value of the slope is taken. This means that an anti-correlation
                # (negative slope) will increase the chi-square of the residuals rather than decrease
                # it, resulting into a delta(sigma) value < 0.
                coef = np.polyfit(ccfModel, ccfReal, 1)
        
                coef[0] = np.abs(coef[0])
                ccfFit = coef[0]*ccfModel + coef[1]
                # Computing the residual cross correlation function
                ccfRes = ccfReal - ccfFit
                # Doing statistical tests on residual cross correlation function.
                # Note that dof decreases by 2 because we are fitting an apmplitude and 
                # offset of the model CCF.
                varModel = np.std(ccfRes[iout])**2   # CCF variance (away from peak) 
                chi2Model = (ccfRes[iin]**2).sum() / varModel
                sigmaModel = stats.norm.isf(stats.chi2.sf(chi2Model,nin-3)/2)
                # Populating the dSigma matrix
                dSigma[ik,iv] = sigmaReal - sigmaModel
                print('- Vrest={:2.1f}, sig_model={:2.1f}, dsig={:2.1f}'.format(vrestVec[iv], sigmaModel, dSigma[ik,iv]))

                maxv = dSigma[ik,iv]
                
                # writing the data rows 
                rows = [ [i, dSigma[ik,iv]] ]
                csvwriter.writerows(rows)

                # Grab maximum smoothing size and dsigma
                if  maxv > maxsum:
                    maxsum = maxv
                    maxindex = i
        
                
print('Max dsig = ', maxsum)
print('Max size = ', maxindex)
