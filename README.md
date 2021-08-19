# SPectral cOntinuum Refinement for telluriKs (SPORK) & Iterative Smoothing

SPORK is a spectrum normalization routine adapted from  the  stellar  abundance  determination  software Spectroscopy Made Hard(er)[https://github.com/andycasey/smh]. The iterative smoothing function is based on the python scipy package _interpolate_.

SPORK is designed specifically to locate a continuum even in the presence of many large dips, such as the absorption features one encounters in stellar spectra.  The tool iteratively fits a univariate natural cubic spline and identifies outlier pixels to be sigma-clipped. Each order of an echelle spectrum has highly varying signal-to-noise as a function of wavelength, so the spectrum uncertainties reported by the pipeline are used to perform sigma clipping (rescaling by the standard deviation of the error-normalized deviations). Spline knots can be placed arbitrarily, but weuseN/2−1evenly spaced knots for each order (whereNis length of the wavelength array). 
