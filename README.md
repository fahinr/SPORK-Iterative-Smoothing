# SPectral cOntinuum Refinement for telluriKs (SPORK) & Iterative Smoothing

SPORK is a spectrum normalization routine adapted from  the  stellar  abundance  determination  software [Spectroscopy Made Hard(er)](https://github.com/andycasey/smhr). The iterative smoothing function is based on the python scipy package *interpolate*. Together, SPORK and iterative smoothing, becomes a powerful tool to remove unwanted noise. 

SPORK is designed specifically to locate a continuum even in the presence of many large dips, such as the absorption features one encounters in stellar spectra.  The tool iteratively fits a univariate natural cubic spline and identifies outlier pixels to be sigma-clipped. Each order of an echelle spectrum has highly varying signal-to-noise as a function of wavelength, so the spectrum uncertainties reported by the pipeline are used to perform sigma clipping (rescaling by the standard deviation of the error-normalized deviations). 

Spline knots can be placed arbitrarily. The lower sigma-clipping threshold is set to 1.0, as points even a little below the continuum belong to stellar or telluric lines and should be ignored. The upper sigma clipping threshold is set to 5.0, as features which rise sharply above a stellar spectrum are typically cosmic rays which should also be ignored.


