# SPectral cOntinuum Refinement for telluriKs (SPORK) & Iterative Smoothing

This Github Page is directly apart of and in reference of the [SPORK research paper](https://arxiv.org/abs/2108.12057).

SPORK is a spectrum normalization routine adapted from  the  stellar  abundance  determination  software [Spectroscopy Made Hard(er)](https://github.com/andycasey/smhr). The iterative smoothing counterpart is based on the python scipy package *interpolate*.

The SPORK algorithm is developed to effectively discover a continuum among unwanted features in spectra.
Cubic splines of smooth interopolations are iteratively fit to minimize errors along with removing outlier pixels to create a more robust estimation by sigma-clipping. Splines can arbitrarily be placed while the values of the lower sigma-clipping threshold is set to 1.0 with upper sigma-clipping thresholds set to 5.0. Features outside these thresholds are to be ignored since they are usually tellurics, cosmic rays, etc.

The algorithm parameters begin with the actual spectrum first, followed by the knot spacing, which is the degree of fit that we set to 5.0, where too high is a bad fit and too low is a tight fit. Next is the wavelength array, followed by the lower and upper sigma clipping degrees.

The repository features an instruction on how to apply and understand the parameters of SPORK, SPORK with Iterative Smoothing, SPORK without Iterative Smoothing, and Iterative Smoothing alone, with telluric removals. The files have their respective titles along with sample spectra to test on (algn_regr.npy).


