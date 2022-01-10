# SPectral cOntinuum Refinement for telluriKs (SPORK) & Iterative Smoothing

This Github Page is directly apart of and in reference of the [SPORK research paper](https://arxiv.org/abs/2108.12057). More information on how the tool is set and used can be found here.

## What is SPORK & Iterative Smoothing?

SPORK is a **spectrum normalization routine** adapted from  the  stellar  abundance  determination  software [Spectroscopy Made Hard(er)](https://github.com/andycasey/smhr). The iterative smoothing counterpart is based on the python scipy package *interpolate*. Iterative smoothing maximizes SPORK by iterating logarithmically over a specific smoothing value to obtain higher significances.  

## How does it work?

The SPORK algorithm is developed to effectively **discover a continuum among unwanted features in stellar spectra**.
Cubic splines of smooth interopolations are iteratively fit to minimize errors along with removing outlier pixels to create a more robust estimation by sigma-clipping. Splines can arbitrarily be placed while sigma-clipping can be separated by the values of the lower sigma-clipping threshold, which is set to 1.0, and the upper sigma-clipping threshold, which is set to 5.0. Features outside these thresholds are to be ignored since they are usually tellurics, cosmic rays, stellar lines etc.

## What are the parameters?

The algorithm parameters **begin with the actual spectrum first**, followed by the **knot spacing**, which is the degree of fit from the algorithm into the spectrum that we set to **5.0**, where too high is a bad fit and too low is a tight fit. Next is the **wavelength array**, usually a 1024 array, followed by the **lower and upper sigma clipping** degrees (**1.0 and 5.0**, respectively, as mentioned above).

Ultimately, SPORK is a general smoothing/normalization routine that is used over spectra. When used with specific parameters and tweaked with iterative smoothing along with telluric removals with a median spectrum, it becomes a powerful tool to remove unwanted noise and obtain as high of a significance as possible.

The repository features an instruction on how to apply and understand the parameters of SPORK (`SPORK_Spectra_Application.py`), SPORK with Iterative Smoothing (`SPORK_Version_Iterative_Smoothing.py`), SPORK without Iterative Smoothing (`SPORK_Version_Telluric_Removal_Only.py`), with telluric removals (which removes any small fluctuating residuals) and finally, Iterative Smoothing with no other dependencies (no SPORK or telluric removal) (`Iterative_Smoothing_No_Tellurics.py`). The files have their respective titles along with sample spectra to test on (`algn_regr.npy`).



**Code created and modified by authors of SPORK**

