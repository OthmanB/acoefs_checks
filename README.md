# acoefs_checks
A set of python function that evaluate the effects of activity and centrifugal forces on low-l modes
Some functions are mostly for visualisation purpose (acoefs.py and acoefs_effects.py), but others are for
performing tests by fitting a2 coefficients using minimisation + emcee either simulated (see fit_a2sig.py) or 
real. This allows to have (1) first rough evaluations on the parameters, (2) tests for the CPP routines, checking the convergence and biases.

Note: Fitting routines will require you to compile the Alm routine from https://github.com/OthmanB/TAMCMC-C/tree/dev/external/integrate  using the provided cmake file.

