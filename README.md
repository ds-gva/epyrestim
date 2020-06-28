# epyrestim
*Warning: This package is work-in-progress and highly preliminary. Breaking changes may occur and the authors cannot guarantee support*

**epyrestim** is a Python 3 implementation of various Rt calculation methods, inspired from and borrowing (heavily) from R's Epiestim package. The original objective of this package is to provide an opportunity for data scientists working primarily in Python to access these modelling tools without having to bridge to R.

All credit for the original models and code go to the EpiEstim team (Cori et al., AJE 2013).

Currently the package provides two implementations of the "Cori" method of calculating Rt:
* *Parametric SI* :  uses the method proposed in Cori et al. (AJE, 2013) for estimating Rt over various time windows, using the incidence curve and assumptions on a Gamma-distributed serial interval distribution.
* *SI From Sampling* : uses the same method as in Cori et al. (AJE, 2013) allowing to add some uncertainty around the serial interval distribution

*Note*: this package does not allow, unlike EpiEstim, to differentiate between local and imported cases. We hope to add that feature in the near future

## Installation
There is currently no installation, you can just clone the whole package and start using it. We aim to provide the package on PyPi very soon.

## Quick start
Load the modelling module
```p
import epyrestim.model as epy
```

Load incidence data (numpy array or pandas series) - here dummy
```p
incidence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```
*estimators* class
```p
estimator = epy.estimators()
```
Run a parametric estimation over our incidence data using a serial interval that is gamma distributed with mean of 4.7 and standard deviation of 2.3. We run this over a window of 6 days.

```p
estimated = estimator.Rt_parametric_si(incidence,
                                        mean_si=4.7,
                                        std_si=2.3,
                                        win_start=2,
                                        win_end=8)
```