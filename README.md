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
import pandas as pd
```

Load incidence data (numpy array or pandas series) - here dummy
```p
incidence = pd.read_csv('incidence.csv')
```
*RtModel* class
```p
model = epy.RtModel()
```
Run a parametric estimation over our incidence data using a serial interval that is gamma distributed with mean of 4.7 and standard deviation of 2.3. We run this over a window of 6 days.

```p
estimated_rt_parametric = model.calc_parametric_rt(incidence,
                                        mean_si=4.7,
                                        std_si=2.3,
                                        win_start=1,
                                        win_end=7, mean_prior=2.3, std_prior=2)
```

Run a sampling estimation by providing a truncated normal distribution for mean & std of the serial intervals to explore. Also provide number of simulations and number of draws from the posterior samples

```p
estimated_rt_sampled = model.calc_sampling_rt(incidence,
                                        sample_mean_truncnorm=(5, 2, 3, 7),
                                        sample_std_truncnorm=(2,1,1,3),
                                        n_si_sims=100,
                                        n_posterior_samples=100,
                                        win_start=1,
                                        win_end=7,
                                        mean_prior=2.3,
                                        std_prior=2)
```

Each of these returns an object (*ParametricOutput* and *SamplingOutput*) with the estimated data. Each object contains a dataframe with all the outputs.

```p
estimated_rt_sampled.dataframe.head(10)
```