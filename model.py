# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import gamma
import warnings

class estimators:
    # This is our estimated Rt object when using the parametric function
    class Rt_parametric:
        def __init__(self, a_posterior, b_posterior, mean_posterior, std_posterior, t_start, t_end, real_dates):
            self.a_posterior = a_posterior
            self.b_posterior = b_posterior
            self.mean_posterior = mean_posterior
            self.median_posterior = stats.gamma.ppf(0.5, a_posterior, scale=b_posterior)
            self.std_posterior = std_posterior
            self.t_start = t_start
            self.t_end = t_end
            self.real_dates = real_dates[len(real_dates)-len(mean_posterior):]
            self.dataframe = self.to_dataframe()

        def confidence_intervals(self):
            low_ci_05 = stats.gamma.ppf(0.05, self.a_posterior, scale=self.b_posterior)
            high_ci_95 = stats.gamma.ppf(0.95, self.a_posterior,scale=self.b_posterior)
            low_ci_025 = stats.gamma.ppf(0.025, self.a_posterior, scale=self.b_posterior)
            high_ci_975 = stats.gamma.ppf(0.975, self.a_posterior, scale=self.b_posterior)
            return low_ci_05, high_ci_95, low_ci_025, high_ci_975

        def to_dataframe(self):
            ci = self.confidence_intervals()
            rt_parametric_df = pd.DataFrame(data={'dates': self.real_dates,
                                    'Rt_mean': self.mean_posterior,
                                    'Rt_median': self.median_posterior,
                                    'low_ci_05': ci[0],
                                    'high_ci_95': ci[1],
                                    'low_ci_025': ci[2],
                                    'high_ci_975': ci[3]})
            return rt_parametric_df
    
    # This is the object for the Rt from SI sampling
    class Rt_sampled:
        def __init__(self, all_sampled, t_start, t_end, real_dates):
            self.all_sampled = all_sampled
            self.t_start = t_start
            self.t_end = t_end
            self.dataframe = self.to_dataframe(all_sampled, real_dates)

        def to_dataframe(self, samples, real_dates):
            """
            Converts the generated Rt samples from our estimation into a pandas dataframe that can be used for plotting or other.
            The dataframe also includes key summary statistics such as mean, median, standard deviation and confidence intervals

            Parameters
            ----------
            samples (ndarray): the samples generated from our estimated Rt
            real_dates: the series of dates from the original incidence data

            Returns:
            ----------
            Pandas dataframe with all Rt statistics and dates
            """
            results_df = pd.DataFrame(samples)
            
            # Calculate key statistics
            results_df['Rt_mean'] = results_df.mean(axis=1)
            results_df['Rt_median'] = results_df.median(axis=1)
            results_df['Rt_Std'] = results_df.std(axis=1)
            results_df['low_ci_05'] = results_df.quantile(q=0.05, axis=1)
            results_df['high_ci_95'] = results_df.quantile(q=0.95, axis=1)
            results_df['low_ci_025'] = results_df.quantile(q=0.025, axis=1)
            results_df['high_ci_975'] = results_df.quantile(q=0.975, axis=1)

            # Add the dates
            dates = real_dates[len(real_dates)-len(results_df['Rt_mean']):]
            results_df['dates'] = dates
            #results_df = results_df[results_df.index > (len(real_dates) - len(dates))]

            rt_sampled_df = pd.DataFrame({'dates': dates,
                                        'Rt_mean': results_df['Rt_mean'],
                                        'Rt_median': results_df['Rt_median'],
                                        'Rt_Std': results_df['Rt_Std'],
                                        'low_ci_05': results_df['low_ci_05'],
                                        'high_ci_95': results_df['high_ci_95'],
                                        'low_ci_025': results_df['low_ci_025'],
                                        'high_ci_975': results_df['high_ci_975'],
                                        })
            return rt_sampled_df

    ### This function returns the starting and ending time periods according to the passed windows
    # also returns the total number of time periods
    def calc_time_windows(self, incidence, win_start=2, win_end=8):
        # adjust the windows to work with python array index
        win_start -= 1
        win_end -= 1

        total_time = len(incidence)

        t_start = np.arange(win_start, total_time-(win_end-win_start))
        t_end = np.arange(win_end, total_time)

        nb_time_periods = len(t_start)

        return t_start, t_end, nb_time_periods

    ### Draw from discretized gamma distribution
    # This is the same implementaton as in Epiestim
    #K is a positive integer or vector
    # mu is the mean
    # sigma is the deviation
    def discr_si(self, k, mu, sigma):
        if sigma < 0:
            print("Error: sigma must be >0")
        if mu <= 1:
            print("Error: mu must be <=1")
        if any(k) < 0:
            print("Error: values in k must all be >0")

        # Shift -1
        a = ((mu - 1) / sigma)**2  # shape
        b = sigma**2 / (mu - 1)   # scale

        res = k * gamma.cdf(k, a, scale=b) + (k - 2) * gamma.cdf(k - 2,
                                                                 a, scale=b) - 2 * (k-1) * gamma.cdf(k-1, a, scale=b)
        res = res + a * b * (2 * gamma.cdf(k - 1, a + 1, scale=b) -
                             gamma.cdf(k - 2, a + 1, scale=b) - gamma.cdf(k, a + 1, scale=b))

        # Return largest of 0 or calculated value
        res2 = [max(0, x) for x in res]

        # Checked for consistency in results with EpiEstim
        return np.array(res2)

    ## Creates the posteriors from the selected SI distribution
    def posterior_from_si_distrib(self, incidence, si_distr, a_prior, b_prior, t_start, t_end):

        distrib_range = np.arange(0, len(si_distr))
        final_mean_si = np.sum(si_distr * distrib_range)
        lam = self.overall_infectivity(incidence, si_distr)

        posteriors = np.empty((len(t_start), 2))

        for i, start in enumerate(t_start):
            if t_end[i] > final_mean_si:
                a_post = a_prior + np.sum(incidence[start: t_end[i]+1])
                b_post = 1 / (1 / b_prior + np.sum(lam[start:t_end[i]+1]))
                posteriors[i] = a_post, b_post
            else:
                posteriors[i] = np.nan, np.nan

        return posteriors  # a_posterior, b_posterior

    ## Description here: https://www.rdocumentation.org/packages/EpiEstim/versions/2.2-1/topics/overall_infectivity
    ## Defines overall infectivity by calculating lambda at time t
    def overall_infectivity(self, incidence, si_distr):
        T = len(incidence)
        lam_t_vector = [np.nan]

        # For each day, we calculate LambdaT the infectivity. To do that we calculate all the infections to that day which we multiply be the probability of
        # infection (the serial distribution flipped)
        for i in range(1, T):
            infections = incidence[0:i+1]
            probabilities = np.flip(si_distr[0:i+1])
            lam_t = np.sum(infections * probabilities)
            lam_t_vector.append(lam_t)
        return lam_t_vector

    ## This is the main function that does all the work
        # Incidenc: our incidence series
        # mean_si: the mean of our serial interval distribution
        # std_si: standard deviation of our serial interval distribution
        # win_start, win_end: starting and ending period for the rolling window (NOTE: in actual time periods not Python index, i.e., 1 = 0)
        # mean_prior, std_prior: the mean and std of our Rt prior
    def Rt_parametric_si(self, incidence, mean_si, std_si, win_start=2, win_end=8, mean_prior=4, std_prior=4):
        # Find how many time periods we have
        T = len(incidence)

        real_dates = incidence.index

        # Create our discretized serial interval distribution
        si_distribution = self.discr_si(np.arange(0, T), mean_si, std_si)

        # Fill our overflowing distribution with 0s (no prob)
        if len(si_distribution) < (T+1):
            over = -(len(si_distribution) - (T+1))
            si_distribution = np.pad(si_distribution, (0, over), 'constant')

        # Return cumulative incidence per time window, starting window and ending window
        t_start, t_end, nb_time_periods = self.calc_time_windows(
            incidence, win_start=win_start, win_end=win_end)

        # Calculate the parameters of our gamma prior based on the provided mean and std of the serial interval
        a_prior = (mean_prior / std_prior)**2
        b_prior = std_prior**2 / mean_prior

        # Calculate our posteriors from our serial interval distribution
        post = self.posterior_from_si_distrib(
            incidence, si_distribution, a_prior, b_prior, t_start, t_end)
        a_posterior, b_posterior = post[:, 0], post[:, 1]

        mean_posterior = a_posterior * b_posterior
        std_posterior = np.sqrt(a_posterior) * b_posterior

        result = self.Rt_parametric(a_posterior, b_posterior, mean_posterior,
                               std_posterior, t_start, t_end, real_dates)
        return result

    # Estimate Rt while adding uncertainty around the SI
    def Rt_from_si_sampling(self, incidence, sample_mean_truncnorm, sample_std_truncnorm, n_si_sims=1, n_posterior_samples=1, win_start=2, win_end=8, mean_prior=4, std_prior=4):

        ## First part is same as for standard Rt estimation (need to create a separate function to encapsulate that)

        # Find how many time periods we have
        T = len(incidence)

        real_dates = incidence.index

        # Generate our pairs of mean and std for the SI by sampling n_si_sims pairs
        si_pairs = self.sample_si_distributions(
            sample_mean_truncnorm, sample_std_truncnorm, n_si_sims)

        results = []
        # Loop through all the pairs
        for pair in si_pairs:

            ## !!! ALL THIS CODE NEEDS TO MOVE TO A FUNCTION
            # Generate the serial interval distribution for the pair
            mean_si, std_si = pair
            si_distribution = self.discr_si(np.arange(0, T), mean_si, std_si)

            # Fill our overflowing distribution with 0s (no prob)
            if len(si_distribution) < (T+1):
                over = -(len(si_distribution) - (T+1))
                si_distribution = np.pad(
                    si_distribution, (0, over), 'constant')

            # Return cumulative incidence per time window, starting window and ending window
            t_start, t_end, nb_time_periods = self.calc_time_windows(
                incidence, win_start=win_start, win_end=win_end)

            # Calculate the parameters of our gamma prior based on the provided mean and std of the serial interval
            a_prior = (mean_prior / std_prior)**2
            b_prior = std_prior**2 / mean_prior

            # Calculate our posteriors from our serial interval distribution
            post = self.posterior_from_si_distrib(
                incidence, si_distribution, a_prior, b_prior, t_start, t_end)
            a_posterior, b_posterior = post[:, 0], post[:, 1]

            # For each time step we sample the posterior distribution
            all_sampled = []
            for t in range(nb_time_periods):
                if np.isnan(a_posterior[t]):
                    sample = [np.nan]*n_posterior_samples
                else:
                    sample = np.random.default_rng().gamma(
                        shape=a_posterior[t], scale=b_posterior[t], size=n_posterior_samples)
                all_sampled.append(sample)

            results.append(all_sampled)
        # Bring all our samples together
        all_samples = np.concatenate(results, axis=1)
        # Move to a dataframe for easier manipulation
        result = self.Rt_sampled(all_samples, t_start, t_end, real_dates)

        # Return the dataframe
        return result

    ## Here the mean and standard deviation of the serial interval distribution varies according to a truncated normal distribution

    ## Mean of our Serial Interval, sampled from Truncated Normal Distribution
    # sample_mean_truncnorm: (mean, std, min, max) of the trunacted normal distribution from which we sample the serial interval mean
    # Equivalent to mean_si (mean) std_mean_si (std), min_mean_si, max_mean_si in epiestim

    ## Standard deviation of our Serial Interval, samples from Truncated Normal Distribution
    # sample_std_truncnorm: (mean, std, min, max) of the of the trunacted normal distribution from which we sample the serial interval standard deviation
    # Equivalent to std_si (mean) std_std_si (std), min_std_si, max_std_si in epiestim

    def sample_si_distributions(self, sample_mean_truncnorm, sample_std_truncnorm, n_samples=100):
        """
        Run estimation of Rt using sampled serial interval distributions
        We sample n_samples pairs of means and standard deviations of serial interval distributions from two truncated normal distributions.

        Parameters
        ----------
        n_samples (int) : number of sampled pairs of mean and std of the SI distribution, needs to be bigger than one
        sample_mean_truncnorm (any number) : (mean, std, min, max) tuple containing parameters of the distribution from which we sample the SI mean 
        param sample_std_truncnorm (any number) : (mean, std, min, max) tuple containing parameters of the distribution from which we sample the SI standard deviation
        
        Returns:
        ----------
        Numpy array containing all the sampled pairs of means and std
        """

        # Unpack our truncated normal to sample SI means
        mean_truncnorm_mean = sample_mean_truncnorm[0] # Mean
        mean_truncnorm_std = sample_mean_truncnorm[1]  # Standard Deviation
        mean_truncnorm_min = sample_mean_truncnorm[2]  # Min (left trunc)
        mean_truncnorm_max = sample_mean_truncnorm[3]  # Max (right trunc)

        # Unpack our truncated normal to sample SI standard deviations
        std_truncnorm_mean = sample_std_truncnorm[0] # Mean
        std_truncnorm_std = sample_std_truncnorm[1]  # Standard Deviation
        std_truncnorm_min = sample_std_truncnorm[2]  # Min (left trunc)
        std_truncnorm_max = sample_std_truncnorm[3]  # Max (right trunc)

        # Make sure our parameters are setup correctly to do the sampling, otherwise return some assertions
        assert n_samples > 0, "Number of samples needs to be bigger than 0"

        # For our Mean sampling distribution
        assert mean_truncnorm_mean > 0, "The mean of the 'Mean SI' truncnorm sampling distribution needs to be >0"
        assert mean_truncnorm_std > 0, "The standard deviation of the 'Mean SI' truncnorm sampling distribution needs to be >0"
        assert mean_truncnorm_min >= 1, "The min of the 'Mean SI' truncnorm sampling distribution needs to be >= 1"
        assert mean_truncnorm_max > mean_truncnorm_mean, "The max of the 'Mean SI' truncnorm sampling distribution needs to be > than the mean"
        assert mean_truncnorm_mean >= mean_truncnorm_min, "The mean of the 'Mean SI' truncnorm sampling distribution needs to be >= the minimum"
        # Issue warning if we the selcted distribution is not centered around the mean
        if (mean_truncnorm_max - mean_truncnorm_mean) != mean_truncnorm_mean - mean_truncnorm_min:
             warnings.warn("Your 'Mean SI' distribution is not centered around the mean")

        # For our Standard Deviation sampling distribution
        assert std_truncnorm_mean > 0, f"The mean ({std_truncnorm_mean}) of the 'Std SI' truncnorm sampling distribution needs to be >0"
        assert std_truncnorm_std > 0, "The standard deviation of the 'Std SI' truncnorm sampling distribution needs to be >0"
        assert std_truncnorm_min >= 1, "The min of the 'Std SI' truncnorm sampling distribution needs to be >= 1"
        assert std_truncnorm_max > std_truncnorm_mean, f"The max ({std_truncnorm_max}) of the 'Std SI' truncnorm sampling distribution needs to be > than the mean({std_truncnorm_mean})"
        assert std_truncnorm_mean >= std_truncnorm_min, "The mean of the 'Std SI' truncnorm sampling distribution needs to be >= the minimum"
        
        # Issue warning if we the selcted distribution is not centered around the mean
        if (std_truncnorm_max - std_truncnorm_mean) != (std_truncnorm_mean - std_truncnorm_min):
            warnings.warn("Your 'Std SI' distribution is not centered around the mean")


        ## This is where the sampling is happenning
        # Calculate the parameters for our truncated distributions
        a_mean = (mean_truncnorm_min - mean_truncnorm_mean) / mean_truncnorm_std
        b_mean = (mean_truncnorm_max - mean_truncnorm_mean) / mean_truncnorm_std
        a_std = (std_truncnorm_min - std_truncnorm_mean) / std_truncnorm_std
        b_std = (std_truncnorm_max - std_truncnorm_mean) / std_truncnorm_std

        # Sample from truncated normal distribution for mean SI
        mean_distrib = stats.truncnorm(a_mean, b_mean, loc=mean_truncnorm_mean, scale=mean_truncnorm_std)
        mean_si_sample = mean_distrib.rvs(n_samples)

        std_distrib = stats.truncnorm(a_std, b_std, loc=std_truncnorm_mean, scale=std_truncnorm_std)
        std_si_sample = std_distrib.rvs(n_samples)

        sampled_si_distributions = np.array([mean_si_sample, std_si_sample]).T
        
        # Return our array of sampled pairs
        return sampled_si_distributions