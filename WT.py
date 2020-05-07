import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import gamma
import warnings

class Model_WT: 
   
    def draw_one_set_of_ancestries(self, onset, incidence, possible_ances_time, si_distrib, t_start, t_end):
        res = np.array([])
        possible_ances_time = np.array(possible_ances_time)

        for t in range(t_start[0], t_end[-1]+1):
            if len(onset[onset == t]) > 0:
                if(len(possible_ances_time[t]) > 0):
                    prob = si_distrib[(t+1) - np.asarray(possible_ances_time[t])] * \
                                       incidence[np.asarray(
                                           possible_ances_time[t])-1]
                    if(any(prob > 0)):
                        ot = np.where(onset == t+1)
                        print(ot)
                        if(len(res) == 0):
                            nans = [np.nan]*ot[0][0]
                            res = np.append(res, nans)
                        ## Draw from multinomial
                        a = np.random.multinomial(
                            1, self.normalize(prob), size=(len(ot[0]))).T
                        b = np.where(a == 1)[0]
                        c = np.take(possible_ances_time[t], b)
                        res = np.append(res, c)
                    else:
                        res = np.append(res, np.nan)
                else:
                    res = np.append(res, np.nan)
                    ## NEED TO COMPLETE
                    #res[which(Onset == t)] <- NA
        return res

        ### Draw from discretized gamma distribution
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
        b = sigma**2 / (mu - 1)  # scale

        res = k * gamma.cdf(k, a, scale=b) + (k - 2) * gamma.cdf(k - 2,
                                                                 a, scale=b) - 2 * (k-1) * gamma.cdf(k-1, a, scale=b)
        res = res + a * b * (2 * gamma.cdf(k - 1, a + 1, scale=b) -
                             gamma.cdf(k - 2, a + 1, scale=b) - gamma.cdf(k, a + 1, scale=b))

        res2 = []
        for i in range(len(res)):
            m = max(0, res[i])
            res2.append(m)

        # Checked for consistency in results with EpiEstim
        return np.array(res2)

    def normalize(self, vector):
        new_vector = []
        for i in vector:
            a = i/np.sum(vector)
            new_vector.append(a)
        return new_vector

        ### Define total incidence per time window
    def calc_incidence_per_time_step(self, incidence, win_start=2, win_end=8, silent=True):
        # adjust the windows to work with python array index
        win_start -= 1
        win_end -= 1

        ## Define time steps as a moving window of 1 week
        total_time = len(incidence)

        t_start = np.arange(win_start, total_time-(win_end-win_start))
        t_end = np.arange(win_end, total_time)

        nb_time_periods = len(t_start)
        if silent == False:
            print("Calculated time periods: ", nb_time_periods)

        cumsum = []
        for i in range(nb_time_periods):
            j = np.sum(incidence[t_start[i]:t_end[i]])
            cumsum.append(j)

        # Checked for consistency in results with EpiEstim
        return cumsum, t_start, t_end, nb_time_periods

    def estimate_R_t_WT(self, incidence, mean_si, std_si, n_sims=1, win_start=2, win_end=8, mean_prior=4, std_prior=4):
        T = len(incidence)
        incidence = incidence.tolist()
        si_distribution = self.discr_si(np.arange(0, T), mean_si, std_si)

        # Fill our overflowing distribution with 0s (no prob)
        if len(si_distribution) < (T+1):
            over = -(len(si_distribution) - (T+1))
            si_distribution = np.pad(si_distribution, (0, over), 'constant')


#for index, item in enumerate(items):
        onset = np.array([])
        for i, el in enumerate(incidence):
            repl = el * (i+1)
            onset = np.append(onset, repl)

        num_cases = len(onset)

        vec = np.arange(0, T)
        delay = np.subtract.outer(vec, vec)

        si_delay = []
        for column in delay.T:
           test = np.minimum(np.maximum(column, 0), len(si_distribution))
           test = np.flip(si_distribution[test])
           si_delay.insert(0, test)

        si_delay = np.array(si_delay)

        sum_on_col_si_delay_tmp = []
        for i in range(0, len(si_delay)):
            applied = np.sum(si_delay[i] * incidence)
            sum_on_col_si_delay_tmp.append(applied)

        sum_on_col_si_delay = []

        for i, el in enumerate(sum_on_col_si_delay_tmp):
            sum_delay = el * incidence[i]
            sum_on_col_si_delay.append(sum_delay)

        #sum_on_col_si_delay = [
            #val for sublist in sum_on_col_si_delay for val in sublist]

        mat_sum_on_col_si_delay = []

        ### CHECK THIS IS REALLY CORRECT
        for j in sum_on_col_si_delay_tmp:
            row = [j]*T
            mat_sum_on_col_si_delay.append(row)

        mat_sum_on_col_si_delay = np.array(mat_sum_on_col_si_delay)
        p = si_delay / mat_sum_on_col_si_delay
        p = np.where(np.isnan(p), 0, p)
        p = np.where(np.isinf(p), 0,p)                          

        mean_r_per_index_case_date = []
        for column in p.T:
            mean_r = np.sum(column*incidence)
            mean_r_per_index_case_date.append(mean_r)

        mean_r_per_index_case_date = np.array(mean_r_per_index_case_date)

        cum_inc, t_start, t_end, nb_time_periods = self.calc_incidence_per_time_step(incidence, win_start=win_start, win_end=win_end)
        nb_time_periods = len(t_start)

        incidence = np.array(incidence)

        mean_r_per_date_wt = []
        for i in range(0, nb_time_periods):
            select = (np.arange(0, T) >= t_start[i]) * (np.arange(0, T) <= t_end[i]) == 1
            a = mean_r_per_index_case_date[select]
            b = incidence[select].astype(int)
            c = np.repeat(a, b)
            mean = np.mean(c)
            mean_r_per_date_wt.append(mean)

        ## SIMULATIONS FOR UNCERTAINY
        possible_ances_time = []
        for t in range(0, T):
            loc = np.asarray(np.where(si_distribution != 0))
            a = (t - (loc) + 1)
            loc2 = (t - np.asarray(np.where(si_distribution != 0)) + 1)
            b = a[loc2 > 0]
            possible_ances_time.append(b.tolist())

        ## NEED TO DO ANCESTRIES_TIME

        ancestries_time = []
        for i in range(0, n_sims):
            ancestry = self.draw_one_set_of_ancestries(
                onset, incidence, possible_ances_time, si_distribution, t_start, t_end)
            ancestries_time.append(ancestry)
        ancestries_time = np.array(ancestries_time)

        r_sim = []
        for period in range(0, nb_time_periods):
            simulations = []
            for row in ancestries_time:
                #print(incidence[t_start[period]:t_end[period]+1])
                #row = row[~np.isnan(row)]
                a = np.count_nonzero(row[(row >= t_start[period]) & (
                    row <= t_end[period])]) / np.sum(incidence[t_start[period]:t_end[period]+1])
                simulations.append(a)
            r_sim.append(simulations)

        df = pd.DataFrame(r_sim)


        df['r025_wt'] = df.quantile(q=0.025, axis=1, interpolation='midpoint')
        df['r0975_wt'] = df.quantile(q=0.975, axis=1, interpolation='midpoint')
        df['rt_mean_wt'] = mean_r_per_date_wt

        # print(t_start)
        # print(mean_r_per_date_wt)
        # print(r_sim)
        results = pd.DataFrame(
            {'Rt_Mean': df['rt_mean_wt'], 'r025_wt': df['r025_wt'], 'r0975_wt': df['r0975_wt']})

        return results
