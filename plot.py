import matplotlib.dates as mdt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import gamma

class rt_plot:
    def plot_base(self, Rt, figsize=(7, 5), ci=0.05, title=None, use_dates=False, marker_size=1.0):

        to_plot = Rt.to_dataframe(ci)
        over_1 = to_plot[to_plot['Rt_mean'] >= 1]
        under_1 = to_plot[to_plot['Rt_mean'] < 1]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(x=over_1['real_dates'], y=over_1['Rt_mean'],
                   color=sns.color_palette()[1], s=marker_size)
        date_form = mdt.DateFormatter("%d-%m")
        date_loc = mdt.WeekdayLocator(byweekday=MO)
        date_loc = mdt.AutoDateLocator()
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(date_loc)
        ax.scatter(x=under_1['real_dates'], y=under_1['Rt_mean'],
                   color=sns.color_palette()[0], s=marker_size)

        ax.fill_between(to_plot['real_dates'], np.where(to_plot['high_ci'] < 1, to_plot['high_ci'], 1), np.where(
            to_plot['low_ci'] < 1, to_plot['low_ci'], 1), alpha=0.25, color=sns.color_palette()[0])
        ax.fill_between(to_plot['real_dates'], np.where(to_plot['high_ci'] > 1, to_plot['high_ci'], 1), np.where(
            to_plot['low_ci'] > 1, to_plot['low_ci'], 1), alpha=0.25, color=sns.color_palette()[1])

        #ax.set_xlim(xmin=Rt.real_dates[1], xmax=Rt.real_dates[-1])
        ax.axhline(y=1, color='black')
        ax.set_ylabel('Estimated Rt')
        ax.set_xlabel('Date Index')

        ax.set_ylim(ymin=0, ymax=np.max(to_plot['high_ci'].max() + 1))
        if title:
            fig.suptitle(title)

        return fig, ax

    def plot_sample(self, to_plot, figsize=(7,5), title=None, use_dates=False, marker_size=1.0):
        over_1 = to_plot[to_plot['Rt_mean'] >= 1]
        under_1 = to_plot[to_plot['Rt_mean'] < 1]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        ax.scatter(x=over_1['real_dates'], y=over_1['Rt_mean'],
                   color=sns.color_palette()[1], s=marker_size)
        date_form = mdt.DateFormatter("%d-%m")
        date_loc = mdt.WeekdayLocator(byweekday=MO)
        date_loc = mdt.AutoDateLocator()
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(date_loc)
        ax.scatter(x=under_1['real_dates'], y=under_1['Rt_mean'],
                   color=sns.color_palette()[0], s=marker_size)

        ax.fill_between(to_plot['real_dates'], np.where(to_plot['high_ci'] < 1, to_plot['high_ci'], 1), np.where(
            to_plot['low_ci'] < 1, to_plot['low_ci'], 1), alpha=0.25, color=sns.color_palette()[0])
        ax.fill_between(to_plot['real_dates'], np.where(to_plot['high_ci'] > 1, to_plot['high_ci'], 1), np.where(
            to_plot['low_ci'] > 1, to_plot['low_ci'], 1), alpha=0.25, color=sns.color_palette()[1])

        #ax.set_xlim(xmin=Rt.real_dates[1], xmax=Rt.real_dates[-1])
        ax.axhline(y=1, color='black')
        ax.set_ylabel('Estimated Rt')
        ax.set_xlabel('Date Index')

        ax.set_ylim(ymin=0, ymax=np.max(to_plot['high_ci'].max() + 1))
        if title:
            fig.suptitle(title)

        return fig, ax

