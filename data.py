import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
from time import time
from pprint import pprint
import os
from warnings import warn
import utils
from InteractivePlotter import InteractiveFigure

def load_data(fname=r'Data/train.csv', nrows=None, skiprows=None):
    df = pd.read_csv(fname, nrows=nrows, skiprows=skiprows)
    df.rename(columns={'acoustic_data': 'signal',
                       'time_to_failure': 'quaketime'},
              inplace=True)
    return df

def load_multi_files(dir=r'Data/test/', n_files=10, i0=0):
    fnames = [dir + '/' + nm for nm in os.listdir(dir)][i0:i0+n_files]
    dfs = []
    for i,nm in enumerate(fnames):
        df = load_data(nm)
        df['segment'] = i0+i
        dfs.append(df)
    return pd.concat(dfs)

def plot_data(df, cols=None, split_col=None, title='', ax=None):
    # initialization
    mpl.rcParams['agg.path.chunksize'] = int(1e7)
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(ax,str) and ax == 'interactive':
        ax = InteractiveFigure().get_axes()
    if cols is None:
        cols = df.columns[:2]
    n = df.shape[0]
    double_y = (not isinstance(cols,str)) and len(cols)>1
    n_samples = utils.counter_to_str(n)
    tit = title+'\n' if title else title
    tit += f'({n_samples:s} samples)'
    shape = '-' if n<=150e3 else ','

    # first axis
    col = cols[0] if double_y else cols
    ax.plot(np.arange(len(df[col])), df[col], 'b'+shape)
    ax.grid()
    ax.set_title(tit, fontsize=14)
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel(col, color='b')
    ax.tick_params('y', colors='b')

    # vertical splitting lines
    if split_col:
        M = np.max(np.abs(df[col]))
        ids = np.where(np.diff(df[split_col])!=0)
        for i in ids:
            ax.plot((i+1,i+1), (-M,M), 'k-')

    # second axis
    if double_y:
        ax2 = ax.twinx()
        col = cols[1]
        ax2.plot(np.arange(len(df[col])), df[col], 'r'+shape)
        ax2.set_ylabel(col, color='r')
        ax2.tick_params('y', colors='r')

    utils.draw()

def plot_distributions(signals, ax=None, quantiles=1000, title='', logscale=False):
    if not isinstance(signals, dict): signals = {'signal':signals}
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(ax,str) and ax == 'interactive':
        ax = InteractiveFigure().get_axes()
    if title: title+='\n'
    if len(signals) == 1:
        title += f'({len(list(signals.values())[0]):d} samples)'
    else:
        lengths = ', '.join([f'{len(signals[sig]):d}' for sig in signals])
        title += f'({lengths:s} samples respectively)'

    for i,sig in enumerate(signals):
        signal = np.log10(1+np.abs(signals[sig])) * np.sign(signals[sig]) \
            if logscale else signals[sig]
        dist = utils.dist(signal, 100*np.arange(quantiles+1)/quantiles)
        ax.plot((0,100), 2*[dist[1]], color=utils.DEF_COLORS[i], linestyle=':')
        ax.plot(100*np.arange(quantiles+1)/quantiles, dist[2:],
                color=utils.DEF_COLORS[i], linestyle='-', label=sig)

    ax.set_xlim((0,100))
    ax.grid()
    ax.set_xlabel('Quantile [%]', fontsize=12)
    ax.set_ylabel('Log-signal [base 10]' if logscale else 'Signal', fontsize=12)
    ax.set_title(title, fontsize=14)
    if len(signals) > 1:
        ax.legend()
    utils.draw()


if __name__ == '__main__':
    # configuration
    demo = True
    signals = False
    distributions = False

    # initialization
    if demo:
        N, n_files, file0 = 1e6, 10, int(np.random.uniform(0,1000))
    else:
        N, n_files, file0 = None, 10000, 0
    t0 = time()

    # load data
    df = load_data(nrows=N)
    print(f'Train data loaded ({time()-t0:.0f} [s])')
    dt = load_multi_files(n_files=n_files, i0=file0)
    print(f'Test data loaded ({time()-t0:.0f} [s])')

    # plot signals
    if signals:
        plot_data(df.head(int(10e6)), ax='interactive', title='Beginning of train')
        plot_data(df.iloc[[1000*i for i in range(int(min(1e5,len(df)/1e3)))],:],
                  ax='interactive', title='Beginning of train with skips (1:1000)')
        i0 = int(np.random.uniform(0,10e6))
        plot_data(df[i0:i0+int(150e3)], ax='interactive',
                  title='Random segment in train')
        plot_data(dt, cols='signal', split_col='segment', ax='interactive',
                  title='Random adjacent segments in test')
        print(f'Signals plotted ({time()-t0:.0f} [s])')

    # plot distributions
    if distributions:
        plot_distributions({'train':df.signal,'test':dt.signal}, ax='interactive',
                           title='Sample of train vs. test signals')
        plot_distributions({'train':df.signal,'test':dt.signal}, ax='interactive',
                           title='Sample of train vs. test signals (log)', logscale=True)
        utils.qqplot(df.signal, dt.signal, ('train','test'), ax='interactive',
               title='QQ-plot: train vs. test')
        utils.qqplot(df.signal, dt.signal, ('train','test'), ax='interactive',
               title='QQ-plot: train vs. test (log)', logscale=True)
        print(f'Distributions plotted ({time()-t0:.0f} [s])')

    # plot FFTs
    # TODO

    plt.show()


# TODO:

# compare fft plot of train segment with various methods:
#    as raw; various interpolations; separated and then averaged
# plot fft of the whole train data with/out chosen method
# plot distribution of the strongest frequency
#    over 4K blocks / 150K segments?
# compare train & test ffts
