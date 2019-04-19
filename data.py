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

def plot_fft(signals, ax=None, title='', logscale=False, force_equal_sizes=True):
    if not isinstance(signals, dict): signals = {'signal': signals}
    n = min([len(s) for s in signals.values()]) if force_equal_sizes else None
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(ax, str) and ax == 'interactive':
        ax = InteractiveFigure().get_axes()
    if title: title += '\n'
    if len(signals) == 1:
        title += f'({len(list(signals.values())[0]):d} samples)'
    else:
        lengths = ', '.join([f'{len(signals[sig][:n]):d}' for sig in signals])
        title += f'({lengths:s} samples respectively)'

    for i, sig in enumerate(signals):
        signal = signals[sig][:n]
        f = np.fft.fft(signal, norm='ortho')[1:]
        if logscale:
            f = np.log10(1 + np.abs(f)) * np.sign(f)
        ax.plot(1+np.arange(len(f)), f, color=utils.DEF_COLORS[i],
                linewidth=0.8, linestyle='-', label=sig)

    ax.grid()
    ax.set_xlabel('Frequency [1/dt]', fontsize=12)
    ax.set_ylabel('Log-FFT [base 10]' if logscale else 'FFT', fontsize=12)
    ax.set_title(title, fontsize=14)
    if len(signals) > 1:
        ax.legend()
    utils.draw()

def enrich_fixed_time(df, drop_open_row=True, show=False):
    df['fixed_time'] = 0
    dt = np.median(np.diff(-df.quaketime))
    transitions = np.where(np.logical_or(
        np.diff(-df.quaketime)>10*dt, np.diff(-df.quaketime)<-10*dt) )[0]
    transitions = np.concatenate(([-1], transitions, [len(df)-1]))
    # shift times
    for ti,tf in zip(transitions[:-1],transitions[1:]):
        last = df.quaketime[tf]
        df.loc[ti+1:tf,'fixed_time'] = last + 250e-9*np.arange(tf-ti,0,-1)
    # remove first entry of each block
    if drop_open_row:
        for ti in transitions[1:-1][::-1]:
            df.drop(ti+1, axis=0, inplace=True)
    if show:
        ax = InteractiveFigure().get_axes()
        ax.plot(df.quaketime, color='blue', label='original')
        ax.plot(df.fixed_time, color='green', label='fixed (250 nano)')
        ax.set_title('Time correction', fontsize=14)
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Time to quake', fontsize=12)
        ax.legend()
        utils.draw()
    return df

def complete_time_grid(df, enriched=False, show=False):
    if not enriched:
        df = enrich_fixed_time(df, show=show)
    dt = np.median(np.diff(-df.fixed_time))
    df['ind'] = -np.round(df.fixed_time/ dt)
    d_all = pd.DataFrame(
        data={'ind':list(range(int(df['ind'].values[0]),
                                 int(df['ind'].values[-1]+1)))}
    )
    d_all = d_all.merge(df, on='ind', how='outer')
    if show:
        ax = InteractiveFigure().get_axes()
        ax.plot(d_all.signal)
        print(d_all)
        print(np.mean(np.isnan(d_all.signal)))
        nans_starts = np.where(np.logical_and(np.logical_not(np.isnan(d_all.signal.values[:-1])),np.isnan(d_all.signal.values[1:])))[0]
        nans_ends = np.where(np.logical_and(np.logical_not(np.isnan(d_all.signal.values[1:])),np.isnan(d_all.signal.values[:-1])))[0]
        print(utils.dist(nans_ends-nans_starts))
    return d_all

def averaged_spectrum(df):
    # find transitions
    dt = np.median(np.diff(-df.quaketime))
    transitions = np.where(np.logical_or(
        np.diff(-df.quaketime)>10*dt, np.diff(-df.quaketime)<-10*dt) )[0] + 1
    transitions = np.concatenate(([0], transitions, [len(df)]))
    print('Distribution of blocks lengths:')
    print(utils.dist([tf-ti for ti,tf in zip(transitions[:-1],transitions[1:])]))
    print(f'Valid blocks (4096 samples):\t' +
          f'{np.sum(np.diff(transitions)==4096):.0f}/{len(transitions):d}')
    # calculate ffts
    ffts = []
    for ti,tf in zip(transitions[:-1],transitions[1:]):
        if tf-ti != 4096: continue
        f = np.fft.fft(df.signal[ti:tf], norm='ortho')[1:]
        ffts.append(np.abs(f))
    # plot averaged fft
    avg_fft = np.mean(ffts, axis=0)
    fig,axs = plt.subplots(3,1)
    axs[0].plot(avg_fft, linewidth=0.8)
    axs[0].set_ylabel('Averaged Fourier Over Blocks')
    axs[1].plot(ffts[3], linewidth=0.8)
    axs[1].set_ylabel('FFT: Block 3')
    axs[2].plot(ffts[30], linewidth=0.8)
    axs[2].set_ylabel('FFT: Block 30')
    # plot distribution of [max(abs(f)) for f in ffts]
    fig,axs = plt.subplots(2,1)
    axs[0].plot(utils.dist([np.max(f) for f in ffts],np.arange(1001)/10)[2:])
    axs[0].set_xlabel('Quantile [%]')
    axs[0].set_ylabel('Max Fourier Amplitude in Block')
    axs[1].plot(utils.dist([np.argmax(f) for f in ffts],np.arange(1001)/10)[2:])
    axs[1].set_xlabel('Quantile [%]')
    axs[1].set_ylabel('Frequency of Max Fourier Amplitude')
    utils.draw()


if __name__ == '__main__':
    # configuration
    demo = True
    signals = False
    distributions = False
    FFTs = True
    time_interpolation = False

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
    if FFTs:
        plot_fft({'long':df.signal,'short':df.signal.head(int(len(df)/10))}, ax='interactive',
                 title='FFT of long vs. short (prefix of the long) signals', force_equal_sizes=False)
        plot_fft({'train':df.signal,'test':dt.signal[:int(150e3)]}, ax='interactive',
                 title='FFT of train vs. test signals')
        averaged_spectrum(df[:int(150e3)])
        print(f'FFTs plotted ({time()-t0:.0f} [s])')

    # time correction
    if time_interpolation:
        df_tmp = load_data(nrows=100000)
        complete_time_grid(df_tmp, show=True)

    plt.show()
