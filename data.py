import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import re
from time import time
from pprint import pprint
import io
from warnings import warn
import utils

def load_data(fname=r'Data/train.csv', nrows=None):
    df = pd.read_csv(fname, nrows=nrows)
    df.rename(columns={'acoustic_data': 'signal',
                       'time_to_failure': 'quaketime'},
              inplace=True)
    return df

def plot_data(df, cols=None, ax=None):
    # initialization
    if ax is None:
        fig, ax = plt.subplots()
    if cols is None:
        cols = df.columns[:2]
    n = df.shape[0]
    double_y = (not isinstance(cols,str)) and len(cols)>1
    title = 'Points: '
    title += f'{n:d}' if n<1e3 \
        else f'{n/1e3:.1f}K' if n<1e6 else f'{n/1e6:.1f}M'


    # first axis
    col = cols[0] if double_y else cols
    ax.plot(np.arange(len(df[col])), df[col], 'b-')
    ax.set_title('Points: ', fontsize=14)
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel(col, color='b')
    ax.tick_params('y', colors='b')

    # second axis
    if double_y:
        ax2 = ax.twinx()
        col = cols[1]
        ax2.plot(np.arange(len(df[col])), df[col], 'r-')
        ax2.set_ylabel(col, color='r')
        ax2.tick_params('y', colors='r')

    utils.draw()


if __name__ == '__main__':
    df = load_data(nrows=10e6)
    plot_data(df)
    plt.show()


# TODO:

# plot train data in interesting resolutions
# plot test data as a long sequence and mark the segment transitions
# plot train & test signal distributions
#    both separately (using twinx?) and as QQ-plot

# compare fft plot of train segment with various methods:
#    as raw; various interpolations; separated and then averaged
# plot fft of the whole train data with/out chosen method
# plot distribution of the strongest frequency
#    over 4K blocks / 150K segments?
# compare train & test ffts

# try to detect test segments which look continuous
#    (according to Zahar's heuristic)?
