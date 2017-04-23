"""
===========================
Generate simulated raw data
===========================

This example generates raw data by repeating a desired source
activation multiple times.

"""
# Authors: Sheraz Khan <me@skhan.me>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import read_source_spaces, find_events, Epochs, compute_covariance
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
src_fname = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
bem_fname = (data_path +
             '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')
subjects_dir = data_path + '/subjects'


##############################################################################
# Generate dipole time series
n_dipoles = 64  # number of dipoles to create
epoch_duration = 2.  # duration of each epoch/event
sfreq = 1000



times = np.arange(0.,epoch_duration,1./sfreq)
src = read_source_spaces(src_fname)



vertices = [src[0]['vertno'], src[1]['vertno']]
stc1 = simulate_sparse_stc(src, n_dipoles=64, times=times)
stc2 = simulate_sparse_stc(src, n_dipoles=64, times=times)
stc3 = simulate_sparse_stc(src, n_dipoles=64, times=times)

stc1.expand(vertices)
stc2.expand(vertices)
stc3.expand(vertices)

stc4 = stc1.copy()

data = np.hstack((stc1.data, stc2.data, stc3.data))

stc4._data = np.hstack((data,data,data,data,data))
stc4._update_times()

stc5 = stc4.mean()

brain = stc5.plot(subject='sample', time_viewer=True,hemi='split', colormap='gnuplot',
                           views=['lateral','medial'],
                 surface='inflated', subjects_dir=subjects_dir)


