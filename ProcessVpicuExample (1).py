# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division

import argparse
import itertools
import pandas as pd
import os
import re
import scipy.io as sio

from pandas import Panel, DataFrame, Series

from pydatasets import icu

# <codecell>

_NOTEBOOK = True

# <codecell>

parser = argparse.ArgumentParser()
parser.add_argument('data', type=unicode)
parser.add_argument('out', type=unicode)
parser.add_argument('-vi', '--var_info', type=unicode, default='../../data/vpicu/physiologic-variable-normals.csv')
parser.add_argument('-rr', '--resample_rate', type=int, default=60)
parser.add_argument('-ma', '--max_age', type=int, default=18)
parser.add_argument('-ml', '--max_length_of_stay', type=int, default=30)

if _NOTEBOOK:
    args = parser.parse_args(args=[ '/Users/davekale/mldata/vpicu/dataset-1.0-deid-20130910184316/', 
                                    '/Users/davekale/mldata/vpicu/', '-rr', '60'])
else:
    args = parser.parse_args()

data_dir = args.data
out_dir = args.out
try:
    os.makedirs(out_dir)
except:
    pass

# Get (Low, High) range and Normal value for each variable. For age-dependent variables, this is the "adult female" normal.
ranges = DataFrame.from_csv(args.var_info, parse_dates=False)
ranges = ranges[['Low', 'Normal', 'High']]
varids = ranges.index.tolist()
resample_rate = '{0}min'.format(args.resample_rate)
max_age = args.max_age
max_los = args.max_length_of_stay

# <codecell>

# Read directory contents (should be one entry per patient episode)
eps = os.listdir(data_dir)
N = len(eps) # this is the number of episodes
Xraw = []    # raw time series (without resampling or imputation)
Xmiss = []   # resampled time series (with imputation)
X = []       # resampled, imputed time series
features = np.zeros((N, ranges.shape[0] * 9))  # hand-engineered features
epids = np.zeros((N,), dtype=int)

Traw = np.zeros((N,), dtype=int)    # Number of (irregular) samples for episode
T = np.zeros((N,), dtype=int)       # Resampled episode lengths (for convenience)
age = np.zeros((N,), dtype=int)     # Per-episode patient ages in months
gender = np.zeros((N,), dtype=int8) # Per-episode patient gender
weight = np.zeros((N,))             # Per-episode patient weight
los = np.zeros((N,))                # Length of stay
lmf = np.zeros((N,))                # Last-first duration

y = np.zeros((N,), dtype=int8)      # Outcome (mortality: 1 if died, 0 if survived)
pdiag = np.zeros((N,), dtype=int)   # Primary diagnosis code

perc = 0.01
sys.stdout.write('processing episodes')
idx = 0
count_nodata = 0
for subdir in eps:
    try:
        epid = int(subdir) # VPICU episode is stored in directory named after episode ID
    except:
        print 'Invalid subdirectory {0}, skipping...'.format(subdir)
        continue
    if idx / len(eps) > perc:
        sys.stdout.write('{0}\n'.format(idx))
        perc += 0.01
    try:
        ep = icu.OldVpicuEpisode.from_directory(os.path.join(data_dir, subdir), varids=varids)
    except icu.InvalidIcuDataException as e:
        if e.field == 'msmts' and e.err == 'size': # no measurements data
            count_nodata += 1
        else:
            sys.stdout.write('\nskipping ' + subdir + ': ' + str(e))
        continue

    s = ep.as_nparray()
    mylmf = (s[:,0].max()-s[:,0].min())/60        
    
    # store raw time series
    Xraw.append(s)
    los[idx] = ep.los_hours()
    lmf[idx] = mylmf
    Traw[idx] = s.shape[0]
        
    # extract hand-engineered features
    features[idx,:] = ep.extract_features(normal_values=ranges.Normal).flatten(order='F')
    
    # resample
    s = ep.as_nparray_resampled(rate=resample_rate)
    Xmiss.append(s)
    
    s = ep.as_nparray_resampled(rate=resample_rate, impute=True, normal_values=ranges.Normal)
    if not np.all(~np.isnan(s)):
        print df
        print ep.episodeid
    assert(np.all(~np.isnan(s)))
    X.append(s)
            
    epids[idx] = ep.episodeid
    
    T[idx] = s.shape[0]
    age[idx] = ep.age_in_months()
    gender[idx] = ep.sex
    weight[idx] = ep.weight
    
    y[idx]   = np.nan if ep.died is None else 1 if ep.died else 0
    pdiag[idx]  = ep.prim_diag
    
    idx += 1
sys.stdout.write('DONE!\n')

features = features[0:idx,]
epids = epids[0:idx]
Traw = Traw[0:idx]
T = T[0:idx]
age = age[0:idx]
gender = gender[0:idx]
weight = weight[0:idx]
y = y[0:idx]
pdiag = pdiag[0:idx]
los = los[0:idx]
lmf = lmf[0:idx]

print '{0} episodes had no measurements'.format(count_nodata)

# <codecell>

sio.savemat(os.path.join(out_dir, 'medical-rate{0}.mat'.format(resample_rate)),
            { 'X': X, 'y': y, 'T': T, 'age': age, 'gender': gender, 'weight': weight,
              'pdiag': pdiag, 'ranges': ranges.as_matrix(),
              'features': features, 'Xraw': Xraw, 'Xmiss': Xmiss, 'Traw': Traw,
              'los': los, 'lmf': lmf})
sio.savemat(os.path.join(out_dir, 'medical-rate{0}-epids.mat'.format(resample_rate)), { 'epids': epids })

# <codecell>

np.savez_compressed(os.path.join(out_dir, 'medical-rate{0}.npz'.format(resample_rate)),
                    X = np.array(X), y = y, T = T, age = age,
                    gender = gender, weight = weight,
                    pdiag = pdiag, ranges = ranges.as_matrix(),
                    features = features, Xraw = np.array(Xraw), Xmiss = np.array(Xmiss),
                    Traw = Traw, los = los, lmf = lmf)
np.save(os.path.join(out_dir, 'medical-rate{0}-epids.npy'.format(resample_rate)), epids)

