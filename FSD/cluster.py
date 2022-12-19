import os
from glob import glob

import numpy as np


def create_tabular_data(fsd_dir_name, num_frequencies):
  df = []
  npz_filenames = sorted(glob(os.path.join(fsd_dir_name) + '*.npz'))
  for npz_filename in npz_filenames:
    npzfile = np.load(npz_filename)
    X = npzfile['X']
    Y = npzfile['Y']
    df_object = []
    df_object.append(os.path.basename(npz_filename))
    for l in range(-num_frequencies // 2, num_frequencies // 2):
      df_object.append(np.abs(X[l] + 1j * Y[l]))
    df.append(df_object)
  return df
