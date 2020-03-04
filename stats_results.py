import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

log_dir = 'runs'
model_name = 'model_geometricAttention_best_122_fusion5'
data_dir = log_dir + '/' + model_name + '/' + 'error_data.pickle'
data = pickle.load(open(data_dir, 'rb'))
data_frame = pd.DataFrame.from_dict(data)

data_frame.sort_values(by=['min_fde'], ascending=False, inplace=True)

print(data_frame['min_fde'].median())
print(data_frame['min_fde'].mean())

print(data_frame[['idx', 'min_fde']][data_frame['min_fde']>10])
print(len(data_frame['idx'][data_frame['min_fde']>10]))

ax = data_frame['min_fde'].plot.hist(bins=100, alpha=0.5)

plt.show()