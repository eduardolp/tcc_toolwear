import pandas as pd
import numpy as np
import pickle as pk
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import evaluatenn as nn
# import random as random
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# from keras import backend as K, optimizers
# from keras import regularizers
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.metrics import make_scorer, r2_score
from matplotlib.offsetbox import AnchoredText
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Configure backend
mpl.use('pdf')

# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=10)

##############################################################################
# BEGIN DATA TREATMENT
##############################################################################
# Read database
#df = pd.read_pickle(r'C:\Users\eduar\OneDrive\UFSC\TCC\data_brought_from_ipt\TCC\clean_scripts\model_input_db_BlackMagic_hardness.pkl') #data with linear regression
df = pd.read_pickle(r'C:\Users\eduar\OneDrive\UFSC\TCC\data_brought_from_ipt\TCC\clean_scripts\model_input_db_BlackMagic_polyreg.pkl') #data with polynomial regression

# add columns for article and experiments numbering
df['experiment_name'] = df.index.get_level_values(0)
df['article_col'] = df['experiment_name'].apply(lambda x: x[3:5])
df['experiment_col'] = df['experiment_name'].apply(lambda x: x[9:])

# creates an object which separates the data from each article. articles_list
# contains the name of each of these objects (article's number)
grouped = df.groupby('article_col')
articles_list = df['article_col'].unique()

# removes all data composed of only 0 (not really representative of the process)
no_zeros = df[df.vb_slice!=0]

# sampling strategy requires classes. 5 were used
bins=[0,0.05,0.1,0.15,0.2,max(no_zeros.vb_slice)]
no_zeros['vb_sliceRANGE'] , interval= pd.cut(no_zeros.vb_slice, bins=bins,labels=['a','b','c','d','e'], retbins=True)


features_list = ['engagements','hardness','bsp','lsp','vc']
targets_list = ['vb_slice']
features = no_zeros.loc[:, features_list+targets_list]
target = no_zeros.loc[:, 'vb_sliceRANGE']

# separate data into training and (validation + testing) datasets in a 70/30 (20/10) proportion
X_train, X_partial, y_train, y_partial = train_test_split(features, target, 
                                                    test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_partial, y_partial, 
                                                    test_size=0.33, random_state=42)

# Undersample the training data
# plt.hist(no_zeros.vb_sliceRANGE, color='#21deb2')
ros = RandomOverSampler(sampling_strategy='not majority',random_state=12)
rus = RandomUnderSampler(sampling_strategy='not minority', random_state=12)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)  

# Reobtain the correct training, validation and testing datasets
X_train_reduced_ros = X_train_ros.loc[:, features_list]
y_train_reduced_ros = X_train_ros.loc[:, targets_list] #Sim, X_train_res está correto

X_train_reduced_rus = X_train_rus.loc[:, features_list]
y_train_reduced_rus = X_train_rus.loc[:, targets_list] #Sim, X_train_res está correto




labels=['(0;0,05]', '(0,05;0,1]', '(0,1;0,15]', '(0,15;0,2]', '(0,2;Max]']

X_train['vb_sliceRANGE'] = pd.cut(X_train.vb_slice, bins=bins, labels=labels)
X_train_ros['vb_sliceRANGE'] = pd.cut(X_train_ros.vb_slice, bins=bins, labels=labels)
X_train_rus['vb_sliceRANGE'] = pd.cut(X_train_rus.vb_slice, bins=bins, labels=labels)

counts_train = Counter(X_train.vb_sliceRANGE)
counts_ros = Counter(X_train_ros.vb_sliceRANGE)
counts_rus = Counter(X_train_rus.vb_sliceRANGE)

# sorts the dictionary keys
counts_train = {k: v for k, v in sorted(counts_train.items(), key=lambda item: item[1], reverse=True)}
counts_ros = {k: v for k, v in sorted(counts_ros.items(), key=lambda item: item[1], reverse=True)}
counts_rus = {k: v for k, v in sorted(counts_rus.items(), key=lambda item: item[1], reverse=True)}


fig, ax = plt.subplots(constrained_layout=False, figsize=(10,8))

# sets the x positions
x = np.arange(len(counts_train))
bar_width = 0.28

ax.bar(x - bar_width, list(counts_train.values()), bar_width,
       align='center', 
       label='Dados originais', 
       color='#76ECD2')

ax.bar(x + bar_width, list(counts_ros.values()), bar_width,
       align='center',
       label='Oversample',
       color='#00D6A6')
ax.bar(x, list(counts_rus.values()), bar_width, 
       align='center',
       label='Undersample',
       color='#00A07C')

plt.xticks(x, list(counts_train.keys()))

plt.legend(loc='upper right')

plt.xlabel('Desgaste [mm]')
plt.ylabel('Ocorrências')

width = 6.2959
height = width/1.618 #/1.2# 1.618

fig.set_size_inches(width, height)
fig.savefig('plots/distr_vbslices_balancing_strat.pdf', format='pdf')
