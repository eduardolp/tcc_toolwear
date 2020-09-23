import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

# Configure backend
mpl.use('pdf')

# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=10)


# Read database
#df = pd.read_pickle(r'C:\Users\eduar\OneDrive\UFSC\TCC\data_brought_from_ipt\TCC\clean_scripts\model_input_db_BlackMagic_hardness.pkl') #data with linear regression
df = pd.read_pickle(r'C:\Users\eduar\OneDrive\UFSC\TCC\data_brought_from_ipt\TCC\clean_scripts\model_input_db_BlackMagic_polyreg.pkl') #data with polynomial regression

# add columns for article and experiments numbering
df['experiment_name'] = df.index.get_level_values(0)
df['article_col'] = df['experiment_name'].apply(lambda x: x[3:5])
df['experiment_col'] = df['experiment_name'].apply(lambda x: x[9:])


no_zeros = df[df.vb_slice != 0]
no_zeros_grouped = no_zeros.groupby('article_col')

names = ['04', '06', '07', '08', '14']

fig, ax = plt.subplots(nrows=3, ncols=2, constrained_layout=False, figsize=(10,8))
fig.subplots_adjust(left=0.2, right=0.95, bottom=.1, top=.9, hspace=0.4, wspace=0.3)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Desgaste [mm]')
fig.text(0.08, 0.5, 'Ocorrências', va='center', rotation='vertical')
temp = np.ravel(ax)
for i,j in enumerate(names):
    temp[i].hist(no_zeros_grouped.get_group(j).vb_slice,  color = '#21deb2')
    temp[i].set_title('Artigo '+ j)
ax[2][1].set_visible(False)

width = 6.2959
height = width#/1.618 #/1.2# 1.618

fig.set_size_inches(width, height)
fig.savefig('distr_vbslices.pdf', format='pdf')