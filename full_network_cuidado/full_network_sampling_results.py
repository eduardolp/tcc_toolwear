import pickle as pk
import evaluatenn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer, r2_score

# Configure backend and plotting
mpl.use('pdf')

# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=10)
plt.rc('legend', fontsize=8)

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
bins=[0,0.05,0.1,0.15,0.2,2]
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
rus = RandomUnderSampler(sampling_strategy='majority', random_state=12)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)  

# Reobtain the correct training, validation and testing datasets
X_train_reduced_ros = X_train_ros.loc[:, features_list]
y_train_reduced_ros = X_train_ros.loc[:, targets_list] #Sim, X_train_res está correto
X_train_reduced_rus = X_train_rus.loc[:, features_list]
y_train_reduced_rus = X_train_rus.loc[:, targets_list] #Sim, X_train_res está correto

X_val_reduced = X_val.loc[:, features_list]
y_val_reduced = X_val.loc[:, targets_list]

X_test_reduced = X_test.loc[:, features_list]
y_test_reduced = X_test.loc[:, targets_list]

# Scaler creation and preparation of the scaled datasets
X_scaler_ros = preprocessing.StandardScaler().fit(X_train_reduced_ros)
X_train_scaled_ros = X_scaler_ros.transform(X_train_reduced_ros)
X_val_scaled_ros = X_scaler_ros.transform(X_val_reduced)
X_test_scaled_ros = X_scaler_ros.transform(X_test_reduced)

X_scaler_rus = preprocessing.StandardScaler().fit(X_train_reduced_rus)
X_train_scaled_rus = X_scaler_rus.transform(X_train_reduced_rus)
X_val_scaled_rus = X_scaler_rus.transform(X_val_reduced)
X_test_scaled_rus = X_scaler_rus.transform(X_test_reduced)

# y_scaler = preprocessing.StandardScaler().fit(np.array(y_train_reduced).reshape(-1,1))
# y_train_scaled = y_scaler.transform(y_train_reduced)
# y_val_scaled = y_scaler.transform(y_val_reduced)
# y_test_scaled = y_scaler.transform(y_test_reduced)
##############################################################################
# END DATA TREATMENT
##############################################################################




## Base code for reading pre-trained model
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# loads the model's architecture and weights
model_ros = load_model(r"results\ros_notmaj_model.h5", custom_objects={'r2_keras':r2_keras})
model_rus = load_model(r"results\rus_notmin_model.h5", custom_objects={'r2_keras':r2_keras})

# reads the training history. Useful for plotting the loss evolution during training      
with open(r"results\ros_notmaj_history.pkl", 'rb') as handle:
    history_ros = pk.load(handle)
with open(r"results\rus_notmin_history.pkl", 'rb') as handle:
    history_rus = pk.load(handle)
    
# reads the cross validation output. Vector with the loss of each cross validation iteration     
with open(r"results\ros_notmaj_crossval.pkl", 'rb') as handle:
    crossval_ros = pk.load(handle)
with open(r"results\rus_notmin_crossval.pkl", 'rb') as handle:
    crossval_rus = pk.load(handle)

names = ['model']    
list_models_ros = [model_ros]
list_history_ros = [history_ros]
list_crossval_ros = [crossval_ros]

list_models_rus = [model_rus]
list_history_rus = [history_rus]
list_crossval_rus = [crossval_rus]

#for i in range(len(list_history)):
#    path = 'article_networks_leveled\poly_reg_2x\history_plots\loss_metric' + names[i] + '.pdf'
#    nn.plot_training_individual_articles(list_history[i], names[i], path)
    
crossval_avg_ros = [np.mean(x) for x in list_crossval_ros]

print("R2 crossval ros: ", crossval_ros)
print("R2 medio ros: ", crossval_avg_ros)
print("Desvio Padrão ros: ", np.std(crossval_ros))

crossval_avg_rus = [np.mean(x) for x in list_crossval_rus]

print("R2 crossval rus: ", crossval_rus)
print("R2 medio rus: ", crossval_avg_rus)
print("Desvio Padrão rus: ", np.std(crossval_rus))

###################################################################################
# Plot history
fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10,8))

ax[0].plot(history_ros['loss'][1:], color = '#179c7d', linestyle='--')
ax[0].plot(history_ros['val_loss'][1:], color = '#179c7d')
ax[0].plot(history_rus['loss'][1:], color = '#21deb2', linestyle='--')
ax[0].plot(history_rus['val_loss'][1:], color = '#21deb2')
ax[0].set_ylabel('Perda')
ax[0].set_xlabel('Época')

ax[1].plot(history_ros['r2_keras'], color = '#179c7d', linestyle='--', label='Treinamento com oversample')
ax[1].plot(history_ros['val_r2_keras'], color = '#179c7d', label='Validação com oversample')
ax[1].plot(history_rus['r2_keras'], color = '#21deb2', linestyle='--', label='Treinamento com undersample')
ax[1].plot(history_rus['val_r2_keras'], color = '#21deb2', label='Validação com undersample')
ax[1].set_ylabel('$R^2$')
ax[1].set_xlabel('Época')

plt.legend()
width = 6.2959
height = width/1.618 #/1.2# 1.618

fig.set_size_inches(width, height)
# fig.savefig('plots/history_evolution_balanced_correcao.pdf', format='pdf')

###################################################################################
# Test data evaluation
y_pred_ros = model_ros.predict(X_test_scaled_ros)
r2_score_ros = r2_score(y_test_reduced, y_pred_ros)
print("ros R2: ", r2_score_ros) 

y_pred_rus = model_rus.predict(X_test_scaled_rus)
r2_score_rus = r2_score(y_test_reduced, y_pred_rus)
print("rus R2: ", r2_score_rus)