import pickle as pk
import evaluatenn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import np_utils
from keras import backend as K
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, roc_auc_score

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
# no_zeros5 = df[df.vb_slice!=0]

# sampling strategy requires classes. 5 were used
bins3 = [0,0.1,0.2,1]
bins5 = [0,0.05,0.1,0.15,0.2,1]
names3 = ('baixo', 'médio', 'alto')
names5 = ('muito baixo', 'baixo', 'médio', 'atenção', 'alto')
no_zeros['vb_sliceRANGE3'] , interval= pd.cut(no_zeros.vb_slice, bins=bins3, labels=names3, retbins=True)
no_zeros['vb_sliceRANGE5'] , interval= pd.cut(no_zeros.vb_slice, bins=bins5, labels=names5, retbins=True)


features_list = ['engagements','hardness','bsp','lsp','vc']
targets_list = ['vb_slice']
features = no_zeros.loc[:, features_list+targets_list]
target3 = no_zeros.loc[:, 'vb_sliceRANGE3']
target5 = no_zeros.loc[:, 'vb_sliceRANGE5']

# encode class values as integers
encoder3 = preprocessing.LabelEncoder()
encoder3.fit(target3)
encoded_Y3 = encoder3.transform(target3)
encoder5 = preprocessing.LabelEncoder()
encoder5.fit(target5)
encoded_Y5 = encoder5.transform(target5)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y3 = np_utils.to_categorical(encoded_Y3)
dummy_y5 = np_utils.to_categorical(encoded_Y5)

# separate data into training and (validation + testing) datasets in a 70/30 (20/10) proportion
X_train3, X_partial3, y_train3, y_partial3 = train_test_split(features, dummy_y3, 
                                                    test_size=0.3, random_state=42)
X_val3, X_test3, y_val3, y_test3 = train_test_split(X_partial3, y_partial3, 
                                                    test_size=0.33, random_state=42)

X_train5, X_partial5, y_train5, y_partial5 = train_test_split(features, dummy_y5, 
                                                    test_size=0.3, random_state=42)
X_val5, X_test5, y_val5, y_test5 = train_test_split(X_partial5, y_partial5, 
                                                    test_size=0.33, random_state=42)
# Undersample the training data
# plt.hist(no_zeros.vb_sliceRANGE, color='#21deb2')
# ros = RandomOverSampler(sampling_strategy='not majority',random_state=12)
# rus = RandomUnderSampler(sampling_strategy='majority', random_state=12)
# X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
# X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)  

# Reobtain the correct training, validation and testing datasets
X_train_reduced3 = X_train3.loc[:, features_list]
y_train_reduced3 = y_train3 
X_train_reduced5 = X_train5.loc[:, features_list]
y_train_reduced5 = y_train5 
# X_train_reduced_ros = X_train_ros.loc[:, features_list]
# y_train_reduced_ros = X_train_ros.loc[:, targets_list] #Sim, X_train_res está correto
# X_train_reduced_rus = X_train_rus.loc[:, features_list]
# y_train_reduced_rus = X_train_rus.loc[:, targets_list] #Sim, X_train_res está correto

X_val_reduced3 = X_val3.loc[:, features_list]
y_val_reduced3 = y_val3
X_val_reduced5 = X_val5.loc[:, features_list]
y_val_reduced5 = y_val5


X_test_reduced3 = X_test3.loc[:, features_list]
y_test_reduced3 = y_test3
X_test_reduced5 = X_test5.loc[:, features_list]
y_test_reduced5 = y_test5


# Scaler creation and preparation of the scaled datasets
X_scaler3 = preprocessing.StandardScaler().fit(X_train_reduced3)
X_train_scaled3 = X_scaler3.transform(X_train_reduced3)
X_val_scaled3 = X_scaler3.transform(X_val_reduced3)
X_test_scaled3 = X_scaler3.transform(X_test_reduced3)

X_scaler5 = preprocessing.StandardScaler().fit(X_train_reduced5)
X_train_scaled5 = X_scaler5.transform(X_train_reduced5)
X_val_scaled5 = X_scaler5.transform(X_val_reduced5)
X_test_scaled5 = X_scaler5.transform(X_test_reduced5)

# X_scaler_ros = preprocessing.StandardScaler().fit(X_train_reduced_ros)
# X_train_scaled_ros = X_scaler_ros.transform(X_train_reduced_ros)
# X_val_scaled_ros = X_scaler_ros.transform(X_val_reduced)
# X_test_scaled_ros = X_scaler_ros.transform(X_test_reduced)

# X_scaler_rus = preprocessing.StandardScaler().fit(X_train_reduced_rus)
# X_train_scaled_rus = X_scaler_rus.transform(X_train_reduced_rus)
# X_val_scaled_rus = X_scaler_rus.transform(X_val_reduced)
# X_test_scaled_rus = X_scaler_rus.transform(X_test_reduced)

# y_scaler = preprocessing.StandardScaler().fit(np.array(y_train_reduced).reshape(-1,1))
# y_train_scaled = y_scaler.transform(y_train_reduced)
# y_val_scaled = y_scaler.transform(y_val_reduced)
# y_test_scaled = y_scaler.transform(y_test_reduced)
##############################################################################
# END DATA TREATMENT
##############################################################################


## Base code for reading pre-trained classifier model

# loads the model's architecture and weights
model3 = load_model(r"results\naive_class3_model.h5")
model5 = load_model(r"results\naive_class5_model.h5")

# reads the training history. Useful for plotting the loss evolution during training      
with open(r"results\naive_class3_history.pkl", 'rb') as handle:
    history3 = pk.load(handle)
with open(r"results\naive_class5_history.pkl", 'rb') as handle:
    history5 = pk.load(handle)
    
# reads the cross validation output. Vector with the loss of each cross validation iteration     
with open(r"results\naive_class3_crossval.pkl", 'rb') as handle:
    crossval3 = pk.load(handle)
with open(r"results\naive_class5_crossval.pkl", 'rb') as handle:
    crossval5 = pk.load(handle)

names = ['model']    
list_models3 = [model3]
list_history3 = [history3]
list_crossval3 = [crossval3]

list_models5 = [model5]
list_history5 = [history5]
list_crossval5 = [crossval5]

#for i in range(len(list_history)):
#    path = 'article_networks_leveled\poly_reg_2x\history_plots\loss_metric' + names[i] + '.pdf'
#    nn.plot_training_individual_articles(list_history[i], names[i], path)
    
crossval_avg3 = [np.mean(x) for x in list_crossval3]

print("Acurácia crossval 3 classes: ", crossval3)
print("Acurácia média 3 classes: ", crossval_avg3)
print("Desvio Padrão 5 classes: ", np.std(crossval3))

crossval_avg5 = [np.mean(x) for x in list_crossval5]

print("Acurácia crossval 5 classes: ", crossval5)
print("Acurácia média 5 classes: ", crossval_avg5)
print("Desvio Padrão 5 classes: ", np.std(crossval5))

###################################################################################
# Plot history
fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(10,8))

ax[0].plot(history3['loss'][1:], color = '#179c7d', linestyle='--')
ax[0].plot(history3['val_loss'][1:], color = '#179c7d', alpha=0.5)
ax[0].plot(history5['loss'][1:], color = '#21deb2', linestyle='--')
ax[0].plot(history5['val_loss'][1:], color = '#21deb2', alpha=0.5)
ax[0].set_title('Perda')
ax[0].set_xlabel('Época')

ax[1].plot(history3['acc'], color = '#179c7d', linestyle='--', label='Treinamento 3 classes')
ax[1].plot(history3['val_acc'], color = '#179c7d', label='Validação 3 classes', alpha=0.5)
ax[1].plot(history5['acc'], color = '#21deb2', linestyle='--', label='Treinamento 5 classes')
ax[1].plot(history5['val_acc'], color = '#21deb2', label='Validação 5 classes', alpha=0.5)
ax[1].set_title('$Acurácia$')
ax[1].set_xlabel('Época')

plt.legend()
width = 6.2959
height = width/1.618 #/1.2# 1.618

fig.set_size_inches(width, height)
fig.savefig('plots/history_evolution_35classes_naive.pdf', format='pdf')

###################################################################################
# Test data evaluation
y_pred3 = model3.predict(X_test_scaled3)
acc3 = accuracy_score(encoder3.inverse_transform(y_test_reduced3.argmax(1)), encoder3.inverse_transform(y_pred3.argmax(1)))
print("Acurácia teste 3 classes: ", acc3) 

y_pred5 = model5.predict(X_test_scaled5)
acc5 = accuracy_score(encoder5.inverse_transform(y_test_reduced5.argmax(1)), encoder5.inverse_transform(y_pred5.argmax(1)))
print("Acurácia teste 5 classes: ", acc5)

report3 = classification_report(encoder3.inverse_transform(y_test_reduced3.argmax(1)), encoder3.inverse_transform(y_pred3.argmax(1)), labels=names3)
confusion3 = confusion_matrix(encoder3.inverse_transform(y_test_reduced3.argmax(1)), encoder3.inverse_transform(y_pred3.argmax(1)), labels=names3, normalize='true')

report5 = classification_report(encoder5.inverse_transform(y_test_reduced5.argmax(1)), encoder5.inverse_transform(y_pred5.argmax(1)), labels=names5)
confusion5 = confusion_matrix(encoder5.inverse_transform(y_test_reduced5.argmax(1)), encoder5.inverse_transform(y_pred5.argmax(1)), labels=names5, normalize='true')

