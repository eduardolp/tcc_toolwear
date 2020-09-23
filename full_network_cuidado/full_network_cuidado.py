import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import evaluatenn as nn
import random as random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from keras import backend as K, optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer, r2_score
from matplotlib.offsetbox import AnchoredText
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

rand_state = random.randint(1, 100) 

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

# Creates categories based on vb_slice 
def categorizer(dataframe):
    '''Creates vb_sliceRANGE category in a dataframe and returns the size of 
    each category in a list along with the new dataframe'''
    # categorizer
    # text = 'ABCDEFGHIJ'
    labels = [0,1,2,3,4,5,6,7,8,9]#list(text)
    dataframe['vb_sliceRANGE'] = pd.cut(dataframe.vb_slice, 10, labels=labels)
    # counter
    cat_group = dataframe.groupby('vb_sliceRANGE') #the name of each group is given by the category label
    group_size = []
    for i in labels:
        group_size.append(len(cat_group.get_group(i).index))
        
    return  dataframe, group_size

def dataframe_leveler(dataframe, rand_state=rand_state):
    '''Returns a balanced dataframe based on vb_sliceRANGE along with the amount
    of data in each category after balancing'''
    df, group_size = categorizer(dataframe)
    df = df.sample(frac=1, random_state=rand_state).groupby('vb_sliceRANGE').head(min(group_size))
    
    text = 'ABCDEFGHIJ'
    labels = list(text)
    cat_group = df.groupby('vb_sliceRANGE') #the name of each group is given by the category label
    group_size = []
    for i in labels:
        group_size.append(len(cat_group.get_group(i).index))
    
    return df, group_size

no_zeros_cat, no_zeros_cat_counts = categorizer(no_zeros)
# no_zeros_lvl, no_zeros_lvl_cat_counts = dataframe_leveler(no_zeros)


features_list = ['engagements','hardness','bsp','lsp','vc']
targets_list = ['vb_slice']
features = no_zeros_cat#[no_zeros_cat!='vb_slice']
target = no_zeros_cat.loc[:, 'vb_sliceRANGE']

# separate data into training and (validation + testing) datasets in a 70/30 (20/10) proportion
X_train, X_partial, y_train, y_partial = train_test_split(features, target, 
                                                    test_size=0.3, random_state=rand_state)
X_val, X_test, y_val, y_test = train_test_split(X_partial, y_partial, 
                                                    test_size=0.33, random_state=rand_state)

# Oversample the training data
ros = RandomOverSampler(sampling_strategy='minority',random_state=12)
rus = RandomUnderSampler(random_state=12)
# X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)  

# Reobtain the correct training, validation and testing datasets
X_train_reduced = X_train_res.loc[:, features_list]
y_train_reduced = X_train_res.loc[:, targets_list] #Sim, X_train_res está correto

X_val_reduced = X_val.loc[:, features_list]
y_val_reduced = X_val.loc[:, targets_list]

X_test_reduced = X_test.loc[:, features_list]
y_test_reduced = X_test.loc[:, targets_list]

# Samples no_zeros giving it the same number of values for all vb_slice ranges
# no_zeros is shuffled and the number of values in each range is given by the 
# range with the least amount of data 
# X_train_lvl_partial, cat_count = dataframe_leveler(X_train)
# X_train_lvl = X_train_lvl_partial.loc[:, features_list]
# y_train_lvl = dataframe_leveler(y_train)[0].loc[:, 'vb_slice']
   

# leveled data histogram plot
# fig, ax = plt.subplots()
# ax.hist(X_train_lvl.vb_slice, bins=10, color = '#21deb2', density=True)
# ax.hist(no_zeros.vb_slice, bins=10, color = '#127A62', density=True,histtype='step')


# Scaler creation and preparation of the scaled datasets
X_scaler = preprocessing.StandardScaler().fit(X_train_reduced)
X_train_scaled = X_scaler.transform(X_train_reduced)
X_val_scaled = X_scaler.transform(X_val_reduced)
X_test_scaled = X_scaler.transform(X_test_reduced)

# y_scaler = preprocessing.StandardScaler().fit(np.array(y_train_reduced).reshape(-1,1))
# y_train_scaled = y_scaler.transform(y_train_reduced)
# y_val_scaled = y_scaler.transform(y_val_reduced)
# y_test_scaled = y_scaler.transform(y_test_reduced)
##############################################################################
# END DATA TREATMENT
##############################################################################
##############################################################################
##############################################################################
# BEGIN MODEL
###############################################################################
# Define a r2 metric for use during the training of the model
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Values for each variable for generating a grid for GridSearch
regularizations1 = [0]
regularizations2 = [0]
learning_rates = [0.001]

activations_in = ['selu']#, 'tanh', 'sigmoid', 'relu', 'hard_sigmoid']
activations_hid = ['elu']#, 'tanh', 'sigmoid', 'relu',  'hard_sigmoid']
activations_out = ['relu']#, 'linear']

# Initial variables 
reg1_ = 0
reg2_ = 0
lrs_ = 0.1

ac_in_ = 'selu'
ac_hid_ = 'elu'
ac_out_ = 'relu'

# Define the layer which will be used and give them a name. This is useful for
# accessing the weights and biases.
input_layer = Dense(5, input_dim=5, activation = ac_in_, 
                    kernel_regularizer = regularizers.l1_l2(0,0))
#hidden_layer1 = Dense(3, activation = ac_hid_, 
#                      kernel_regularizer = regularizers.l1_l2(0,0))
hidden_layer2 = Dense(10, activation = ac_hid_, 
                      kernel_regularizer = regularizers.l1_l2(0,0))
#hidden_layer3 = Dense(5, activation = ac_hid_, 
#                      kernel_regularizer = regularizers.l1_l2(0,0))
hidden_layer4 = Dense(100, activation = ac_hid_, 
                      kernel_regularizer = regularizers.l1_l2(0,0))
hidden_layer5 = Dense(100, activation = ac_hid_, 
                      kernel_regularizer = regularizers.l1_l2(0,0))
hidden_layer6 = Dense(100, activation = ac_hid_, 
                      kernel_regularizer = regularizers.l1_l2(0,0))
hidden_layer7 = Dense(10, activation = ac_hid_, 
                      kernel_regularizer = regularizers.l1_l2(0,0))
output_layer = Dense(1, activation = ac_out_, 
                      kernel_regularizer = regularizers.l1_l2(0,0))
dropout_layer = Dropout(0)

attempt = [input_layer, 
              hidden_layer2,
              hidden_layer4,
              hidden_layer5,
              hidden_layer6,
              hidden_layer7,
              output_layer]

def create_model(lrs=lrs_, ac_in=ac_in_, ac_hid=ac_hid_, ac_out=ac_out_, 
                  reg1=reg1_, reg2=reg2_):
    model = Sequential()
    
    for i in attempt:
        model.add(i)

    #adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, 
    #                       decay=0.0, amsgrad=False)
    sgd = optimizers.SGD(lr=lrs, momentum=0, decay=0.0, nesterov=False) 
            
    model.compile(optimizer=sgd, loss='mse', metrics=[r2_keras])
    return model


model = KerasRegressor(build_fn=create_model, epochs=150, batch_size=100, verbose = False)
history = model.fit(X_train_scaled, y_train_reduced, validation_data = (X_val_scaled, y_val_reduced), verbose = True)#, callbacks = [reduce_lr])
# cross_val = cross_val_score(model, X_train_scaled, y_train_reduced, cv=5, verbose=10, scoring=make_scorer(r2_score))

# # Save training results
# with open('poly_reg_2xlvl/history.pkl' , 'wb') as handle:
#     pk.dump(history.history, handle, protocol=pk.HIGHEST_PROTOCOL)
# with open('poly_reg_2xlvl/crossval.pkl', 'wb') as handle:
#     pk.dump(cross_val, handle, protocol=pk.HIGHEST_PROTOCOL)
# model.model.save(r'poly_reg_2xlvl/model.h5')

y_pred = model.predict(X_val_scaled)
r2_score = r2_score(y_val_reduced, y_pred)
print("R2: ", r2_score) 
##############################################################################
# END MODEL
###############################################################################

nn.plot_training(history.history)
nn.plot_predictions(X_val_reduced, y_val_reduced, y_pred, "nome")
