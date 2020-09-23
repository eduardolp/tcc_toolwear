import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import evaluatenn as nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from keras import backend as K, optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer, r2_score, SCORERS
from matplotlib.offsetbox import AnchoredText

# Read database
#df = pd.read_pickle(r'C:\Users\eduar\OneDrive\UFSC\TCC\data_brought_from_ipt\TCC\clean_scripts\model_input_db_BlackMagic_hardness.pkl') #data with linear regression
df = pd.read_pickle(r'C:\Users\eduar\OneDrive\UFSC\TCC\data_brought_from_ipt\TCC\clean_scripts\model_input_db_BlackMagic_polyreg.pkl') #data with polynomial regression

# add columns for article and experiments numbering
df['experiment_name'] = df.index.get_level_values(0)
df['article_col'] = df['experiment_name'].apply(lambda x: x[3:5])
df['experiment_col'] = df['experiment_name'].apply(lambda x: x[9:])

# creates an object which separates the data from each article. articles_list
# contains the name of each of these objects (article's nuber)
grouped = df.groupby('article_col')

articles_list = df['article_col'].unique()

#grouped.get_group(articles_list[2])

# Function for plotting the data distribution. Plots an histogram of vb_slice
# before and after removal of zeros
def input_data_distr(article):
    """ article is the name of the group inside grouped"""
    path = 'input_data_visualization/poly_reg/vb_bars/' + article +'_PT.pdf'
    
    dataset = grouped.get_group(article)  
    dataset_nozeros = dataset[dataset.vb_slice != 0]
    
    at = AnchoredText('Total antes: %d\nTotal depois: %d\nZeros removidos: %d' %(len(dataset), len(dataset_nozeros), len(dataset)-len(dataset_nozeros)),
                  prop=dict(size=10), frameon=True,
                  loc='upper right',
                  )
    
    fig, ax = plt.subplots(ncols = 2, nrows = 1,figsize = (10,5))
    
    ax[0].hist(dataset.vb_slice, color = '#21deb2')
    ax[0].set_title('Antes de remover os zeros')
    ax[0].set_xlabel('VB_fatia')
    ax[0].set_ylabel('Quantidade')
    ax[0].add_artist(at)
        
    ax[1].hist(dataset_nozeros.vb_slice, color = '#21deb2')
    ax[1].set_title('Após remover os zeros')
    ax[1].set_xlabel('VB_fatia')
#    ax[1].set_ylabel('Count')
    
    fig.suptitle('Distribuição dos VB_fatia - Artigo %d' %int(article))
    #    fig.legend(['Train', 'Validation'], loc='upper right')
    # fig.savefig(path)
#    plt.show()

for i in articles_list:
    input_data_distr(i)

# function for dealing with data preparation
def prepare_data(dataframe, plot=False):
    df = dataframe
    dataframe_number = dataframe.loc[:, 'article_col'].unique()
    no_zeros = df[df.vb_slice != 0]
    # Extract the relevant columns and then divide them into a features and a targets 
    # dataframes. 
    reduced_df = no_zeros.loc[:,['engagements','hardness','bsp','lsp','vc','vb_slice']]
    features = reduced_df.loc[:,['engagements','hardness','bsp','lsp','vc']]
    target = reduced_df.loc[:,'vb_slice']
    
    if plot==True:
        nn.plot_inputs(features, target, dataframe_number[0], save=False, pdf=True)
    
    # 2-set split
    X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                    test_size=0.2, random_state=42)
    
    # Standardize the features by subtracting the mean and dividing by the standard
    # deviation. The scaler is calculated for the training sample and then applied
    # to the test data. In the end the targets were used without standardization.
    X_scaler = preprocessing.StandardScaler().fit(X_train)
    y_scaler = preprocessing.StandardScaler().fit(np.array(y_train).reshape(-1,1))
    
    X_train_scaled = X_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(np.array(y_test).reshape(-1, 1))
    #X_validate_scaled = X_scaler.transform(X_validate)
    #y_validate_scaled = y_scaler.transform(np.array(y_validate).reshape(-1, 1))
    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(np.array(y_test).reshape(-1, 1))
    
    return X_train_scaled, y_train, X_test_scaled, y_test, dataframe_number

for i in articles_list:
    prepare_data(grouped.get_group(i), plot=False)

###############################################################################
# Define the model - same for all articles
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
# hidden_layer3 = Dense(5, activation = ac_hid_, 
#                       kernel_regularizer = regularizers.l1_l2(0,0))
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
# dropout_layer = Dropout(0)

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

def train_model(dataframe):
    X_train_scaled, y_train, X_test_scaled, y_test, dataframe_number = prepare_data(dataframe)
    model = KerasRegressor(build_fn=create_model, epochs=150, batch_size=100, verbose = False)
    # history = model.fit(X_train_scaled, y_train, validation_data = (X_test_scaled, y_test), verbose = True)#, callbacks = [reduce_lr])
    # cross_val = cross_val_score(model, X_train_scaled, y_train, cv=5, verbose=10, scoring=make_scorer(r2_score))
    cross_val = cross_val_score(model, X_train_scaled, y_train, cv=5, verbose=10, scoring='neg_mean_absolute_error')
    #cross_val = cross_validate(model, X_train_scaled, y_train, cv=5, verbose=10, scoring=make_scorer(r2_score))
    
    # Save training results
    # with open('article_networks_nozeros/poly_reg/article%s_history_mae.pkl' %dataframe_number, 'wb') as handle:
    #     pk.dump(history.history, handle, protocol=pk.HIGHEST_PROTOCOL)
    with open('article_networks_nozeros/poly_reg/article%s_crossval_mae.pkl' %dataframe_number, 'wb') as handle:
        pk.dump(cross_val, handle, protocol=pk.HIGHEST_PROTOCOL)
    # model.model.save(r'article_networks_nozeros/poly_reg/article%s_model_mae.h5' %dataframe_number)
    
    return model, cross_val
    # return model, history, cross_val

# Call training for all articles
for i in articles_list:
    train_model(grouped.get_group(i))
# modelo, mse = train_model(grouped.get_group('04'))
    
