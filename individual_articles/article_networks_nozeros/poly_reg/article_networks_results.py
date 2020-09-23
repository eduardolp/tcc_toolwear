import pickle as pk
import evaluatenn as nn
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K
from sklearn import preprocessing

# Configure backend
mpl.use('pdf')

# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=10)

## Base code for reading pre-trained model
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# loads the model's architecture and weights
model04 = load_model(r"article['04']_model.h5", custom_objects={'r2_keras':r2_keras})
model06 = load_model(r"article['06']_model.h5", custom_objects={'r2_keras':r2_keras})
model07 = load_model(r"article['07']_model.h5", custom_objects={'r2_keras':r2_keras})
model08 = load_model(r"article['08']_model.h5", custom_objects={'r2_keras':r2_keras})
model14 = load_model(r"article['14']_model.h5", custom_objects={'r2_keras':r2_keras})

# reads the training history. Useful for plotting the loss evolution during training      
with open(r"article['04']_history.pkl", 'rb') as handle:
    history04 = pk.load(handle)
with open(r"article['06']_history.pkl", 'rb') as handle:
    history06 = pk.load(handle)
with open(r"article['07']_history.pkl", 'rb') as handle:
    history07 = pk.load(handle)
with open(r"article['08']_history.pkl", 'rb') as handle:
    history08 = pk.load(handle)
with open(r"article['14']_history.pkl", 'rb') as handle:
    history14 = pk.load(handle)
    
# reads the training history. Useful for plotting the loss evolution during training      
with open(r"article['04']_crossval.pkl", 'rb') as handle:
    crossval04 = pk.load(handle)
with open(r"article['06']_crossval.pkl", 'rb') as handle:
    crossval06 = pk.load(handle)
with open(r"article['07']_crossval.pkl", 'rb') as handle:
    crossval07 = pk.load(handle)
with open(r"article['08']_crossval.pkl", 'rb') as handle:
    crossval08 = pk.load(handle)
with open(r"article['14']_crossval.pkl", 'rb') as handle:
    crossval14 = pk.load(handle)

names = ['04', '06', '07', '08', '14']    
list_models = [model04, model06, model07, model08, model14]
list_history = [history04, history06, history07, history08, history14]
list_crossval = [crossval04, crossval06, crossval07, crossval08, crossval14]

#for i in range(len(list_history)):
#    path = 'history_plots\loss_metric' + names[i] + '.pdf'
#    nn.plot_training_individual_articles(list_history[i], names[i], path)
    
crossval_avg = [np.mean(x) for x in list_crossval]

fig, ax = plt.subplots()
ax.bar(names, crossval_avg, color = '#21deb2')
ax.set_ylabel("$R^2$")
ax.set_xlabel("Artigos")
fig.suptitle("$R^2$ para os modelos preliminares")

width = 6.2959
height = width/1.618 #/1.2# 1.618

fig.set_size_inches(width, height)

fig.savefig("resultado_crossval2.pdf", format='pdf')