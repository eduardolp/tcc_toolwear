import pickle as pk
import evaluatenn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K
from sklearn import preprocessing

## Base code for reading pre-trained model
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# #LOAD MODELS TRAINED WITHOUT ZEROS
# # loads the model's architecture and weights
# model04_clean = load_model(r"article_networks_nozeros\poly_reg\article['04']_model.h5", custom_objects={'r2_keras':r2_keras})
# model06_clean = load_model(r"article_networks_nozeros\poly_reg\article['06']_model.h5", custom_objects={'r2_keras':r2_keras})
# model07_clean = load_model(r"article_networks_nozeros\poly_reg\article['07']_model.h5", custom_objects={'r2_keras':r2_keras})
# model08_clean = load_model(r"article_networks_nozeros\poly_reg\article['08']_model.h5", custom_objects={'r2_keras':r2_keras})
# model14_clean = load_model(r"article_networks_nozeros\poly_reg\article['14']_model.h5", custom_objects={'r2_keras':r2_keras})

# # reads the training history. Useful for plotting the loss evolution during training      
# with open(r"article_networks_nozeros\poly_reg\article['04']_history.pkl", 'rb') as handle:
#     history04_clean = pk.load(handle)
# with open(r"article_networks_nozeros\poly_reg\article['06']_history.pkl", 'rb') as handle:
#     history06_clean = pk.load(handle)
# with open(r"article_networks_nozeros\poly_reg\article['07']_history.pkl", 'rb') as handle:
#     history07_clean = pk.load(handle)
# with open(r"article_networks_nozeros\poly_reg\article['08']_history.pkl", 'rb') as handle:
#     history08_clean = pk.load(handle)
# with open(r"article_networks_nozeros\poly_reg\article['14']_history.pkl", 'rb') as handle:
#     history14_clean = pk.load(handle)
    
# reads the training history. Useful for plotting the loss evolution during training      
with open(r"article_networks_nozeros\poly_reg\article['04']_crossval_mae.pkl", 'rb') as handle:
    crossval04_clean = pk.load(handle)
with open(r"article_networks_nozeros\poly_reg\article['06']_crossval_mae.pkl", 'rb') as handle:
    crossval06_clean = pk.load(handle)
with open(r"article_networks_nozeros\poly_reg\article['07']_crossval_mae.pkl", 'rb') as handle:
    crossval07_clean = pk.load(handle)
with open(r"article_networks_nozeros\poly_reg\article['08']_crossval_mae.pkl", 'rb') as handle:
    crossval08_clean = pk.load(handle)
with open(r"article_networks_nozeros\poly_reg\article['14']_crossval_mae.pkl", 'rb') as handle:
    crossval14_clean = pk.load(handle)

# #LOAD MODELS TRAINED WITH ZEROS
# # loads the model's architecture and weights
# model04 = load_model(r"article_networks\article['04']_model.h5", custom_objects={'r2_keras':r2_keras})
# model06 = load_model(r"article_networks\article['06']_model.h5", custom_objects={'r2_keras':r2_keras})
# model07 = load_model(r"article_networks\article['07']_model.h5", custom_objects={'r2_keras':r2_keras})
# model08 = load_model(r"article_networks\article['08']_model.h5", custom_objects={'r2_keras':r2_keras})
# model14 = load_model(r"article_networks\article['14']_model.h5", custom_objects={'r2_keras':r2_keras})

# # reads the training history. Useful for plotting the loss evolution during training      
# with open(r"article_networks\article['04']_history.pkl", 'rb') as handle:
#     history04 = pk.load(handle)
# with open(r"article_networks\article['06']_history.pkl", 'rb') as handle:
#     history06 = pk.load(handle)
# with open(r"article_networks\article['07']_history.pkl", 'rb') as handle:
#     history07 = pk.load(handle)
# with open(r"article_networks\article['08']_history.pkl", 'rb') as handle:
#     history08 = pk.load(handle)
# with open(r"article_networks\article['14']_history.pkl", 'rb') as handle:
#     history14 = pk.load(handle)
    
# # reads the training history. Useful for plotting the loss evolution during training      
# with open(r"article_networks\article['04']_crossval.pkl", 'rb') as handle:
#     crossval04 = pk.load(handle)
# with open(r"article_networks\article['06']_crossval.pkl", 'rb') as handle:
#     crossval06 = pk.load(handle)
# with open(r"article_networks\article['07']_crossval.pkl", 'rb') as handle:
#     crossval07 = pk.load(handle)
# with open(r"article_networks\article['08']_crossval.pkl", 'rb') as handle:
#     crossval08 = pk.load(handle)
# with open(r"article_networks\article['14']_crossval.pkl", 'rb') as handle:
#     crossval14 = pk.load(handle)


names = ['04', '06', '07', '08', '14']    

# list_models_clean = [model04_clean, model06_clean, model07_clean, model08_clean, model14_clean]
# list_history_clean = [history04_clean, history06_clean, history07_clean, history08_clean, history14_clean]
list_crossval_clean = [crossval04_clean, crossval06_clean, crossval07_clean, crossval08_clean, crossval14_clean]

# list_models = [model04, model06, model07, model08, model14]
# list_history = [history04, history06, history07, history08, history14]
# list_crossval = [crossval04, crossval06, crossval07, crossval08, crossval14]

#for i in range(len(list_history)):
#    path = 'article_networks_nozeros\poly_reg\history_plots\loss_metric' + names[i] + '.pdf'
#    nn.plot_training_individual_articles(list_history[i], names[i], path)

crossval_avg_clean = [np.mean(x) for x in list_crossval_clean]    
# crossval_avg = [np.mean(x) for x in list_crossval]

# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# ax[0].bar(names, crossval_avg_clean, color = '#21deb2')
# ax[1].bar(names, crossval_avg, color = '#00A07C')
# ax.set_ylabel("$R^2$")
# ax.set_xlabel("Artigos")
# fig.suptitle("$R^2$ para os modelos preliminares")
# fig.savefig("resultado_crossval_comparacao.pdf")