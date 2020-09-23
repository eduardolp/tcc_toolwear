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

# loads the model's architecture and weights
model = load_model(r"poly_reg_2xlvl\model.h5", custom_objects={'r2_keras':r2_keras})

# reads the training history. Useful for plotting the loss evolution during training      
with open(r"poly_reg_2xlvl\history.pkl", 'rb') as handle:
    history = pk.load(handle)
    
# reads the cross validation output. Vector with the loss of each cross validation iteration     
with open(r"poly_reg_2xlvl\crossval.pkl", 'rb') as handle:
    crossval = pk.load(handle)


names = ['model']    
list_models = [model]
list_history = [history]
list_crossval = [crossval]

#for i in range(len(list_history)):
#    path = 'article_networks_leveled\poly_reg_2x\history_plots\loss_metric' + names[i] + '.pdf'
#    nn.plot_training_individual_articles(list_history[i], names[i], path)
    
crossval_avg = [np.mean(x) for x in list_crossval]

print("R2 crossval: ", crossval)
print("R2 medio: ", crossval_avg)

fig, ax = plt.subplots()
ax.bar(names, crossval_avg, color = '#21deb2')
ax.set_ylabel("R^2")
ax.set_xlabel("Artigos")
fig.suptitle("R^2 para os modelos preliminares")
fig.savefig("poly_reg_2xlvl/resultado_crossval.pdf")