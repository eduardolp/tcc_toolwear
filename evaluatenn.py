import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

def plot_inputs(features, targets, name, save=False, pdf=False, png=False):
    '''
    Plots all features by the target. Show the distribution of the input data.
    If save is set to True, saves a copy of the plot.
    '''
    fig = plt.figure(figsize=(5,5))
    G = gridspec.GridSpec(3, 2)
    
    fig.suptitle(name, fontsize=16)
    
    axes1 = plt.subplot(G[2, :])
    axes2 = plt.subplot(G[0, 1])
    axes3 = plt.subplot(G[1, 0])
    axes4 = plt.subplot(G[1, 1])
    axes5 = plt.subplot(G[0, 0])
    
    axes1.plot(features.engagements, targets, 'x', color = 'g', alpha = 0.1)
    axes2.plot(features.lsp, targets, 'x', color = 'g', alpha = 0.1)
    axes3.plot(features.bsp, targets, 'x', color = 'g', alpha = 0.1)
    axes4.plot(features.vc, targets, 'x', color = 'g', alpha = 0.1)
    axes5.plot(features.hardness, targets, 'x', color = 'g', alpha = 0.1)
    
    axes1.set_title('Engagements')
    axes2.set_title('lsp')
    axes3.set_title('bsp')
    axes4.set_title('cutting velocity')
    axes5.set_title('hardness')
    
    if save==True:
        article = name #int(name[2:4])
#        if pdf==True and png==True:
#            print('Please choose a single format to save the plots.')
        if pdf==True:
            path = 'input_data_visualization/poly_reg/features_distr' + article + '.pdf'
        elif png==True:
            path = 'input_data_visualization/poly_reg/features_distr' + article + '.png'
        fig.savefig(path)
    
    plt.show()
    
def test_model(unscaled_features, model, X_scaler):
    '''
    Used for testing specific conditions of the data. The input variable is a 
    1x5 array with unscaled features. 
    '''
    return model.predict(X_scaler.transform(unscaled_features).reshape(1, -1), verbose = True)  
    
def plot_training(history):
    '''
    Plots the loss and the R2 evolution during training. Returns training
    and validation values
    '''
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10,5))
    
    ax[0].plot(history['loss'][1:], color = '#21deb2')
    ax[0].plot(history['val_loss'][1:], color = '#179c7d')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    
    ax[1].plot(history['r2_keras'], color = '#21deb2')
    ax[1].plot(history['val_r2_keras'], color = '#179c7d')
    ax[1].set_title('R2')
    ax[1].set_xlabel('Epoch')
    fig.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    
def plot_training_individual_articles(history, name, path):
    '''
    Plots the loss and the R2 evolution during training. Returns training
    and validation values
    '''
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10,5))
    
    ax[0].plot(history['loss'][1:], color = '#21deb2')
    ax[0].plot(history['val_loss'][1:], color = '#179c7d')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    
    ax[1].plot(history['r2_keras'], color = '#21deb2')
    ax[1].plot(history['val_r2_keras'], color = '#179c7d')
    ax[1].set_title('R2')
    ax[1].set_xlabel('Epoch')
    fig.suptitle(name)
    fig.legend(['Train', 'Validation'], loc='upper right')
    fig.savefig(path)
    plt.show()    
    
def plot_weights(model):
    layer_list = []
    for layer in model.layers:
        layer_list.append(layer)
    
    a = int(np.sqrt(len(layer_list)))+1
    m = 0
    fig, ax = plt.subplots(ncols = a, nrows = a, figsize = (20,20))
    for row in range(a):
        for col in range(a):
            if len(layer_list) > m:
                im = ax[row][col].imshow(layer_list[m].get_weights()[0])
                ax[row][col].set_title(str(layer_list[m].name))
                #ax[row][col].set_xlabel('Current Layer Nodes')
                #ax[row][col].set_ylabel('Previous Layer Nodes')
                fig.colorbar(im, ax=ax[row][col])
            m += 1
            
def plot_weights_hist(model):
    layer_list = []
    for layer in model.layers:
        layer_list.append(layer)
        
    a = int(np.sqrt(len(layer_list)))+1
    m = 0
    fig, ax = plt.subplots(ncols = a, nrows = a, figsize = (20,20))
    for row in range(a):
        for col in range(a):
            if len(layer_list) > m:
                ax[row][col].hist(layer_list[m].get_weights()[0].reshape(-1,1))
                ax[row][col].set_title(str(layer_list[m].name))
                #fig.colorbar(im, ax=ax[row][col])
            m += 1

def plot_predictions(features, targets, prediction, name):
    '''
    Plot of a comparison between the known wear and the wear predicted for each 
    of the input features
    '''
    
    fig = plt.figure(figsize=(5,5))
    G = gridspec.GridSpec(3, 2)
    
    fig.suptitle(name, fontsize=16)
    
    axes1 = plt.subplot(G[2, :])
    axes2 = plt.subplot(G[0, 1])
    axes3 = plt.subplot(G[1, 0])
    axes4 = plt.subplot(G[1, 1])
    axes5 = plt.subplot(G[0, 0])
    
    axes1.plot(features.engagements, targets, 'x', color = 'g', alpha = 0.1)
    axes1.plot(features.engagements, prediction, 'x', color = 'r', alpha = 0.1)
    axes2.plot(features.lsp, targets, 'x', color = 'g', alpha = 0.1)
    axes2.plot(features.lsp, prediction, 'x', color = 'r', alpha = 0.1)
    axes3.plot(features.bsp, targets, 'x', color = 'g', alpha = 0.1)
    axes3.plot(features.bsp, prediction, 'x', color = 'r', alpha = 0.1)
    axes4.plot(features.vc, targets, 'x', color = 'g', alpha = 0.1)
    axes4.plot(features.vc, prediction, 'x', color = 'r', alpha = 0.1)
    axes5.plot(features.hardness, targets, 'x', color = 'g', alpha = 0.1)
    axes5.plot(features.hardness, prediction, 'x', color = 'r', alpha = 0.1)
    
    axes1.set_title('Engagements')
    axes2.set_title('lsp')
    axes3.set_title('bsp')
    axes4.set_title('cutting velocity')
    axes5.set_title('hardness')
    plt.show()
    
def predictions_review(model, features_scaled, features, targets, name):
    '''
    Applies the trained model to a given set of samples and evaluates error
    statistics as well as plots the features by the predictions. Helps evaluating
    if the network is overfitting the data. 
    '''
    y_predicted = model.predict(features_scaled, verbose = True)
    plot_predictions(features, targets, y_predicted, name)
    print("Predictions average: %.03f" %y_predicted.mean())
    print('RMS: %.03f' %np.sqrt(mean_squared_error(targets, y_predicted)))
    print('MAE: %.03f' %mean_absolute_error(targets, y_predicted))
    print('MSE: %.05f' %mean_squared_error(targets, y_predicted))
    print('MedAE: %.05f' %median_absolute_error(targets, y_predicted))
    print('R2 score: %.03f' %r2_score(targets, y_predicted))
