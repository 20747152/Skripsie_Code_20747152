import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import sys
sys.path.insert(1, 'D:\Work\Skripsie\Coding\Python')
from Code_Backend import dataRead


if __name__ == "__main__":


    
    file = open('Processed_dataset.pickle','rb')
    
    Training_X = pickle.load(file)
    Training_Y = pickle.load(file)
    

    Test_X = pickle.load(file)
    Test_Y = pickle.load(file)
    
    Match_array =  pickle.load(file)
    file.close()
    

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(2,activation = None))
    
    model.compile(optimizer ='adam',
                  loss = "mse",
                  metrics = ["mse",'mae'])
    
    model_test = model.fit(Training_X,Training_Y,validation_split = 0.10,epochs = 500)
    plt.figure(dpi = 1200)
    plt.plot(model_test.history['loss'])
    plt.plot(model_test.history['val_loss'])
    # plt.title('1 Hidden Layers',fontsize = 15)
    plt.xlim(0, 500)
    plt.ylim(0, 1)
    plt.ylabel('MSE' , fontsize = 15)
    plt.xlabel('Epoch', fontsize = 15)
    plt.legend(['train', 'validation'], loc='upper left',prop={'size': 12})
    plt.show()
    model.save('Regression32_NN.model')
    
    

