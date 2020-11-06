import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches 

def RMSE_value(predict,observ):
    
    delta_loss = (predict-observ)**2
    delta_loss = delta_loss.astype(float)
    delta_loss = np.sum(delta_loss)
    print(delta_loss)
    N = predict.size
    print(N)
    delta_loss = delta_loss/N
    RMSE = np.sqrt(delta_loss)
    return RMSE

def MAE_value(predict,observ):
    delta_loss = np.abs((predict[:]-observ[:]))
    delta_loss = delta_loss.astype(float)
    delta_loss = np.sum(delta_loss)
    N = predict.size
    MAE = delta_loss/N
    return MAE

def Maximum_error(predict,observ):
    delta_loss = np.abs((predict[:]-observ[:]))
    max = np.amax(delta_loss)
    
    return max
    
file = open("Processed_dataset.pickle",'rb')

Rem_X = pickle.load(file)
Rem_Y = pickle.load(file)


RemT_X = pickle.load(file)
RemT_Y = pickle.load(file)

throw = pickle.load(file)
file.close()











    
file = open('Processed_dataset_Test.pickle','rb')
    
Training_X = pickle.load(file)
Training_Y = pickle.load(file)
    
# Valid_X = pickle.load(file)
# Valid_Y = pickle.load(file)
    
Test_X = pickle.load(file)
Test_Y = pickle.load(file)
    
Match_array =  pickle.load(file)
file.close()

file = open("Complete_dataset.pickle",'rb')
Complete_dataset = pickle.load(file).astype(np.float)
file.close()






unique_complete = np.unique(Rem_Y,axis=0) 



complete_dataset_testing = np.concatenate([Match_array,throw])
# complete_dataset_testing = np.sort(complete_dataset_testing, axis = 0)


model = tf.keras.models.load_model('Regression32_NN.model')
test = model.predict([[Match_array[:,[0,1]]]])
test_complete = model.predict([[complete_dataset_testing[:,[0,1]]]])


final_test_array = np.concatenate([test_complete, complete_dataset_testing[:,[2,3]]], axis=1)
final_test_array_organized = np.sort(final_test_array,axis = 0)

RMSE_z = RMSE_value(test_complete[:,0], complete_dataset_testing[:,2])
RMSE_x = RMSE_value(test_complete[:,1], complete_dataset_testing[:,3])
# MAE_z = MAE_value(test[:,0], Test_Y[:,0])
# MAE_x = MAE_value(test[:,1], Test_Y[:,1])


# RMSE_z_complete = RMSE_value(test_remaining[:,0], Rem_Y[:,0])
# RMSE_x_complete = RMSE_value(test_remaining[:,1], Rem_Y[:,1])
# # MAE_z_complete = MAE_value(test_remaining[:,0], Rem_Y[:,0])
# # MAE_x_complete = MAE_value(test_remaining[:,1], Rem_Y[:,1])

max_z = Maximum_error(test_complete[:,0], complete_dataset_testing[:,2])
max_x = Maximum_error(test_complete[:,1], complete_dataset_testing[:,3])

# max_z_complete = Maximum_error(test_remaining[:,0], Rem_Y[:,0])
# max_x_complete = Maximum_error(test_remaining[:,1], Rem_Y[:,1])


unique,Index = np.unique(Test_Y,axis = 0,return_index = True)
# x_ydata=np.linspace(1, unique.shape[0],unique.shape[0])
# x_xdata = np.linspace(1,test.shape[0],test.shape[0])
plt.figure(dpi=1200)
plt.title("True and Predicted $F_{z}$ and $F_{x}$ Values",fontsize = 15)
plt.ylabel('$F_{z}$ (Newton)',fontsize = 15)
plt.xlabel('$F_{x}$ (Newton)',fontsize = 15)

plt.scatter(unique[:,1],unique[:,0],c = 'brown' , alpha=0.2,s=70)
plt.scatter(test[:,1],test[:,0],c='red', alpha = 0.2, s = 5)

plt.scatter(unique_complete[:,1], unique_complete[:,0], c = 'blue' , alpha = 0.2, s=70)
plt.scatter(test_complete[:,1], test_complete[:,0], c = 'red' , alpha = 0.2, s=5)

# plt.scatter(Test_Y[:,0],Test_Y[:,1], c = 'red',alpha=0.1,s=20)
# rect1 = patches.Rectangle((36.5,-1.5), width = 3,height = 2.8, linewidth = 1,edgecolor = 'black',facecolor = 'lime',alpha = .2)
# rect1 = patches.Rectangle((-34.5,17.5), width = 3,height = 2.8, linewidth = 1,edgecolor = 'black',facecolor = 'lime',alpha = .2)
# plt.gca().add_patch(rect1)
plt.legend(['Out-of-Sample True Value','NN Prediction','In-Sample True Value'], loc='upper left', prop={'size': 10})
plt.show()






plt.figure(dpi=1200)
plt.ylim(17.5,20.3)
plt.xlim(-34.5,-31.5)
plt.title("True and Predicted $F_{z}$ and $F_{x}$ Values",fontsize = 15)
plt.ylabel('$F_{z}$ (Newton)',fontsize = 15)
plt.xlabel('$F_{x}$ (Newton)',fontsize = 15)
plt.scatter(unique_complete[:,1],unique_complete[:,0],c = 'brown' , alpha=1,s=40)
plt.scatter(test[:,1],test[:,0],c='red', alpha = 0.2, s = 5)
plt.legend(['In-Sample True Value','NN Prediction'], loc='upper left',prop={'size': 12})
plt.show()


plt.figure(dpi = 1200)
plt.title("$F_{z}$ Force Estimation",fontsize = 15)
plt.ylabel("$F_{z}$ (Newton)",fontsize = 15)
plt.xlabel("Sample",fontsize = 15)
plt.plot(test[:,0], c = 'red',linewidth = 0.5)
plt.plot(Match_array[:,2], c = 'blue',linewidth = 0.4,linestyle = '--')
plt.legend(['NN Prediction','True Value'], loc='upper left', prop={'size': 12})
plt.show()


plt.figure(dpi = 1200)
plt.title("$F_{x}$ Force Estimation",fontsize = 15)
plt.ylabel("$F_{x}$ (Newton)",fontsize = 15)
plt.xlabel("Sample",fontsize = 15)
plt.plot(test[:,1], c = 'red',linewidth = 0.5)
plt.plot(Match_array[:,3], c = 'blue',linewidth = 0.4,linestyle = '--')
plt.legend(['NN Prediction','True Value'], loc='upper left', prop={'size': 12})
plt.show()







# plt.figure(dpi=1200)
# plt.ylim(18,20)
# plt.xlim(-34.5,-31.5)
# plt.title("True and predicted $F_{z}$ and $F_{x}$ values for testing data")
# plt.ylabel('$F_{z}$ (Newton)')
# plt.xlabel('$F_{x}$ (Newton)')
# plt.scatter(unique[:,1],unique[:,0],c = 'blue' , alpha=1,s=20)
# plt.scatter(test[:,1],test[:,0],c='red', alpha = 0.2, s = 20)
# plt.legend(['True Value','NN Prediction'], loc='upper right')
# # plt.scatter(Test_Y[:,0],Test_Y[:,1], c = 'red',alpha=0.1,s=20)
# plt.show()








# plt.figure(dpi=1200)
# plt.plot(Test_Y[:,0], c = 'blue', linewidth = 1,linestyle = 'dashed')
# plt.plot(test[:,0], c='red',linewidth = 0.5)
# rect1 = patches.Rectangle((100,20), width = 1000,height = 15, linewidth = 1,edgecolor = 'black',facecolor = 'lime',alpha = .2)
# rect2 = patches.Rectangle((8500,10), width = 550, height = 5, linewidth = 1,edgecolor = 'black',facecolor = 'lime',alpha = .2)
# plt.gca().add_patch(rect1)
# plt.gca().add_patch(rect2)
# plt.legend(['True Value','NN Prediction'], loc='upper right')
# plt.title("In-sample testing data")
# plt.ylabel("$F_{z}$ (Newton)")
# plt.xlabel("Sample")
# plt.show()


# plt.figure(dpi=1200)
# plt.plot(Test_Y[:,1], c = 'blue', linewidth = 1,linestyle = 'dashed')
# plt.plot(test[:,1], c='red',linewidth = 0.5)
# plt.legend(['True Value','NN Prediction'], loc='upper right')
# plt.title("In-sample testing data")
# plt.ylabel("$F_{x}$ (Newton)")
# plt.xlabel("Sample")
# plt.show()










# plt.figure(dpi=1200)
# plt.ylim(20, 35)
# plt.xlim(100, 1100)
# plt.plot(Test_Y[:,0], c = 'blue', linewidth = 1,linestyle = 'dashed')
# plt.plot(test[:,0], c='red',linewidth = 0.5)
# plt.legend(['True Value','NN Prediction'], loc='upper right')
# plt.title("Focus block one")
# plt.ylabel("$F_{z}$ (Newton)")
# plt.xlabel("Sample")
# plt.show()



# plt.figure(dpi=1200)
# plt.ylim(10, 15)
# plt.xlim(8500, 9050)
# plt.plot(Test_Y[:,0], c = 'blue', linewidth = 1,linestyle = 'dashed')
# plt.plot(test[:,0], c='red',linewidth = 0.5)
# plt.plot([100,9050],[12.1,12.1], c='black',linewidth = .5)
# plt.plot([100,9050],[11.95,11.95], c='black',linewidth = .5)
# plt.legend(['True Value','NN Prediction'], loc='upper right')
# plt.title("Focus block two")
# plt.ylabel("$F_{z}$ (Newton)")
# plt.xlabel("Sample")
# plt.show()













































# plt.figure(dpi=900)
# # plt.ylim(-50, 300)
# plt.title("Out-of-Sample Testing Graph for $F_{z}$")
# plt.ylabel('$F_{z}$ (Newton)')
# plt.xlabel('Sample')
# plt.plot(Test_Y[:,0], c= 'blue',linewidth = 1,linestyle = 'dashed')
# plt.plot(test[:,0],c = 'red' ,linewidth = .5)

# plt.legend(['True Value','NN Prediction'], loc='upper right')
# plt.show()




# plt.figure(dpi=900)
# plt.title("Out-of-Sample Testing Graph for $F_{x}$")
# plt.ylabel('$F_{x}$ (Newton)')
# plt.xlabel('Sample')
# plt.plot(Test_Y[:,1],c = 'blue' ,linewidth = 1,linestyle = 'dashed')
# plt.plot(test[:,1], c = 'red' ,linewidth = .5)

# plt.legend(['True Value','NN Prediction'], loc='upper right')
# plt.show()



# plt.figure(dpi=900)
# plt.ylim(-300, 50)
# plt.title("Gauge Readings")
# plt.ylabel('Gauge Reading')
# plt.xlabel('Sample')
# plt.plot(Complete_dataset[:,0],c = 'blue' ,linewidth = .5)
# plt.plot(Complete_dataset[:,1], c = 'red' ,linewidth = .5)
# plt.legend(['Gauge 1','Gauge 2'], loc='upper right')
# plt.show()












# plt.figure(dpi=900)
# plt.title("Neural Network Output Graph")
# plt.ylabel('Force (Newton)')
# plt.xlabel('Sample')
# plt.plot(test[:,0],c = 'blue' ,linewidth = .5)
# plt.plot(test[:,1], c = 'red' ,linewidth = .5)
# plt.legend(['$F_{y}$','$F_{x}$'], loc='upper right')
# plt.show()

# plt.figure(dpi=900)
# plt.title("Neural Network Output Graph")
# plt.ylabel('Force (Newton)')
# plt.xlabel('Sample')

# plt.legend(['$F_{x}$'], loc='upper right')
# plt.show()
















































# plt.figure(dpi=900)
# plt.title("Neural Network Predictions of Zero Input")
# plt.ylabel('Force (Newton)')
# plt.xlabel('Time (Minutes)')
# plt.plot(test[::600,0],c = 'blue' ,linewidth = .5)
# plt.plot(test[::600,1], c = 'red' ,linewidth = .5)
# plt.legend(['$F_{y}$','$F_{x}$'], loc='upper right')
# plt.show()






# plt.figure(dpi=900)
# data_amount = Complete_dataset.shape[0]



# Gauge1 = Complete_dataset[::600,0]
# Gauge2 = Complete_dataset[::600,1]
# minutes = np.linspace(1,Complete_dataset[::600,0].shape[0],Complete_dataset[::600,0].shape[0])



# Gauge1 = Gauge1.astype(np.float)
# Gauge2 = Gauge2.astype(np.float)
# plt.title("Gauge Output with Zero Input")
# plt.ylabel('Gauge Reading')
# plt.xlabel('Time (Minutes)')
# plt.plot(minutes,Gauge1,c="blue",label = 'Gauge 1',linewidth = .5)
# plt.plot(minutes,Gauge2,c="red",label = 'Gauge 2',linewidth = .5)
# plt.legend(['Gauge 1','Gauge 2'], loc='upper right')
# plt.show()

fx = np.array([0.1096,0.1784,0.1708,0.5127])
fz = np.array([0.0735,0.0983, 0.2651, 0.2688])
data_points = np.array([73,64,37,9])

plt.title("RMSE of NN Trained With Different Amounts of Data Points")
plt.plot(data_points,fx)
plt.plot(data_points,fz)
