import tkinter
import re
import glob
import numpy as np
import pickle
from numpy import savetxt
import re



def dataRead(file_directory,Output_location):
    # file_directory = "D:\Work\Skripsie\DataLogging"
    Logs_array = glob.glob(file_directory+"\*")                                                         #Creates a list of all .log files in the given directory
    log_Names = []
    #print(len(Logs_array))
    
    
    for i in range(0,len(Logs_array)):                                                                #Stores the names of each of the files as it contains the degrees and weight
        name = Logs_array[i]
        # print(name[len(file_directory)+1:])
        log_Names.append(name[len(file_directory)+1:])
    
    data_list = []
    final_data_array = []
    j = 0                                                                                               # The loop converts all data into dictionary list containting the Gauge 1, Gauge 2, f_z , f_x
    lines_index = 0
   # print(len(log_Names))
    data_array = np.array((len))
    for log in Logs_array:
        file = open(log)
        lines = file.read().splitlines()
        line_counter = 0
       # print(j)
        for line in lines:
            if not(('Gauge' in line) or ('log' in line)):
                if((line_counter>=500) and (line_counter<=1500)):
                    data_list_array_data = []
                
                
                    data_list.append(line)
                    #print((re.sub(r'(deg)',r'\1_',log_Names[j])))
                    degree,cut1,weight,cut2 = (re.sub(r'(deg)',r'\1_',log_Names[j])).split("_")
                    f_z,f_x = newton_reduction(degree, weight)
                    gauge_1,gauge_2=line.split(":")
                    data_list_array_data.append(gauge_1)
                    data_list_array_data.append(gauge_2)
                    data_list_array_data.append("{:.2f}".format(f_z))
                    data_list_array_data.append("{:.2f}".format(f_x))
                    data_list_array_data.append(degree)
                    data_list_array_data.append(weight)
                    #print(data_list_array_data)
                    final_data_array.append(data_list_array_data)
                    #print(data_list_array_data)
                line_counter += 1
        j = j+1;
                
   #print(final_data_array)
   #print(final_data_array)
   #  print(len(data_list))
    final_data_array = np.array(final_data_array)                                                       
    pickle_out = open("dataset.pickle","wb")
    pickle.dump(final_data_array, pickle_out)
    pickle_out.close()
    
    
    np.savetxt('D:\Work\Skripsie\data.csv', final_data_array,fmt = '%s',delimiter=',')
   #print(final_data_array)
   #  file = open(r'D:\Work\Skripsie\Coding\Python\putty.log','r')
   #  lines = file.read().splitlines()
   #  del lines[0:1]
   #  for line in lines:
   #      if not ('Gauge' in line):       
   #          Data_Points.append(line)
    
   # # print(Data_Points)
   #  file.close()
    Data_Preprocessing("dataset",Output_location)
    return final_data_array


def Read_All(file_directory,Output_location):
        # file_directory = "D:\Work\Skripsie\DataLogging"
    Logs_array = glob.glob(file_directory+"\*")                                                         #Creates a list of all .log files in the given directory
    log_Names = []
    #print(len(Logs_array))
    
    
    for i in range(0,len(Logs_array)):                                                                #Stores the names of each of the files as it contains the degrees and weight
        name = Logs_array[i]
        # print(name[len(file_directory)+1:])
        log_Names.append(name[len(file_directory)+1:])
    
    data_list = []
    final_data_array = []
    j = 0                                                                                               # The loop converts all data into dictionary list containting the Gauge 1, Gauge 2, f_z , f_x
    lines_index = 0
   # print(len(log_Names))
    data_array = np.array((len))
    for log in Logs_array:
        file = open(log)
        lines = file.read().splitlines()
        line_counter = 0
       # print(j)
        for line in lines:
            if not(('Gauge' in line) or ('log' in line)):
                data_list_array_data = []
                
                
                data_list.append(line)
                #print((re.sub(r'(deg)',r'\1_',log_Names[j])))
                # degree,cut1,weight,cut2 = (re.sub(r'(deg)',r'\1_',log_Names[j])).split("_")
                # f_z,f_x = newton_reduction(degree, weight)
                gauge_1,gauge_2=line.split(":")
                data_list_array_data.append(gauge_1)
                data_list_array_data.append(gauge_2)
                # data_list_array_data.append("{:.2f}".format(f_z))
                # data_list_array_data.append("{:.2f}".format(f_x))
                # data_list_array_data.append(degree)
                # data_list_array_data.append(weight)
                #print(data_list_array_data)
                final_data_array.append(data_list_array_data)
                #print(data_list_array_data)

        j = j+1;
                
   #print(final_data_array)
   #print(final_data_array)
   #  print(len(data_list))
    final_data_array = np.array(final_data_array)                                                       
    pickle_out = open(Output_location+".pickle","wb")
    pickle.dump(final_data_array, pickle_out)
    pickle_out.close()





def newton_reduction(degree,weight):                                                                #Converts the weight and angles from the log file names into their f_z and f_x forces
    f_z = (float(weight)/1000*9.81)*np.cos(np.radians(float(degree)))
    f_x = (float(weight)/1000*9.81)*np.sin(np.radians(float(degree)))
    
    return f_z,f_x

def Data_Preprocessing(input_pickle,output_pickle):
    
    pickle_out = open(input_pickle+".pickle","rb")
    data_array = pickle.load(pickle_out)
    
    data_array = data_array.astype(float)
   # Original_array = data_array
    Original_array = np.copy(data_array)
    # print(round(data_array.shape[0]*0.77))
   # print(Original_array)
   # data_array[:,[0,1]] = tf.keras.utils.normalize(data_array[:,[0,1]],axis = 1)

    random_indices_training = np.random.choice(data_array.shape[0], size= round(data_array.shape[0]*0.85), replace=False)
    Training_data = data_array[random_indices_training,:]
    Rem_array = np.delete(data_array, random_indices_training,axis = 0)
    #random_indices_validation = np.random.choice(Rem_array.shape[0],size = 10836,replace = False)
    #Validation_data = data_array[random_indices_validation,:]
    Test_data = Rem_array
    
  
    
    Training_data_X = Training_data[:,[0,1]]
    Training_data_Y = Training_data[:,[2,3]]
    
   # Validation_data_X = Validation_data[:,[0,1]]
    #Validation_data_Y = Validation_data[:,[2,3]]
    
    Test_data_X = Test_data[:,[0,1]]
    Test_data_Y = Test_data[:,[2,3]]
    
    file = open(output_pickle+'.pickle','wb')
    pickle.dump(Training_data_X, file)
    pickle.dump(Training_data_Y, file)
   # pickle.dump(Validation_data_X, file)
    #pickle.dump(Validation_data_Y, file)
    pickle.dump(Test_data_X, file)
    pickle.dump(Test_data_Y, file)
    pickle.dump(data_array,file)
    file.close()


dataRead("D:\Work\Skripsie\DataLogging - Copy", "Processed_dataset")
dataRead("D:\Work\Skripsie\Data_test", "Processed_dataset_Test")
# Read_All("D:\Work\Skripsie\Testing Data\Temperature", "Complete_dataset")