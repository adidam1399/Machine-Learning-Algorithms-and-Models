
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Read the data

def read_data(data):
    Data_read=pd.read_csv(data)
    return np.array(Data_read.iloc[:, :-1]), np.array(Data_read.iloc[:,-1])

# Convert the data to augmented Space
def convert_augmented(data):
    Aug_data=[]
    for i in range(len(data)):
        k=np.hstack((1,(data[i])))
        Aug_data.append(k)
    return np.array(Aug_data)


# Calculate the RMS Error
def error(weight,data,output):
    error=0
    for i in range(len(data)):
        error+=(np.dot(weight,data[i].T)-output[i])**2/(len(data))
    return np.sqrt(error)


# Function which performs Linear Regression based on MSE 

def Linear_Regressor(data_train, output_train,A,B,Dimension):
    N=len(data_train)
    indices=random.sample([i for i in range(N)],N)
    data_train_aug=convert_augmented(data_train)
    weight=(np.ones(Dimension+1))*np.random.uniform(-0.1,0.1)
    rms_error=[]
    # Storing rms error before first iteration
    error_first=error(weight,data_train_aug,output_train)
    rms_error.append(error_first)
    epoch=0
    epoch_list=[]
    epoch_list.append(0)
    while epoch<100:
        for i in range(len(indices)):
            weight-=(A/(B+i))*((np.dot(weight,data_train_aug[indices[i]].T)-output_train[indices[i]]))*data_train_aug[indices[i]]
        rms_error.append(error(weight,data_train_aug,output_train))
        epoch_list.append(epoch+1)
        if(error(weight,data_train_aug,output_train)<0.001*error_first):
            print("First halting condition has reached")
            break
        epoch+=1
    print("Second halting conditon is reached, 100 epochs completed")
    return weight,rms_error,epoch_list

# Function to predict output based on optimal weight vector 
def predict(data,weight):
    return np.dot(weight, data.T)

# Function which does Training for different learning rates

def train_various_rates(data,output,D):
    A_list=[0.01, 0.1, 1, 10, 100]
    B_list=[1, 10, 100, 1000]
    weight_vectors=[]
    min_error_list=[]
    A_B_list=[]
    for i in A_list:
        for j in B_list:
            weight, error_list,epoch_list=Linear_Regressor(data, output,i,j,D)
            weight_vectors.append(weight)
            min_error_list.append((error_list[-1]))
            A_B_list.append((i,j))
            plt.plot(epoch_list,error_list,label="B: " + str(j))
            print("Optimal weight when regressor is trained with A={0} and B={1} is {2}".format(i,j,weight))
            print("Error when regressor trained with A={0} and B={1} is {2}".format(i,j,error_list[-1]))
            plt.legend()
        plt.show()
           
    return min(min_error_list),weight_vectors,min_error_list,A_B_list


# A trivial regressor which always outputs the means value of test labels

def trivial_regressor(test_y):

    output_trivial=np.mean(test_y)
    error_trivial=0
    for i in range(len(test_y)):
        error_trivial+=(output_trivial-test_y[i])**2/(len(test_y))
    print("RMS Error on test for trivial regressor is {0}".format(np.sqrt(error_trivial)))


# Training the model with the training data and verifying the performance on test data

def train_linear_regressor(data_train, data_test,D):

    train_X, train_y=read_data(data_train)
    final_error, weights_list, final_errors_list, A_B_list=train_various_rates(train_X,train_y,D)

    #  Checking which set of (A,B) achieves the minimum error

    print("The minimum error obtained is {0}".format(final_error))
    final_weight=weights_list[np.nanargmin(final_errors_list)]
    print("The weight vector corresponding to that is {0}".format(weights_list[np.nanargmin(final_errors_list)]))
    print("The set (A,B) that achieves it is {0}".format(A_B_list[np.nanargmin(final_errors_list)]))

    # Checking the performance on Test data

    test_X, test_y=read_data(data_test)
    augmented_test=convert_augmented(test_X)
    Error_test_set=error(final_weight,augmented_test,test_y)
    print("RMS Error on test set is {0}".format(Error_test_set))

    # Comparing the performance with a trivial regressor which always outputs the means value of test labels

    trivial_regressor(test_y)
    


