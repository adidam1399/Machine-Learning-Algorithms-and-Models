# Perceptron Algorithm


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import PlotDecisionBoundaries


# Defining a function which converts normal data points to reflected data points

def data_reflect(values,labels):
    for i in range(len(values)):
        if(labels[i]==2):
            values[i]*=-1
    return values


# Defining a function to Convert the data to augmented Space

def convert_augmented(data):
    Aug_data=[]
    for i in range(len(data)):
        k=np.hstack((1,np.array(data[i])))
        Aug_data.append(k)
    return np.array(Aug_data)


# Defining a function which performs Perceptron Learning

def Perceptron_train(data,eta,D,labels,epochs=10000):
    Augm_data=convert_augmented(data)
    ref_data=data_reflect(Augm_data,labels)
    #Shuffling the data
    shuffled_indices=random.sample([i for i in range(len(data))],len(data))
    W=(np.ones(D+1)*0.1)
    epoch=0
    weight_list=[]
    cost_list=[]
    #Running for epochs no. of iterations
    while epoch<=epochs:
        Cost_J=0
        for j in shuffled_indices:
            if(np.matmul(W,(ref_data[j]))<=0):
                W+=(eta*ref_data[j].T)
                Cost_J+=np.dot(W,ref_data[j])
        #If the cost is zero, all points are correctly classified
        if Cost_J==0:
           print("Data is Linearly seperable")
        #Appending weights and costs after 9500 iterations
        if epoch>=9500:
            weight_list.append(W.reshape(3,1))
            cost_list.append(Cost_J)
        if epoch==epochs:
            weight_index=min(range(len(cost_list)),key=lambda k:abs(cost_list[k]))
            weight_final=weight_list[weight_index]
            Cost=cost_list[weight_index]
            print("Data is Not linearly seperable")
        epoch+=1
    return weight_final,-1*Cost


# Function to classify data based on trained algorithm's weight vector

def classify_data(W,data):
    classified_labels=[]
    aug_data=convert_augmented(data)
    for i in range(len(data)):
        if(np.dot(W.T,aug_data[i].reshape(3,1)))>0:
            classified_labels.append(1)
        else:
            classified_labels.append(2)
    return classified_labels


# Function to get error rate

def error_rate_calc(pred_labels, act_labels):
    er_count=0
    for i in range(len(pred_labels)):
        if(pred_labels[i]!=act_labels[i]):
            er_count+=1
    return er_count/len(pred_labels)*100

    

# Function to perform Perceptron Classification For a dataset 

def train_perceptron(train_data,test_data):

    train_data=(pd.read_csv(train_data,header=None)).values
    train_X=train_data[:,[0,1]]
    train_y=train_data[:,-1]

    weight_train,cost_train=Perceptron_train(train_X,1,2,train_y)
    classified_labels=classify_data(weight_train,train_X)

    error_rate_train=error_rate_calc(classified_labels,train_y)
    print("final weight vector is {0}".format(weight_train))
    print("The error rate on training set is {0}:".format(error_rate_train))
    print("The cost after iterations is {0}:".format(cost_train))

    #  Plotting the decision region and boundary for training set

    PlotDecisionBoundaries.plot_Decision_boundaries(train_X,train_y,weight_train)

    # For test set

    test_data=(pd.read_csv(test_data,header=None)).values
    test_X=test_data[:,[0,1]]
    test_y=test_data[:,-1]

    classified_labels_test=classify_data(weight_train,test_X)
    error_rate_test=error_rate_calc(classified_labels_test,test_y)

    print("The error rate on test set f is {0}:".format(error_rate_test))

    # Plotting the data and Dec Bdry for test set

    PlotDecisionBoundaries.plot_Decision_boundaries(test_X,test_y,weight_train)


