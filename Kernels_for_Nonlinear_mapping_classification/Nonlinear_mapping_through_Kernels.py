
# Goal - Perform non-linear mapping thorugh Kernel transformation for classification in the non=linear transformed space

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_regions import *

# Read the data
def read_data(data):
    Data_read=pd.read_csv(data, header=None)
    return np.array(Data_read.iloc[:])

# Coding a 2-class kernel classifier that can use RBF Kernel or Linear Kernel

# Defining a function which calculates individual components of RBF kernel's discriminant function - Kernel Trick (for original Nearest Means classifier)

# Function which returns discriminant function (g(x)) of RBF Kernel
def RBF_components(class_1_data, class_2_data, gamma, data):
    comp_1=0
    for i in range(len(class_1_data)):
        comp_1+=np.exp(-gamma*((data-class_1_data[i]) @ (data-class_1_data[i]).T))
    comp_1=comp_1*2/len(class_1_data)
    comp_2=0
    for i in range(len(class_2_data)):
        comp_2+=np.exp(-gamma*((data-class_2_data[i]) @ (data-class_2_data[i]).T))
    comp_2=comp_2*2/len(class_2_data)
    comp_3=0
    for i in range(len(class_1_data)):
        for j in range(len(class_1_data)):
            comp_3+=np.exp(-gamma*((class_1_data[i]-class_1_data[j]) @ (class_1_data[i]-class_1_data[j]).T))
    comp_3=comp_3/(len(class_1_data)**2)
    comp_4=0
    for i in range(len(class_2_data)):
        for j in range(len(class_2_data)):
            comp_4+=np.exp(-gamma*((class_2_data[i]-class_2_data[j]) @ (class_2_data[i]-class_2_data[j]).T))
    comp_4=comp_4/(len(class_2_data)**2)
    return comp_1-comp_2-comp_3+comp_4


# Defining a function which calculates individual components of Linear kernel's discriminant function

# Function which returns discriminant function of Linear Kernel
def Linear_kernel_components(class_1_data, class_2_data, data):
    comp_1=0
    for i in range(len(class_1_data)):
        comp_1+=class_1_data[i]@data.T
    comp_1=comp_1*2/len(class_1_data)
    comp_2=0
    for i in range(len(class_2_data)):
        comp_2+=class_2_data[i]@data.T
    comp_2=comp_2*2/len(class_2_data)
    comp_3=0
    for i in range(len(class_1_data)):
        for j in range(len(class_1_data)):
            comp_3+=class_1_data[i] @ class_1_data[j].T
    comp_3=comp_3/(len(class_1_data)**2)
    comp_4=0
    for i in range(len(class_2_data)):
        for j in range(len(class_2_data)):
            comp_4+=class_2_data[i] @ class_2_data[j].T
    comp_4=comp_4/(len(class_2_data)**2)
    return comp_1-comp_2-comp_3+comp_4



# Functions to predict the classes based on discriminant functions for both RBF and Linear Kernel
def RBF_nearest_means(class_1_data, class_2_data, gamma, data,label):
    g_u=RBF_components(class_1_data, class_2_data,gamma, data)
    if(g_u>0):
        return label[0]
    else:
        return label[1]
def Linear_nearest_means(class_1_data, class_2_data, data,label):
        g_u=Linear_kernel_components(class_1_data, class_2_data, data)
        if(g_u>0):
            return label[0]
        else:
            return label[1]

# Function to find the error rate of predicted labels compared to actual labels

def error_calc(act_labels, pred_labels):
    count=0
    for i in range(len(act_labels)):
        if act_labels[i]!=pred_labels[i]:
            count+=1
    return (count/len(act_labels))*100

# Function to calculate error on any data for RBF Kernel classifier

def RBF_classify(class1_data,class2_data,gamma,data,labels):
    pred=[]
    for i in range(len(data)):
        pred.append(RBF_nearest_means(class1_data,class2_data,gamma,data[i],np.unique(labels)))
    return error_calc(labels,pred)

# Function to calculate error on any data for Linear Kernel classifier

def Linear_classify(class1_data, class2_data, data, labels):
    pred_list=[]
    for i in range(len(data)):
        pred_list.append(Linear_nearest_means(class1_data,class2_data,data[i],np.unique(labels)))
    return error_calc(pred_list,labels)



# Function which returns predicted labels for Linear Kernel function
def Linear_kernel_function(class1_data,class2_data,data_to_predict,label):
    pred_list_linear=[]
    for i in range(len(data_to_predict)):
        pred_list_linear.append(Linear_nearest_means(class1_data,class2_data,data_to_predict[i],label))
    return np.array(pred_list_linear)

# Function which returns predicted labels for RBF Kernel function
def RBF_kernel_function(class1_data,class2_data,gamma,data_to_predict,label):
    pred_list=[]
    for i in range(len(data_to_predict)):
        pred_list.append(RBF_nearest_means(class1_data,class2_data,gamma,data_to_predict[i],label))
    return np.array(pred_list)




# Perform Kernel Trick on a dataset and plot the results in the orignal space after inverse transformation

def kernel_transformation (training_data, val_data, test_data):

    train_data=read_data(training_data)
    train_X=train_data[:,:-1]
    class_1_train_X=train_data[train_data[:,-1]==1][:,:-1]
    class_2_train_X=train_data[train_data[:,-1]==2][:,:-1]
    train_y=train_data[:,-1]

    val_data=read_data(val_data)
    val_X=val_data[:,:-1]
    val_y=val_data[:,-1]

    test_data=read_data(test_data)
    test_X=test_data[:,:-1]
    test_y=test_data[:,-1]


    # Checking various values of gamma (RBF Kernel) on validation set to get optimal gamma

    k_list=[]
    error_list=[]
    for k in np.arange(-2,2,0.1):
        k_list.append(k)
        error_list.append(RBF_classify(class_1_train_X,class_2_train_X,10**k,val_X,val_y))
    plt.title("Error on validation set vs K (log(gamma))")
    plt.plot(k_list,error_list)
    plt.show()



    # finding the test set error for RBF Kernel with optimal gamma as well as with Linear kernel
    gamma_opt=k_list[np.argmin(error_list)]


    error_rate_test_RBF=RBF_classify(class_1_train_X,class_2_train_X,gamma_opt, test_X,test_y)
    print("Error rate of test set using RBF Kernel is {0}".format(error_rate_test_RBF))
    print("Error rate of test set using Linear Kernel is {0}".format(Linear_classify(class_1_train_X,class_2_train_X,test_X,test_y)))


    # Plotting the training data and decision regions using Linear Kernel

    fig1=plot_region(train_X,train_y,class_1_train_X,class_2_train_X,Linear_kernel_function,RBF_kernel_function)

    # Plotting the training data and decision regions using RBF Kernel

    fig2=plot_region(train_X,train_y,class_1_train_X,class_2_train_X,RBF_kernel_function,RBF_kernel_function,gamma_opt)


    # Plotting repeated for gamma = various scales of optimal gamma

    gamma_list=[0.01*gamma_opt, 0.1*gamma_opt,0.3*gamma_opt,3*gamma_opt,10*gamma_opt,100*gamma_opt]
    for i in gamma_list:
        print("For gamma = {0}".format(i))
        plot_region(train_X,train_y,class_1_train_X,class_2_train_X,RBF_kernel_function,i)



