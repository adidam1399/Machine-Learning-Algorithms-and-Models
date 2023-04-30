# Nearest Means classifier from scratch based on Euclidean Distance
# KRISHNA KAMAL ADIDAM

# Building a nearest-means classifier.

import numpy as np
import pandas as pd
import argparse
from plotDecBoundaries import plotDecBoundaries


#Writing function which trains the classifier

def Nearest_means_train(input_data_values,input_data_labels,C):
        """ Function to train the nearest means classifier """

        indices_class_1=[]
        indices_class_2=[]
        indices_class_3=[]
        #Getting indices for each class
        for i in range(input_data_labels.shape[0]):
            if(input_data_labels[i]==1):
                indices_class_1.append(i)
            elif(input_data_labels[i]==2):
                indices_class_2.append(i)
            elif(input_data_labels[i]==3):
                indices_class_3.append(i)
        #Getting values for that indices which belongs to that class
        class_1_values=input_data_values[indices_class_1]
        class_2_values=input_data_values[indices_class_2]
        if(C==3):
            class_3_values=input_data_values[indices_class_3]
        #Finding mean for each class
        class_1_mean=(np.mean(class_1_values,axis=0))
        class_2_mean=(np.mean(class_2_values,axis=0))
        #Returning the mean co-ordinates for each class
        if(C==2):
            return class_1_mean,class_2_mean
        elif(C==3):
            class_3_mean=(np.mean(class_3_values,axis=0))
        return class_1_mean,class_2_mean, class_3_mean


# Function which calculates nearest mean and classifies data as that corresponding class

def classification(values,mean_list,C):
    """ Function to classify the data based on the class means found from training data """

    labels_classified=[]
    for i in values:
        if(C==2):
            if(np.linalg.norm(i-mean_list[0])<np.linalg.norm(i-mean_list[1])):
                labels_classified.append(1)
            else:
                labels_classified.append(2)
        elif(C==3):
            if(np.linalg.norm(i-mean_list[0])==min(np.linalg.norm(i-mean_list[1]),np.linalg.norm(i-mean_list[2]),np.linalg.norm(i-mean_list[0]))):
                labels_classified.append(1)
            if(np.linalg.norm(i-mean_list[1])==min(np.linalg.norm(i-mean_list[1]),np.linalg.norm(i-mean_list[2]),np.linalg.norm(i-mean_list[0]))):
                labels_classified.append(2)
            if(np.linalg.norm(i-mean_list[2])==min(np.linalg.norm(i-mean_list[1]),np.linalg.norm(i-mean_list[2]),np.linalg.norm(i-mean_list[0]))): 
                labels_classified.append(3)
    return labels_classified
    

# ##### Plotting the training points, means and decision boundaries and regions after training


def plot_bdry_data(data,test,C):

    """ Function to find the means of each class based on training data in a pandas data frame and plot the corresponding decision boundary """
    data_train=pd.read_csv(data, header=None).values
    data_X=data_train[:,:-1]
    data_y=data_train[:,-1]
    if C==2:
        class_1_mean, class_2_mean=Nearest_means_train(data_X,data_y,2)
        mean_vector_list= np.array([class_1_mean,class_2_mean])
    elif C==3:
        class_1_mean, class_2_mean,class_3_mean=Nearest_means_train(data_X,data_y,3)
        mean_vector_list= np.array([class_1_mean,class_2_mean,class_3_mean])
    
    plotDecBoundaries(data_X,data_y,mean_vector_list)

    data_test=pd.read_csv(test, header=None).values
    data_X_test=data_test[:,:-1]
    data_y_test=data_test[:,-1]

    plotDecBoundaries(data_X_test,data_y_test,mean_vector_list)

    return mean_vector_list

def error_rate_calc(data_X, data_y, mean_vector_list,C):

    """ Function to calculate error rate based on the trained classifier """

    classified_labels=classification(data_X,mean_vector_list,C)
    count=0
    for i in range(len(classified_labels)):
        if(data_y[i]!=classified_labels[i]):
            count+=1
    error_rate=((count)/len(classified_labels))*100
    print("error rate for the dataset is : {0}".format(error_rate))







