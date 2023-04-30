
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn
import h5_w7_helper_functions

def read_data(data):
    Data_read=pd.read_csv(data)
    return np.array(Data_read.iloc[:, :-1]), np.array(Data_read.iloc[:,-1])


# Reading the data and plotting in Non-augmented Space

data, labels=read_data("/Users/krishnakamaladidam/Downloads/h5w7_pr3_python_files/h5w7_data.csv")
data_class_1=data[labels==1]
data_class_2=data[labels==2]
plt.figure(figsize=(10,8))
plot_1=plt.scatter(data_class_1[:,0],data_class_1[:,1],c='r',marker='x')
plot_2=plt.scatter(data_class_2[:,0],data_class_2[:,1],c='b',marker='o')
plt.legend((plot_1,plot_2),("Class-1","Class-2"))
plt.show()


# Using sklearn Perceptron and obtaining the accuracy


from sklearn.linear_model import Perceptron
perceptron=Perceptron(fit_intercept=False)
perceptron.fit(data,labels)
accuracy_linear=perceptron.score(data,labels)
perceptron_weight=perceptron.coef_[0]
print("Accuracy score for perceptron classifier is {0}".format(accuracy_linear))
print("Weight vector for perceptron classifier in non-expanded space is {0}".format(perceptron_weight))


# Plotting the learned decision boundaries from perceptron

figure_1=h5_w7_helper_functions.plot_perceptron_boundary(data,labels,perceptron_weight,h5_w7_helper_functions.linear_decision_function)

# Function to expand the feature space (To quadratic feature space)

def feature_expansion(train_data):
    expanded_feature=[]
    for i in range(len(train_data)):
        expanded_feature.append([train_data[i,0],train_data[i,1],train_data[i,0]*train_data[i,1],train_data[i,0]**2,train_data[i,1]**2])
    return np.array(expanded_feature)

expanded_feature=feature_expansion(data)


# Training the classifier on expanded space

perceptron_exp=Perceptron(fit_intercept=False)
perceptron_exp.fit(expanded_feature,labels)
accuracy_exp=perceptron_exp.score(expanded_feature,labels)
weight_expanded=perceptron_exp.coef_[0]
print("Accuracy score for perceptron classifier in expanded feature space is {0}".format(accuracy_exp))

print("The learned weight vector from training the perceptron in expanded feature space is {0} ".format(weight_expanded))

# As we can see, the most relevant features in terms of absolute values of weights are feature 3 and 5 

# Creating a new feature matrix that contains only the two most relevant features

def convert_two_feature(data_expanded,a,b):
    converted_data=[]
    for i in range(len(data_expanded)):
        converted_data.append(data_expanded[i,[a,b]])
    return np.array(converted_data)


converted_features=convert_two_feature(expanded_feature,2,4)
converted_weight=np.array([weight_expanded[2],weight_expanded[4]])


# Plotting in expanded space with two most relevant features


figure_2=h5_w7_helper_functions.plot_perceptron_boundary(converted_features,labels,converted_weight,h5_w7_helper_functions.linear_decision_function)

#  The data is linearly seperable in this feature space

# Plotting the decision boundary and regions in the original feature space

figure_3=h5_w7_helper_functions.plot_perceptron_boundary(data,labels,converted_weight,h5_w7_helper_functions.nonlinear_decision_function)


# Modified nonlinear_decision_function

def nonlinear_decision_function(X, weight, labels):
    """
    Implements a non linear decision function
    :param X: feature matrix of dimension NxD
    :param weight: weight vector of dimension 1xD
    :param labels: possible class assignments
    :return:
    """
    # Modified non-linear decision function
    pred_labels=[]
    # Going over every data point and finding its discriminant function
    for i in range(len(X)):
        g_x=weight[0]*X[i,0]*X[i,1]+weight[1]*X[i,1]**2
        if(g_x>0):
            pred_labels.append(labels[0])
        else:
            pred_labels.append(labels[1])
    return np.array(pred_labels)


