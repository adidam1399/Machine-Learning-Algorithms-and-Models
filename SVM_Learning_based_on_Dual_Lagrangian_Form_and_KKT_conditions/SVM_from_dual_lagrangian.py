
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Writing functions which find the optimal weight vector by doing the matrix inversion based on dual Lagrangian

# A* (rho)=B

def matrix_A_calc(z1,u1,z2,u2,z3,u3):
  """ Function to calculate matrix A"""
  row_1=[z1**2*(u1.T @ u1), 0.5*z1*z2*((u1.T @ u2)+(u2.T @ u1)), 
         0.5*z1*z3*((u1.T @ u3)+(u3.T @ u1)), -z1 ]
  row_2=[0.5*z1*z2*((u1.T @ u2)+(u2.T @u1)), z2*z2*(u2.T @ u2), 
         0.5*z2*z3*((u2.T @ u3)+(u3.T @ u2)), -z2 ]  
  row_3=[0.5*z3*z1*((u1.T @ u3)+(u3.T @u1)),0.5*z3*z2*((u3.T @ u2)+(u2.T @ u3)), 
         z3*z3*(u3.T @ u3), -z3 ]  
  row_4=[z1,z2,z3,0]
  return np.matrix([row_1,row_2,row_3,row_4])


def solve_matrix(z1,u1,z2,u2,z3,u3):
  """ Returning the vector of lambdas and mu after calculating A inverse"""
  A=matrix_A_calc(z1,u1,z2,u2,z3,u3)
  print(A)
  b=np.matrix([1,1,1,0]).T
  return (np.linalg.inv(A) @ b)


# Checking that lambda vector satisfies KKT conditions involving lambda (but not w)


def check_KKT_lambda(rho,z1,z2,z3):
  """Function which takes rho vector and checks KKT condition on lambda"""
  if((rho[0]*z1+rho[1]*z2+rho[2]*z3)==0 and rho[0]>=0 and rho[1]>=0 and 
     rho[2]>=0):
    return "KKT conditions involving lambda satisfied"
  else:
    print ("KKT conditions involving lambda not satisfied")
    return (rho[:3])


# Calculating the optimal (non augmented) weight vector and w0

def weight_calc(rho,z1,z2,z3,u1,u2,u3):
  """Function to calculate optimal (non augmented) weight vector"""
  return rho[0]*z1*u1+rho[1]*z2*u2+rho[2]*z3*u3

def w0_calc(w_opt,z,u):
  """Function to calculate bias weight w0"""
  return (1-(z*np.dot(w_opt.T  ,u)))/z

# Checking if the resulting weights satisfy the other KKT conditions

def check_KKT_weight(rho,u1,u2,u3,z1,z2,z3,w_opt,w0):
  """ Function to check resulting weights satisfy the other KKT conditions"""
  if(np.round_(rho[0]*((z1*(w_opt.T@u1+w0)-1)))==0 and 
     np.round_(rho[1]*(z2*(w_opt.T@u2+w0)-1))==0 
     and np.round_(rho[2]*(z3*(w_opt.T@u3+w0)-1))==0 and 
     (z1*((w_opt.T@u1)+w0)-1>=0)and
     (z2*((w_opt.T@u2)+w0)-1>=0) and (z3*((w_opt.T@u3)+w0)-1>=0)):
     return "KKT conditions on weights are satisfied"
  else:
    return "KKT conditions on weights are not satisfied"
    
# Helper functions for plotting

def linear_decision_boundary_function(X, weight, w0, labels):
    """Decision function is linear in the expanded space, so, decision boundary 
    and the class labels are calculated"""
    g_x = np.dot(X, weight)+w0
    pred_label = np.zeros((X.shape[0], 1))
    pred_label[g_x > 0] = labels[0]
    pred_label[g_x < 0] = labels[1]
    return pred_label

def plot_perceptron_boundary(training, label_train, weight, w0
                             ,decision_function):
  # Plot dec function, inspired from EE-559 HW
    """
    Plot the 2D decision boundaries of a linear classifier
    :param training: training data
    :param label_train: class labels correspond to training data
    :param weight: weights of a trained linear classifier. This
     must be a vector of dimensions (1, D)
    :param decision_function: a function that takes in a matrix with N
     samples and returns N predicted labels
    """
    # Checking if the training data is other than a numpy array, and converting 
    # it to numpy if it is'nt
    if isinstance(training, pd.DataFrame):
        training = training.to_numpy()
    if isinstance(label_train, pd.DataFrame):
        label_train = label_train.to_numpy()

    # Total number of classes
    classes = np.unique(label_train)
    nclass = len(classes)

    class_names = []
    for c in classes:
        class_names.append('Class ' + str(int(c)))

    # Set the feature range for plotting
    max_x1 = np.ceil(np.max(training[:, 0])) + 1.0
    min_x1 = np.floor(np.min(training[:, 0])) - 1.0
    max_x2 = np.ceil(np.max(training[:, 1])) + 1.0
    min_x2 = np.floor(np.min(training[:, 1])) - 1.0

    # Getting the x and y ranges
    xrange = (min_x1, max_x1)
    yrange = (min_x2, max_x2)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. This will be the basis of the decision
    # boundary visualization.
    (x1, x2) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                           np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x1, x2) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x1.shape
    # make (x1, x2) pairs as a bunch of row vectors.
    grid_2d = np.hstack((x1.reshape(x1.shape[0] * x1.shape[1], 1, order='F'),
                         x2.reshape(x2.shape[0] * x2.shape[1], 1, order='F')))

    # Labels for each (x1, x2) pair.
    # We get the labels from our decision function
    pred_label = decision_function(grid_2d, weight,w0, classes)
   
    #print(pred_label)
    # reshape the idx (which contains the class label) into an image.
    decision_map = pred_label.reshape(image_size, order='F')

    # creating the fig
    fig, ax = plt.subplots()
    # show the image, give each coordinate a color according to its class
    # label
    ax.imshow(decision_map, vmin=np.min(classes), vmax=9, cmap='Pastel1',
              extent=[xrange[0], xrange[1], yrange[0], yrange[1]],
              origin='lower')

    # plot the class training data.
    data_point_styles = ['rx', 'bo', 'g*']
    for i in range(nclass):
        ax.plot(training[label_train == classes[i], 0],
                training[label_train == classes[i], 1],
                data_point_styles[int(classes[i]) - 1],
                label=class_names[i])
    
  
    
    # Getting the legend
    ax.legend()
    
    # Plotting the figure
    plt.tight_layout()
    #plt.show()

    return fig


# ### Performing the SVM learning on dataset 1

# Defining u1, u2 , u3 for dataset 1
u1_1=np.array([1,2])
u2_1=np.array([2,1])
u3_1=np.array([0,0])
rho_1=solve_matrix(1,u1_1,1,u2_1,-1,u3_1)
print(rho_1)

check_KKT_lambda(rho_1,1,1,-1)

w_opt_1=weight_calc(rho_1,1,1,-1,u1_1,u2_1,u3_1).T
print("w optimal for dataset 1 is {0} ".format(w_opt_1))
w0_1=(w0_calc(w_opt_1,-1,u3_1))
print("w0 optimal for dataset 1 is {0} ".format(w0_1))

check_KKT_weight(rho_1,u1_1,u2_1,u3_1,1,1,-1,w_opt_1,w0_1)


fig_1=plot_perceptron_boundary(np.array([u1_1,u2_1,u3_1]),np.array([1,1,2]),
                               w_opt_1, w0_1,linear_decision_boundary_function)



# ### Performing the SVM learning on dataset 3


# Defining u1, u2 , u3 for dataset 3
u1_3=np.array([1,2])
u2_3=np.array([2,1])
u3_3=np.array([0,1.5])
rho_3=solve_matrix(1,u1_3,1,u2_3,-1,u3_3)

lambda_matrix=check_KKT_lambda(rho_3,1,1,-1)
print(lambda_matrix)


# As lambda-2 is less than zero, set it to zero. After setting it to zero, we get a new matrix for A, so programming it


def A_dataset_3(z1,u1,z3,u3):
  """ Defining matrix A for dataset 3, where lambda 2 was setup as 0"""
  row_1=[z1**2*(u1.T @ u1), 0.5*z1*z3*((u1.T @ u3)+(u3.T @ u1)), -z1 ]
  row_2=[0.5*z1*z3*((u1.T @ u3)+(u3.T @u1)), z3*z3*(u3.T @ u3), -z3 ] 
  row_3=[z1,z3,0]
  return np.matrix([row_1,row_2,row_3])


def solve_dataset_3(z1,u1,z3,u3):
    A=A_dataset_3(z1,u1,z3,u3)
    b=np.array([1,1,0])
    return (np.linalg.inv(A) @ b)


lambda_vector_dataset_3=solve_dataset_3(1,u1_3,-1,u3_3)
print(lambda_vector_dataset_3)
lambda_final=np.array([1.6, 0, 1.6, 2.2])


check_KKT_lambda(lambda_final,1,1,-1)

w_opt_3=weight_calc(lambda_final,1,1,-1,u1_3,u2_3,u3_3)
print("w optimal for dataset 3 is {0} ".format(w_opt_3))
w0_3=(w0_calc(w_opt_3,1,u1_3))
print("w0 optimal for dataset 3 is {0} ".format(w0_3))


check_KKT_weight(lambda_final,u1_3,u2_3,u3_3,1,1,-1,w_opt_3,w0_3)

fig_3=plot_perceptron_boundary(np.array([u1_3,u2_3,u3_3]),np.array([1,1,2]),
                               w_opt_3, w0_3,linear_decision_boundary_function)


