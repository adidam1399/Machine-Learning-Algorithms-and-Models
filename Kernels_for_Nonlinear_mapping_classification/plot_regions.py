
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Nonlinear_mapping_through_Kernels import *

# Writing plotting helper functions

def plot_region(training, label_train, class1_data, class2_data,decision_function,RBF_kernel_function, gamma=0):
    
    #Converting data to numpy array if it is other than numpy array
    if isinstance(training, pd.DataFrame):
        training = training.to_numpy()
    if isinstance(label_train, pd.DataFrame):
        label_train = label_train.to_numpy()

    classes = np.unique(label_train)
    nclass = len(classes)
    class_names = []
    for c in classes:
        class_names.append('Class ' + str(int(c)))

    # Setting the feature range for plotting
    # Max range and min range for x1
    max_x1 = np.ceil(np.max(training[:, 0])) + 1.0
    min_x1 = np.floor(np.min(training[:, 0])) - 1.0
    # Max and min range for x2
    max_x2 = np.ceil(np.max(training[:, 1])) + 1.0
    min_x2 = np.floor(np.min(training[:, 1])) - 1.0

    #Getting x and y ranges
    xrange = (min_x1, max_x1)
    yrange = (min_x2, max_x2)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.05


    (x1, x2) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                           np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x1, x2) image
    image_size = x1.shape
    # make (x1, x2) pairs as a bunch of row vectors.
    grid_2d = np.hstack((x1.reshape(x1.shape[0] * x1.shape[1], 1, order='F'),
                         x2.reshape(x2.shape[0] * x2.shape[1], 1, order='F')))

    # Labels for each (x1, x2) pair by prediction.
    # Prediction based on each kernel function
 

    if(decision_function==RBF_kernel_function):
        pred_label = decision_function(class1_data,class2_data,gamma,grid_2d,classes)
    else:
        pred_label = decision_function(class1_data,class2_data,grid_2d,classes)

    # reshape the idx (which contains the class label) into an image.
    decision_map = pred_label.reshape(image_size, order='F')

    # creating figure
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
    ax.legend()
    # Plotting the final figure
    plt.tight_layout()
    plt.show()
    return fig