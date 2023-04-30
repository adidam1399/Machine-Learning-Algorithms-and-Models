# Function to plot decision boundary and data points

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def plot_Decision_boundaries(data, labels, weight):
    
    #Feature range defining for plotting
    max_x = np.ceil(max(data[:, 0])) + 1
    min_x = np.floor(min(data[:, 0])) - 1
 
    plot_x = np.arange(min_x, max_x, 0.01) 
    #Plotting the data
    plt.plot(data[labels == 1, 0],data[labels == 1, 1], 'rx')
    plt.plot(data[labels == 2, 0],data[labels == 2, 1], 'go')
    #Specifying legend
    l1 = plt.legend(('Class-1', 'Class-2'), loc=4)
    plt.plot(plot_x, (-1 * weight[1] * plot_x - weight[0]) / weight[2], c='b', label = "decision_boundary")
    plt.gca().add_artist(l1)
    plt.show()