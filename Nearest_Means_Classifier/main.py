import argparse
from Nearest_means_classifier import *

def main(args):
    print(args)
    mean_val_list=plot_bdry_data(args['training_data'],args['test_data'],int(args['classes']))
    
    




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'CLI args for training, testing and visualizing the Nearest Means classifier')
    parser.add_argument("-train","--training_data",help="Training data")
    parser.add_argument("-test","--test_data",help="Testing data")
    parser.add_argument("-classes","--classes",help="number of classes")
    args = vars(parser.parse_args())
    main(args)

