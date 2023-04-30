
import argparse
from Linear_Regression import *

def main(args):
    train_linear_regressor(args['training_data'],args['test_data'],int(args['dimension']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'CLI args for training, testing and visualizing the Linear Regression Model implemented from scratch')
    parser.add_argument("-train","--training_data",help="Training data")
    parser.add_argument("-test","--test_data",help="Testing data")
    parser.add_argument("-dimension","--dimension",help="dimensionality of the data")
    args = vars(parser.parse_args())
    main(args)
