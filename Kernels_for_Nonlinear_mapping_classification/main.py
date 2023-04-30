import argparse
from Nonlinear_mapping_through_Kernels import *

def main(args):
    kernel_transformation(args['training_data'],args['val_data'],args['test_data'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Command line args for training, testing and visualizing the Non-Linear Kernel Transformation Classifier Model implemented from scratch')
    parser.add_argument("-train","--training_data",help="Training data")
    parser.add_argument("-val","--val_data",help="Validation data")
    parser.add_argument("-test","--test_data",help="Testing data")
    args = vars(parser.parse_args())
    main(args)
