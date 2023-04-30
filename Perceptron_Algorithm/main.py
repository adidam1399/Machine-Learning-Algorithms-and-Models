
import argparse
from Perceptron_Algorithm import *

def main(args):
    print(args)
    train_perceptron(args['training_data'],args['test_data'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'CLI args for training, testing and visualizing the Perceptron classifier')
    parser.add_argument("-train","--training_data",help="Training data")
    parser.add_argument("-test","--test_data",help="Testing data")
    args = vars(parser.parse_args())
    main(args)