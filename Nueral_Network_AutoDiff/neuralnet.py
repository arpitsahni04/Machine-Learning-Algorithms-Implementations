'''

A Pure Numpy Implementation of an 2 Layer Vanila Nueral Network with Pytorch-style Autodiff.
Author: Arpit Sahni
Date: 20-November-2022 
'''

import numpy as np
import argparse
import logging
from typing import Callable
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             str, str, str, int, int, int, float]:
    '''
    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in* (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in* (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    '''
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:] # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:] # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    '''
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    '''
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def random_init(shape):
    '''
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    '''
    M, D = shape
    np.random.seed(M*D) # Don't change this line!
    
    # TODO: create the random matrix here! 
    # Hint: numpy might have some useful function for this
    W = np.random.uniform(low=-0.1, high=0.1, size=shape)

    return W


def zero_init(shape):
    '''
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    '''
    return np.zeros(shape = shape)


def softmax(z: np.ndarray) -> np.ndarray:
    '''
    :param z: input logits of shape (num_classes,)
    :return: softmax output of shape (num_classes,)
    '''
    return(np.exp(z)/np.exp(z).sum())


def cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    '''
    Compute cross entropy loss.
    :param y: label (a number or an array containing a single element)
    :param y_hat: prediction with shape (num_classes,)
    :return: cross entropy loss
    '''
    y_hot = np.zeros(y_hat.shape[0])
    y_hot[y] = 1
    return -np.dot(y_hot.T,np.log(y_hat))


def d_softmax_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    '''
    Compute gradient of loss w.r.t. ** softmax input **.

    :param y: label (a number or an array containing a single element)
    :param y_hat: predicted softmax probability with shape (num_classes,)
    :return: gradient with shape (num_classes,)
    '''
    y_hot = np.zeros(y_hat.shape[0])
    y_hot[y] = 1
    return y_hat-y_hot


class Sigmoid(object):
    def __init__(self):
        '''
        Initialize state for sigmoid activation layer
        '''
        # Create cache to hold values for backward pass
        self.cache: dict[str, np.ndarray] = dict()

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape (output_size,)
        '''
        e = np.exp(x)
        z = e/(1 + e) 
        assert z.shape == x.shape # z not operated on elementwise, shape != x.shape
        self.cache["z"] = z
        return z
    
    def backward(self, dz: np.ndarray) -> np.ndarray:
        '''
        :param dz: partial derivative of loss with respect to output of sigmoid activation
        :return: partial derivative of loss with respect to input of sigmoid activation
        '''
        return dz* self.cache["z"]*(1-self.cache["z"])

INIT_FN_TYPE = Callable[[tuple[int, int]], np.ndarray]

class Linear(object):
    def __init__(self, input_size: int, output_size: int, 
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        '''
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        '''

        self.w = weight_init_fn((output_size,input_size))
        self.w = np.hstack((np.zeros((self.w.shape[0],1)),self.w))
        self.dw = np.zeros(self.w.shape)
        self.lr = learning_rate
        self.cache: dict[str, np.ndarray] = dict()

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
        :return: output z of linear layer with shape (output_size,)

        '''
        bias_x = np.ones((1,1))
        x = x[...,np.newaxis]

        z =  np.vstack((bias_x, x)) # folding in bias
        b = np.dot(self.w,z) # z.Beta_T
        self.cache["z"]  = z
        b = b.reshape(b.shape[0])
        # print("b shape :" ,b)
        return b

    def backward(self, dz: np.ndarray) -> np.ndarray:
        '''
        :param dz: partial derivative of loss with respect to output z of linear
        :return: dx, partial derivative of loss with respect to input x of linear

        '''

        dz = dz[...,np.newaxis]
        self.dw = np.dot(dz,self.cache["z"].T) # g_beta
        dx = np.dot(self.w[:,1:].T ,dz) # g_b
        dx = np.squeeze(dx)
        return dx

    def step(self) -> None:
        '''
        Apply SGD update to weights using self.dw, which is
        set in NN.backward().
        '''
        self.w = self.w -(self.lr*self.dw)



class NN(object):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        '''
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        '''
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = Linear(self.input_size,  self.hidden_size, 
                 self.weight_init_fn, learning_rate)
        self.activation = Sigmoid()
        self.linear2 = Linear(self.hidden_size,self.output_size,
                              self.weight_init_fn, learning_rate)


    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        :param X: input data point *without the bias folded in*
        :param nn: neural network class
        :return: output prediction with shape (num_classes,). This should be 
                 a valid probability distribution over the classes.
        '''

        a =  self.linear1.forward(x)
        z = self.activation.forward(a)
        b = self.linear2.forward(z) # bias added inside linear 
        y_hat =softmax(b)
        return y_hat        

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        '''
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        :param nn: neural network class
        '''

        g_j = 1
        g_b = d_softmax_cross_entropy(y,y_hat)
        g_z = self.linear2.backward(g_b)
        g_a = self.activation.backward(g_z)
        g_x = self.linear1.backward(g_a) #we dicard 

    def step(self):
        '''
        Apply SGD update to weights.
        '''
        self.linear2.step()
        self.linear1.step() 

    def print_weights(self) -> None:
        '''
         logging to print out debugging infos.
        '''
        logging.debug(f"shape of w1: {self.linear1.w.shape}")
        logging.debug(self.linear1.w)
        logging.debug(f"shape of w2: {self.linear2.w.shape}")
        logging.debug(self.linear2.w)


def test(X: np.ndarray, y: np.ndarray, nn: NN) -> tuple[np.ndarray, float]:
    '''
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    '''
    labels = np.zeros(y.shape)
    for i in range(len(X)):
        labels[i] = np.argmax(nn.forward(X[i]))
    error_rate = np.around(np.mean(labels != y, dtype=np.float32), 6)
    return (labels,error_rate)


def train(X_tr: np.ndarray, y_tr: np.ndarray, 
          X_test: np.ndarray, y_test: np.ndarray, 
          nn: NN, n_epochs: int) -> tuple[list[float], list[float]]:
    '''
    :param X_tr: train data
    :param y_tr: train label
    :param X_te: train data
    :param y_te: train label
    :param nn: neural network class
    :param n_epochs: number of epochs to train for
    :return:
    train_losses: Training losses *after* each training epoch
    test_losses: Test losses *after* each training epoch
    '''
    train_losses = 0
    test_losses =0
    for epoch in range(n_epochs):
        X,y = shuffle(X_tr,y_tr,epoch)
        test_loss = 0
        train_loss =0
        for i in range(len(X)):
            y_hat_train= nn.forward(X[i])
            nn.backward(y[i],y_hat_train)
            nn.step()
       
        #test loop

        for i in  range(len(X_test)):
            y_hat_test= nn.forward(X_test[i])
            test_loss+=cross_entropy(y_test[i],y_hat_test)  


        for i in  range(len(X_tr)):
            y_hat_train= nn.forward(X_tr[i])
            train_loss+=cross_entropy(y_tr[i],y_hat_train)

        test_loss = float(test_loss)/len(X_test)
        train_loss=float(train_loss)/len(X_tr)    
        train_losses = train_loss
        test_losses  = test_loss
    return train_losses,test_losses

def cross_entropy_curves(X_tr, y_tr, X_test, y_test,n_epochs):
    
    train_loss_vector = []
    val_loss_vector = []
    hidden_unit_vector = [5,20,50,100,200]
    for n_hid in hidden_unit_vector:
        nn = NN(X_tr.shape[1], n_hid, 100,random_init, 0.001)
        train_losses, val_losses = train(X_tr, y_tr, X_test, y_test, nn, n_epochs)
        train_loss_vector.append(train_losses)
        val_loss_vector.append(val_losses) 
        
        
    plt.plot(hidden_unit_vector,train_loss_vector ,ms =2,ls="solid",c ="blue",marker = "x")
    plt.plot(hidden_unit_vector,val_loss_vector,ms=2,ls="solid",c="red",marker = "x")
    plt.legend(["Average Training Cross Entropy","Average Validation Cross Entropy"])
    plt.title("Avg. Train and Validation Cross-Entropy Loss vs No of Hidden Units")
    plt.xlabel(" No of hidden Units")
    plt.ylabel("Average Training Cross Entropy")
    plt.savefig("Train_val_Losses_Final.png")
    
    

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)




    if (init_flag ==1):
        nn = NN(X_tr.shape[1], n_hid, 10,random_init, lr)
    else:
        nn = NN(X_tr.shape[1], n_hid, 10,zero_init, lr)
    
    cross_entropy_curves(X_tr, y_tr, X_test, y_test,n_epochs)
    # train model 

    train_losses, test_losses = train(X_tr, y_tr, X_test, y_test, nn, n_epochs)
    train_labels, train_error_rate = test(X_tr, y_tr, nn)
    test_labels, test_error_rate = test(X_test, y_test, nn)

    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(
                cur_epoch, cur_tr_loss))
            f.write("epoch={} crossentropy(validation): {}\n".format(
                cur_epoch, cur_te_loss))
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))

