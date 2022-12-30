import numpy as np
import sys
import matplotlib.pyplot as plt

def read_dataset(file):
    data = np.loadtxt(file)
    return data


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(theta, X, y, num_epoch, learning_rate,X_valid,y_valid):
    # TODO: Implement `train` using vectorization
    n,m= X.shape  # m examples=200 , n featues=301 for large dataset
    n_valid,m_valid = X_valid.shape
    cost_vect_train = []
    cost_vect_valid = []
    for i in range(num_epoch):
        cost_valid = 0
        cost_train = 0
        for j in range(n): # loop over examples
            Z = np.dot(X[j], theta)
            A = sigmoid(Z)
            dw = np.expand_dims((A-y[j] ).item()*X[j].T,axis =1)
            # update parameters
            theta -= learning_rate*dw
        # calculate cost after each epoch
        Z_train = np.dot(X, theta)

        cost_train =  ((y_train*Z_train)-np.log(1+np.exp(Z_train)))

        cost_train =(- np.mean(cost_train))
        cost_vect_train.append(float(cost_train))
        
        Z_valid = np.dot(X_valid, theta)
        
        cost_valid =   ((y_valid*Z_valid)-np.log(1+np.exp(Z_valid)))
        cost_valid =(- np.mean(cost_valid))

        cost_vect_valid.append(float(cost_valid))
        
    return theta,cost_vect_train,cost_vect_valid


def predict(theta, X):
    # TODO: Implement `predict` using vectorization
    Z = np.dot(X, theta)
    A = sigmoid(Z)
    A = (A>=0.5).astype(np.int8)
    return A


def compute_error(y_pred, y):
    # TODO: Implement `compute_error` using vectorization
    error = np.around(np.mean(y_pred != y, dtype=np.float32), 6)
    return error

def writePredstofile(file,A):
    with open(file,"w") as file:
        for prediction in A:
            file.writelines(str(prediction.item()) + "\n")
            
def writeMetricstofile(error_train, error_test):
    with open(metrics_out, "w") as file:
        file.write("error(train): " + str(error_train) + "\n")
        file.write("error(test): " + str(error_test) + "\n")

def plot_likelihood(cost_vector_train,cost_vector_valid,epoch):
    epoch_axis=  [ i for i in range(1,epoch+1)]
    plt.plot(epoch_axis,cost_vector_train,ms =2,ls="solid",c ="cyan")
    plt.plot(epoch_axis,cost_vector_valid,ms=2,ls="solid",c="purple")
    plt.legend(["negative Log-Likelihood training","negative Log-Likelihood validation"])
    plt.xlabel("Epochs")
    plt.ylabel("negative Log-Likelihood")
    plt.savefig("liklihood.png")
    # plt.show()
   
   
def compare_learning_rates(theta, X_train, y_train,epoch=1000):
    epoch_axis=  [ i for i in range(epoch)]
    
    theta = np.zeros((X_train.shape[1],1))
    learning_rate = 10**-3
    theta,cost_vector_train_lr1,cost_vector_valid= train(theta, X_train, y_train, num_epoch, learning_rate,X_valid,y_valid)
    theta = np.zeros((X_train.shape[1],1))
    learning_rate = 10**-4
    theta,cost_vector_train_lr2,cost_vector_valid = train(theta, X_train, y_train, num_epoch, learning_rate,X_valid,y_valid)
    theta = np.zeros((X_train.shape[1],1))
    learning_rate = 10**-5
    theta,cost_vector_train_lr3,cost_vector_valid = train(theta, X_train, y_train, num_epoch, learning_rate,X_valid,y_valid)
    print("lr1",cost_vector_train_lr1)
    print("lr1",cost_vector_train_lr2)
    print("lr1",cost_vector_train_lr3)
    sys.exit
    
    plt.plot(epoch_axis,cost_vector_train_lr1 ,ms =2,ls="solid",c ="green")
    plt.plot(epoch_axis,cost_vector_train_lr2,ms=2,ls="solid",c="purple")
    plt.plot(epoch_axis,cost_vector_train_lr3,ms=2,ls="solid",c="cyan")
    plt.legend(["lr1 = 10^-3 ","lr2 = 10^-4","lr3 = 10^-5"])
    plt.xlabel(" Epochs")
    plt.ylabel("negative log-likelihood")
    plt.savefig("learning_rate.png")
    
if __name__ == '__main__':
    args = sys.argv
    assert (len(args) == 9)  # insuffcient number of arguments passed
    formatted_train_input = args[1]
    formatted_validation_input = args[2]
    formatted_test_input = args[3]
    train_out = args[4]
    test_out = args[5]
    metrics_out = args[6]
    num_epoch = int(args[7])
    learning_rate = float(args[8])
    # read the data to a numpy array
    formatted_train_input = read_dataset(formatted_train_input)
    formatted_test_input = read_dataset(formatted_test_input)
    formatted_valid_input = read_dataset(formatted_validation_input)

    X_train = formatted_train_input[:, 1:]
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    y_train = np.expand_dims(formatted_train_input[:, 0],axis=1)
    #test data
    X_test = formatted_test_input[:, 1:]
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_test =  np.expand_dims(formatted_test_input[:, 0],axis=1)
    #validation data
    X_valid = formatted_valid_input[:, 1:]
    X_valid = np.hstack((np.ones((X_valid.shape[0], 1)), X_valid))
    y_valid = np.expand_dims(formatted_valid_input[:, 0],axis=1)

    theta = np.expand_dims(np.zeros(X_train.shape[1]),axis =1)


    theta,cost_vector_train,cost_vector_valid = train(theta, X_train, y_train, num_epoch, learning_rate,X_valid,y_valid)
 
    plot_likelihood(cost_vector_train,cost_vector_valid,num_epoch)

    # compare_learning_rates(theta, X_train, y_train, num_epoch)

    y_pred_train = predict(theta, X_train)
    y_pred_test = predict(theta, X_test)

    writePredstofile(train_out,y_pred_train)
    writePredstofile(test_out,y_pred_test)

    error_train = compute_error(y_pred_train,y_train)
    error_test = compute_error(y_pred_test,y_test)

    writeMetricstofile(error_train,error_test)

