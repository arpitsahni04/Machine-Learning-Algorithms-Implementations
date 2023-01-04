import argparse
import numpy as np
def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices
t
        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    # parser.add_argument("train_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()
    
    # train_data = list()
    # with open(args.train_data, "r") as f:
    #     examples = f.read().strip().split("\n\n")
    #     for example in examples:
    #         xi = [pair.split("\t") for pair in example.split("\n")]
    #         train_data.append(xi)
    

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        indices_to_tags = {index: tag for index,
                           tag in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices,\
        hmminit, hmmemit, hmmtrans, args.predicted_file, \
            args.metric_file,indices_to_tags

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix

def forwardbackward(seq, loginit, logtrans, logemit,words_to_indices):
    """
    implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
<<<<<<< HEAD
    
=======
>>>>>>> c14cfa96de894faa8ccd8741eddc9ea57c8dfeeb
    """
    L = len(seq) # T
    M = len(loginit) # possible states of y K
    # print("log_init :",loginit)
    # sys.exit()

    # Initialize log_alpha and fill it in
    log_alpha = np.zeros((L,M)) # 3*2
    # Initialize log_beta and fill it in
    log_beta = np.zeros((L,M)) #3*2x    
    # Compute the predicted tags for the sequence
    # alpha
    for t in range(L):
        for j in range(M):
            if t==0:
                log_alpha[t][j]= loginit[j] + \
                logemit[j][words_to_indices[seq[t]]]
                
            else:
                c = float("-inf")
                for k in range(M):
                    c = max(c,log_alpha[t-1][k]+ logtrans[k][j])
                exp_sum = 0
                for k in range(M):
                    exp_sum+=np.exp(log_alpha[t-1][k]+ logtrans[k][j]-c)
                log_alpha[t][j] =  logemit[j][words_to_indices[seq[t]]] \
                +np.log(exp_sum)+c
    
    # beta
    for t in range(L-1,-1,-1):
        for j in range(M):
            if t==L-1:
                log_beta[t][j]= 0
            else:
                c = float("-inf")
                for k in range(M):
                    c = max(c,logemit[k][words_to_indices[seq[t+1]]] 
                            + log_beta[t+1][k] + logtrans[j][k])
                exp_sum = 0
                for k in range(M):
                    exp_sum+=np.exp(logemit[k][words_to_indices[seq[t+1]]]
                                    + log_beta[t+1][k] + logtrans[j][k] -c)
                log_beta[t][j] = c + np.log(exp_sum)

    # Compute the log-probability of the sequence
    # print(np.exp(log_alpha))
    print(np.exp(log_beta))
    alpha_beta = log_alpha + log_beta
    # print("log_alpha :",log_alpha)
    # sys.exit()
    predicted_tags = alpha_beta.argmax(1) # over rows to select C or D  
    print(predicted_tags)
    
    c = np.amax(log_alpha,axis =1)[-1]
    exp_sum = 0
    
    for log_alpha_kt in log_alpha[-1]:
        exp_sum+=np.exp(log_alpha_kt-c)
    log_prob =np.log(exp_sum) + c
    # print("log_prob :",log_prob)
    # Return the predicted tags and the log-probability
    return predicted_tags,log_prob

    

def write_metrics(log_LL,avg_log_LL_train,val_accuracy):
        with open(metric_file,"w") as file:
            file.write("Average Log-Likelihood: " + str(log_LL)+ "\n")
            file.write("Average Log-Likelihood_train: " + str(avg_log_LL_train)+ "\n")
            file.write("Accuracy: " + str(val_accuracy)+ "\n")   


if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, \
    hmminit, hmmemit, hmmtrans, predicted_file, metric_file, \
    indices_to_tags= get_inputs()
    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    log_LL = []
    val_accuracy = []
    with open(predicted_file,"w") as ofile:
        for sent in  validation_data:
            sequence = []
            ground_truth_tags = []
            for seq_x in sent:
                # for X,Y pair at a timestep
                sequence.append(seq_x[0])
                ground_truth_tags.append(seq_x[1])
            # print(sequence)
            # sys.exit()
            pred_tag,log_probability = forwardbackward(sequence,np.log(hmminit),
                                                        np.log(hmmtrans),np.log(hmmemit),
                                                        words_to_indices)
            print(log_probability)
            # print("pred_tag " ,pred_tag)
            log_LL.append(log_probability) # append log probability of each seq
            for word_i in range(len(sequence)):
                prediction = indices_to_tags[pred_tag[word_i]]
                if prediction == ground_truth_tags[word_i]:
                    val_accuracy.append(1)
                else:
                    val_accuracy.append(0)
                ofile.write(sequence[word_i]+"\t"+ prediction+"\n")
            # print("predicted_tags :",predicted_tags)
            ofile.write("\n")
    
    
    # log_LL_train = []
    # for sent in  train_data:
    #     sequence = []
    #     ground_truth_tags = []
    #     for seq_x in sent:
    #         # for X,Y pair at a timestep
    #         sequence.append(seq_x[0])
    #         ground_truth_tags.append(seq_x[1])
    #     # print(sequence)
    #     # sys.exit()
    #     pred_tag,log_probability = forwardbackward(sequence,np.log(hmminit),
    #                                                 np.log(hmmtrans),np.log(hmmemit),
    #                                                 words_to_indices)
    #     # print("pred_tag " ,pred_tag)
    #     log_LL_train.append(log_probability) # append log probability of each seq
    
    # Compute the average log-likelihood and the accuracy. The average log-likelihood 

    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    avg_log_LL = sum(log_LL)/len(log_LL)   
    avg_log_LL_train = sum(log_LL_train)/len(log_LL_train)   
    val_accuracy  = sum(val_accuracy)/len(val_accuracy)
    write_metrics(avg_log_LL,avg_log_LL_train,val_accuracy)

    
    
