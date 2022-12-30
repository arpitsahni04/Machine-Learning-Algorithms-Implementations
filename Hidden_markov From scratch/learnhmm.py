import argparse
import numpy as np
import sys

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_indices, tags_to_indices, \
    hmmprior, hmmemit, hmmtrans = get_inputs()
    # Initialize the initial, emission, and transition matrices
    # train_data = train_data[:10000]
    # print(train_data)
    C = np.zeros((len(tags_to_indices),1))    #initial
    A = np.zeros((len(tags_to_indices),len(words_to_indices))) #emission
    B = np.zeros((len(tags_to_indices),len(tags_to_indices))) #transition
    # Increment the matrices
    for seq in train_data: #
            C[tags_to_indices[seq[0][1]]]+=1
            
    print(train_data[0])
    for seq in range(len(train_data)):#Emision
        for sent in train_data[seq]:
            A[tags_to_indices[sent[1]]][words_to_indices[sent[0]]] +=1
            

    for seq in range(len(train_data)):#Transition
        for i in range(len(train_data[seq])-1):
            B[tags_to_indices[train_data[seq][i][1]]][tags_to_indices[train_data[seq][i+1][1]]] +=1
            
    # Add a pseudocount
    C+=1
    C/=np.sum(C)
    
    
    A+=1
    A /=A.sum(axis=1).reshape(-1,1)

    B+=1
    B /=B.sum(axis=1).reshape(-1,1)
    # Save your matrices to the output files --- the reference solution uses 
    np.savetxt (hmmemit,A,delimiter=" ")
    np.savetxt (hmmtrans,B,delimiter=" ")
    np.savetxt (hmmprior,C,delimiter=" ")

