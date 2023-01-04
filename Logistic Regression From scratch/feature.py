import csv
import numpy as np
import sys

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


def Trim(file, feature_dictionary_input, formatted_train_out):
    raw_data = load_tsv_dataset(file)
    word2vec_map = load_feature_dictionary(feature_dictionary_input)
    features_in_map = VECTOR_LEN

    output_feature_vector = ""
    with open(formatted_train_out, "w") as file:
        for i in range(len(raw_data)):
            label = np.array(float(raw_data[i][0]))
            feature_vector = np.zeros(features_in_map)  # can we hardcode this?
            feature_vector = np.array(
                [word2vec_map[ele] for ele in raw_data[i][1].split() if ele in word2vec_map])
            # print("feature_vector_prior: ",feature_vector.shape)
            if len(feature_vector) != 0:
                feature_vector = np.around(np.mean(feature_vector, axis=0,dtype = np.float32), 6)

            # print("label",label)
            # print("feature_vector_post",feature_vector.shape)

            output_feature_vector = "\t".join(
                str(feature) for feature in feature_vector)
            output_feature_vector = str(
                label) + "\t" + output_feature_vector + "\n"
            file.write(str(output_feature_vector))


if __name__ == '__main__':
    args = sys.argv
    # assert(len(args) == 8)
    train_input = args[1]
    validation_input = args[2]
    test_input = args[3]
    feature_dictionary_input = args[4]
    formatted_train_out = args[5]
    formatted_validation_out = args[6]
    formatted_test_out = args[7]

    # write formatted data
    Trim(train_input, feature_dictionary_input, formatted_train_out)
    Trim(test_input,feature_dictionary_input,formatted_test_out)
    Trim(validation_input,feature_dictionary_input,formatted_validation_out)
