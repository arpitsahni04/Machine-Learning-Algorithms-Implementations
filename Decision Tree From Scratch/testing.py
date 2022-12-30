import numpy as np
import csv

def Mutual_Information_tester(mutual_information):
    '''
    Method to test out functions
    '''
    # testing mutual information
    x = np.array([0,0,1,0,0,1,1,0])
    y = np.array([1,0,0,1,1,0,0,1])
    ans =  mutual_information(y,x)
    print("Mutual Information: ", ans)
    
def read_dataset_tester(read_dataset,dataset):
    data = read_dataset(dataset,exclude_header = False)
    print("\ndata\n",type(data))
    
    # test read dataset with and without features,
    
def best_split_tester(Decision_Tree,max_depth,data):
    classifer = Decision_Tree(max_depth)
    feature = classifer.best_split(data)
    print(feature)
    

def np_loadtxt_function(data):
    dataset=  np.loadtxt(data,skiprows=1)
    print(dataset)
    with open(data,"r") as file:
        tsv_file  = csv.reader(file,delimiter = "\t")
        data = list(tsv_file)
        features = data[0]
    return dataset,features


