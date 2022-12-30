import sys
import csv
import numpy as np
import inspection
import matplotlib.pyplot as plt
import testing
class Node:

    def __init__(self,depth=0,data=None,vote=None,attr=None,leftHalfData =None,rightHalfData =None):
        self.left = None
        self.right = None
        self.attr = attr # feature we have split on for current node 
        # index of feature we split on
        self.data = data # subset of data at given node
        self.depth = depth # depth of node 
        self.vote = vote # only on leaf node. exists only when self.type is leaf
        self.leftHalfData =leftHalfData
        self.rightHalfData = rightHalfData
        # # HAS TO BE STR
        # self.type = type

def mutual_information(Y,Xd):
    '''
    Calculates I(Y;Xd)
    '''
    # print("y\n")
    H_y = inspection.entropy(Y) 
    # print("H_y: ",H_y)
    values, counts = np.unique(Xd,return_counts = True)
    # print("values",values,"\ncounts",counts)
    total_entropy = 0
    for i in range(len(values)):
        total_entropy+=((counts[i]/float(len(Xd)))*inspection.entropy(Y[Xd==values[i]]))
        # print("Current Total Entropy: ",total_entropy)
    Iyx = H_y - total_entropy
    
    return Iyx

class Decision_Tree:
    def __init__(self,max_depth):
        self.max_depth = int(max_depth)
    
    def predict(self,node,example,features):
        
        '''
        walk from root till any leaf node based on given testing_example xd_dash
        if cur_node is internal or not None:
        go down according to cur_node's xj feature
        else return label stored in cur_node
        cur node can be leaf or internal node. if leaf, it stores label. 
        if internal it stores attribute and threshold
        '''
        if node ==None:
            return 
        if node.vote != None:
            return node.vote
        # print("features",features)
        feature_index = features.index(str(node.attr))
        if example[feature_index] == 0:
            return self.predict(node.left,example,features)
        return self.predict(node.right,example,features)
    
    
    
    def predict_labels(self,dataset,root,features):
        predictions_test = np.zeros(len(dataset))
        # print("predict_labels features :",features)
        for i in range(len(dataset)): ###
            row = dataset[i,:]
            predictions_test[i] = self.predict(root,row,features[:])
        return predictions_test
    
    
    
    def identical_features(self,data):
        container  = set()
        for row in data:
            row_wise_common = "".join(str(int(ele)) for ele in row)
            container.add(row_wise_common)
        if len(container) == 1:
            return True
        return False
    
    
    def train(self,train_input,features):
        '''
        returns the root of the decision tree we build
        '''
        root = self.tree_recurse(train_input,0,features)
        
        return root
    
    def tree_recurse(self,dataset_subset,depth,features):
        # print(dataset_subset.shape)
        # print(features)
        node = Node(depth=depth,data=dataset_subset)
        # print("node.data: \n",node.data)
        label =  dataset_subset[:,-1]                               #doublecheck
        if len(dataset_subset) == 0 or inspection.entropy(label)==0  or self.identical_features(dataset_subset[:,:-1]) or int(self.max_depth) == 0 \
            or node.data.ndim ==1 or int(node.depth) >=int(self.max_depth) : # to handle max_depth > attributes
            #entropy to check if all labels in d' are same
            node.vote = str(inspection.majority_vote(node.data[:,-1]))
            # node.typ ="leaf"
            return node
        else:            
            # for columns in D, column with greatest mutual info gain  = Xd 
            best_feature,max_infoGain= self.best_split(node.data,features)
            node.attr = best_feature
            if max_infoGain<=0:
                node.vote = str(inspection.majority_vote(node.data[:,-1]))
                return node

            feature_index = features.index(str(node.attr))

            # split dataset based on unique values in feature index column
            features.pop(feature_index)

            left_data = np.copy(node.data[np.where(node.data[:,feature_index]==0)])
            left_data=  np.delete(left_data,feature_index,1)
            # print("left_data :",left_data)
            
            right_data = np.copy(node.data[np.where(node.data[:,feature_index]==1)])
            right_data=  np.delete(right_data,feature_index,1)
            # print("right_data :",right_data)
            
            node.left = self.tree_recurse(left_data,depth+1,features[:])
            node.leftHalfData = left_data
            node.right = self.tree_recurse(right_data,depth+1,features[:])
            node.rightHalfData =  right_data

        return node
        
    
    def best_split(self,data,features):
        best_feature =None
        max_infoGain = -1 # info gain can never be -ve so any value is greater
        label =  data[:,-1]

        best_feature_index = None

        for i in range(len(features)-1): 

            info_gain = mutual_information(label,data[:,i])
            # print("info_gain:",info_gain)
            # tie breaker
            if info_gain > max_infoGain:
                max_infoGain =  info_gain
                best_feature = features[i]
                best_feature_index = i

        return best_feature,max_infoGain
    
    def compute_train_test_errors(self,train_input,train_out,test_input,test_out,metrics_out,features,depth):
        # print("features before : ",features)
        root = self.train(train_input,features[:])

        predictions_test = self.predict_labels(test_input,root,features[:])
        predictions_train = self.predict_labels(train_input,root,features[:])
        
        train_labels = train_input[:,-1].astype('int32')
        test_labels  = test_input[:,-1].astype('int32')
        
        error_train =np.mean(predictions_train != train_labels,dtype=np.float32) # need to improve precison
        error_test  =np.mean(predictions_test  != test_labels,dtype=np.float32)
        
        with open(metrics_out,"w") as file:
            file.write("error(train): " + str(error_train)+ "\n")
            file.write("error(test): " + str(error_test)+ "\n")   
            
        with open(test_out,"w") as file:
            for prediction in predictions_test:
                file.writelines(str(prediction) + "\n")
        with open(train_out,"w") as file:
            for prediction in predictions_train:
                file.writelines(str(prediction) + "\n")
        
        ans = {"error_train":error_train,"error_test":error_test,"depth":depth}
        
        return ans
    def pretty_print_tree(self,root,features):
        if root.attr == None:
            return 
        level_dash = "| " +"| "* root.depth 

        feature_index = features.index(str(root.attr))
        dict_left=  {1:0,0:0} 
        dict_right=  {1:0,0:0} 
        values_left, counts_left = np.unique(root.leftHalfData[:,-1],return_counts = True)
        values_right, counts_right = np.unique(root.rightHalfData[:,-1],return_counts = True)
        for i in range(len(values_left)):
            if values_left[i] in dict_left:
                dict_left[values_left[i]]+=counts_left[i]
                
        for i in range(len(values_right)):
            if values_right[i] in dict_right:
                dict_right[values_right[i]]+=counts_right[i]
            

        
        head_left = "["+str(dict_left[0]) +"  "+str(0)+"/"+ \
            str(dict_left[1])+"  "+str(1) +"]"
        
        
        head_right = "["+str(dict_right[0]) +"  "+str(0)+"/"+ \
            str(dict_right[1])+"  "+str(1) +"]"
            

        print(level_dash+ str(root.attr) +" = 0:"+"\t    "+ head_left)
        self.pretty_print_tree(root.left,features[:])
        print(level_dash+ str(root.attr) +" = 1:\t  "+ head_right)
        self.pretty_print_tree(root.right,features[:])

    
    def pretty_print(self,root,features):
        # pass head less data
        values, counts = np.unique(root.data[:,-1],return_counts = True)
        head = "["+str(counts[0]) +" "+str((int(values[0])))+"/"+ \
                str(counts[1])+" "+str(int(values[1])) +"]"
        print(head)
        self.pretty_print_tree(root,features[:])
        # pre-order dfs, each row should correspond to a node in the tree
        
    def pre_order(self,root):
        if root.vote != None:

            print("vote: ",root.vote) 
            return    
        print("name: ",root.attr)
        self.pre_order(root.left)
        self.pre_order(root.right)
        

        
if __name__ == '__main__':
    args = sys.argv
    print(len(args))
    # assert(len(args) == 7)
    train_input = args[1]
    test_input = args[2]
    max_depth = args[3]
    train_out = args[4]
    test_out = args[5]
    metrics_out = args[6]
    
    # sending along with header
    dataset_train=  inspection.read_dataset(train_input)
    dataset_test=  inspection.read_dataset(test_input)
    
    train_input_numpy = dataset_train["data"]
    features_train = dataset_train["features"]
    test_input_numpy = dataset_test["data"]
    

    # train_error_array= [0.49,0.215,0.215,0.14,0.125,0.085,0.08,0.07]
    # test_error_array = [0.40206185,0.2783505,0.3298969,0.17525773,0.25773194,0.25773194,0.24742268,0.25773194]
    # depth_axis=  [ i for i in range(8)]
    # plt.plot(depth_axis,train_error_array,marker ="x",mec = "navy",ms =8,ls="solid",c ="navy")
    # plt.plot(depth_axis,test_error_array,marker ="x",mec="red",ms=8,ls="solid",c="red")
    # plt.legend(["train error","test error"])
    # plt.xlabel("Depth (No of edges from leaf to node) ")
    # plt.ylabel("Error %")
    classifier = Decision_Tree(max_depth)
    root = classifier.train(train_input_numpy,features_train[:])
    classifier.pretty_print(root,features_train[:])
    # # classifier.pre_order(root)
    testing.Mutual_Information_tester(mutual_information)
    
    