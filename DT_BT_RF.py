
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:26:51 2017

@author: aparn
"""

import sys
import string, random
import copy
import numpy
import math
from collections import Counter
from operator import itemgetter
from collections import Counter

def process_str(s):
    return s.translate(None, string.punctuation).lower().split()

# dataset format:
# list of (class_label, set of words)

# this function is for 0 1 2
#def generate_vectors(dataset, common_words):
#    d = {}
#    for i in range(len(common_words)):
#        d[common_words[i]] = i
#    
#    vectors = []
#    for item in dataset:
#        vector = [0] * len(common_words)
#        for word in item[1]:
#            if word in d:
#                if (vector[d[word]] > 0):
#                    vector[d[word]] = 2
#                else:
#                    vector[d[word]] = 1
#
#        vectors.append( (item[0], vector) )
#
#    return vectors


def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
#           dataset.append( (int(class_label), list(words)) )#uncomment for 0 1 2
            dataset.append( (int(class_label), set(words)) )#comment for 0 1 2
    return dataset


def get_most_commons(dataset, skip=100, total=100):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

# the length of the common words will be the
# length of the vectors
def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i
    
    vectors = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append( (item[0], vector) )

    return vectors



def DotProduct(a, b):
    dotProduct = 0
    if(len(a) != len(b)):
        print("Vectors are not of the same length!")
    else:
        for i in range(len(a)):
            temp = a[i] * b[i]
            dotProduct += temp
    
    return dotProduct


    
def ZeroOneLoss(PredictedLabels, ActualLabels):
    sum = 0
    Zero_One_Loss = 0
    for i in range(0, len(PredictedLabels)):
        if (PredictedLabels[i] != ActualLabels[i]):
            sum = sum + 1
    
    Zero_One_Loss = float(sum)/(len(PredictedLabels))
    #print(Zero_One_Loss)
    return Zero_One_Loss
    
    
def SVM_learn(train_vectors, num_features, learn_Rate):
    #declaring the required variables and constants
    maxiter = 100
    regConst_lambda = 0.01
    learn_rate = 0.5

    svm_train_vectors = []
    true_labels = numpy.zeros(len(train_vectors))
    currentWts = numpy.zeros(num_features + 1)

    #loop to extratct feature vectors and modify labels to -1 and 1
    for i in range(len(train_vectors)):
        svm_train_vectors.append(train_vectors[i][1])
        if(train_vectors[i][0] == 0):
            true_labels[i] = -1
        else:
            true_labels[i] = 1
    
    #loop to append bias term to each feature vector
    for each_review in svm_train_vectors:             
        each_review.append(1)  #bias term
    #loop to update weight vector- each iteration updates all the weights once
    for iter in range(maxiter):
        updatedWts = numpy.zeros(len(currentWts))
        gradient = numpy.zeros(len(currentWts))
        i = 0

        #loop to go through all the training samples 
        #and compute the gradient by which weights should be updated
        for each_review in svm_train_vectors:
            try:
                if((true_labels[i] * numpy.dot(each_review, currentWts)) >= 1):
                    gradient += (regConst_lambda * currentWts)
                    #print "OK"
                else:
                    gradient += (regConst_lambda * currentWts) - (true_labels[i] * numpy.array(each_review))
            except:
                import pdb; pdb.set_trace()
            i += 1

        updatedWts = currentWts - (learn_rate * gradient / len(true_labels))

        #compare updated and current weights
        euclidian_dist = (numpy.sqrt(sum(numpy.square(updatedWts - currentWts))))
        if((euclidian_dist < 1E-6)):
            return updatedWts
        else:
            currentWts = updatedWts

    return updatedWts

def SVM_Predict(test_vectors, FinalWt_Vector):
    svm_test_vectors =[]
    True_test_labels = []
    YPredict = []
    
    for i in range(len(test_vectors)):
        True_test_labels.append(test_vectors[i][0])
        if(test_vectors[i][0] == 0):
            True_test_labels[i] = -1
        else:
            True_test_labels[i] = 1
    
    for i in range(len(test_vectors)):
        svm_test_vectors.append(test_vectors[i][1])
        #True_test_labels.append(test_vectors[i][0])
    
    for eachSample in svm_test_vectors:
        eachSample.append(1)    
    #import pdb; pdb.set_trace()
    for eachSample in svm_test_vectors:
        hx = numpy.dot(eachSample, FinalWt_Vector)
        if(hx >= 0):
            YPredict.append(1)
        else:
            YPredict.append(-1)
    
    return YPredict, True_test_labels
    
def SVM_train_test(train_vectors, test_vectors, num_features):
    
    learn_Rate = 0.5
    
    finalWt_vector = SVM_learn(train_vectors, num_features, learn_Rate)
    PredictedLabels, true_labels_test = SVM_Predict(test_vectors, finalWt_vector)
    #import pdb; pdb.set_trace()
    Zero_One_Loss = ZeroOneLoss(PredictedLabels, true_labels_test)
    #print 'SVM Zr One Loss' + str(Zero_One_Loss)
    return Zero_One_Loss

class TreeNode:
    def __init__(self):
        self.featureIndex = None
        self.predictionLeaf = None
        self.leftChild = None
        self.rightChild = None
    
    def predict(self, sample):
        #print 'self.predictionLeaf' + str(self.predictionLeaf)
        if (self.predictionLeaf == None):
            #print("it entered the first if condition")
            if sample[self.featureIndex] == 1:
                #print("recursing to the left")
                return self.leftChild.predict(sample)
            else:
                #print("recursing to the right")
                return self.rightChild.predict(sample)
        else:
            #print 'going to the leaf' + str(self.predictionLeaf)
            return self.predictionLeaf
    #count = 0
#def DT_create_ChildNode

def giniData(anyLabelList):
    
    giniIndex = 1 - ((float(anyLabelList.count(1))/len(anyLabelList))**2) - ((float(anyLabelList.count(0))/len(anyLabelList))**2)

    return giniIndex

def Node_train(Node, reviews, labels, featureIndexList, modelIdx, depth = 1):
    #print 'this is the feature index as soon as the nodeTrain fn is called' + str(Node.featureIndex)
    Gain = []
    giniParent = giniData(labels)
    MAXDEPTH = 11
    P_sample = []
    #print 'depth' + str(depth)
    if(modelIdx == 3):
        P_sample = random.sample(featureIndexList , int(len(featureIndexList)**0.5))
    else:
        P_sample = featureIndexList
    
    for i in list(P_sample):
        childNode_leftReviews = []
        childNode_rightReviews = []
        rightLabelList = []
        leftLabelList = []
        
        j= 0
        for eachReview in reviews:
            if(eachReview[i] == 1):
                childNode_leftReviews.append(eachReview)
                leftLabelList.append(labels[j])
            else:
                childNode_rightReviews.append(eachReview)
                rightLabelList.append(labels[j])
            
            j += 1
        
#        print(len(leftLabelList))
#        print(len(rightLabelList))
        if len(leftLabelList) != 0:
            gini_left = giniData(leftLabelList)
        else:
            gini_left = 0
        if len(rightLabelList) != 0:
            gini_right = giniData(rightLabelList)
        else:
            gini_right = 0
            
        gain_value = giniParent - (float(len(leftLabelList))/len(labels) * gini_left) - (float(len(rightLabelList))/len(labels) * gini_right)
        Gain.append(gain_value)
    #print(Gain)
    max_gain = max(Gain)
    index_maxGain = Gain.index(max_gain)
    
    bestFeature = list(P_sample)[index_maxGain]
    #print 'best feature is' + str(bestFeature)
#flushing the variables    
    childNode_leftReviews = []
    childNode_rightReviews = []
    rightLabelList = []
    leftLabelList = []
#splitting right and left branches of reviews based on best feature
    j= 0
    for eachReview in reviews:
        if(eachReview[bestFeature] == 1):
            childNode_leftReviews.append(eachReview)
            leftLabelList.append(labels[j])
        else:
            childNode_rightReviews.append(eachReview)
            rightLabelList.append(labels[j])
        j += 1
    
    Node.featureIndex = bestFeature
    #print 'I am checking the feature index of the current node' + str(Node.featureIndex)
    Node.leftChild = TreeNode()
    Node.rightChild = TreeNode()
    
    #print(depth + 1)
    if not ((depth+1 > MAXDEPTH) or (len(childNode_leftReviews) < 10)):
        Node_train(Node.leftChild, childNode_leftReviews, leftLabelList, featureIndexList, modelIdx, depth+1)
        #print 'left-IF-depth' + str(depth)

    else:
        #return leaves
        if len(leftLabelList) == 0:
            Node.leftChild.predictionLeaf = 0
        else:
            Node.leftChild.predictionLeaf = Counter(leftLabelList).most_common(1)[0][0]
        #print 'Node.leftChild.predictionLeaf' + str(Node.leftChild.predictionLeaf)
        #print 'left-Else-depth' + str(depth)

    if not ((depth+1 > MAXDEPTH) or (len(childNode_rightReviews) < 10)):
        Node_train(Node.rightChild, childNode_rightReviews, rightLabelList, featureIndexList, modelIdx, depth+1)
        #print 'right-IF-depth' + str(depth)

    else:
        #return leaves
        #tmpVar = 
        if len(rightLabelList) == 0:
            Node.leftChild.predictionLeaf = 1
            
        else:
            Node.rightChild.predictionLeaf = Counter(rightLabelList).most_common(1)[0][0]
            
        #print 'Counter(labels).most_common(1)[0][0]' + str(Counter(labels).most_common(1)[0][0])
        #print 'Node.rightChild.predictionLeaf' + str(Node.rightChild.predictionLeaf)
        #print 'right-Else-depth' + str(depth)

  

def DT_train_test(train_vectors, test_vectors, num_features, modelIdx):
    
    DT_train_vectors = []
    DT_True_labels = []
    DT_test_vectors = []
    DT_test_labels = []
    DT_predicted_labels = []
    F_list = range(0, num_features)
    
    for i in range(len(train_vectors)):
        DT_train_vectors.append(train_vectors[i][1])
        DT_True_labels.append(train_vectors[i][0])
    
    for i in range(len(test_vectors)):
        DT_test_vectors.append(test_vectors[i][1])
        DT_test_labels.append(test_vectors[i][0])
    
    rootNode = TreeNode()
    count = 0
    Node_train(rootNode, DT_train_vectors, DT_True_labels, F_list, modelIdx)
    #print 'rootNode.leftChild.leftChildfeatureIndex' + str(rootNode.leftChild.leftChild.featureIndex)
    
    #print(type(rootNode))
    #print(rootNode.featureIndex)
    #to test/predict
    for eachSample in DT_test_vectors:
        #print(rootNode.predict(eachSample))
        DT_predicted_labels.append(rootNode.predict(eachSample))
    
    #print(DT_predicted_labels)
    DT_ZrOneLoss = ZeroOneLoss(DT_predicted_labels, DT_test_labels)
    return DT_ZrOneLoss
    

def main():
    if len(sys.argv) == 4:
        train_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        modelIdx = int(sys.argv[3])

        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)

        common_words = get_most_commons(train_data, skip=100, total=1000)
        num_features = len(common_words) 
        
        train_vectors = generate_vectors(train_data, common_words)
        test_vectors = generate_vectors(test_data, common_words)
#        train_vectors = random.sample(Train_vectors, 500)
#        test_vectors =  random.sample(Test_vectors, 100)
        #zero_one_loss = naive_bayes_train_test(train_vectors, test_vectors)
#        if int(modelIdx) == 1:
#            Logistic_Zr_Loss = logisticRegression_train_test(train_vectors, test_vectors, num_features)
#            print 'ZERO-ONE-LOSS-LR ' + str(Logistic_Zr_Loss)
#        else:
#            if int(modelIdx) == 2:
#                SVM_Zr_Loss = SVM_train_test(train_vectors, test_vectors, num_features)
#                print 'ZERO-ONE-LOSS-SVM ' + str(SVM_Zr_Loss)
#       



        #CrossFold_Validation(train_data)
        if int(modelIdx) == 1:
            rootNode = TreeNode()
            DecisionTree_Zr_Loss = DT_train_test(train_vectors, test_vectors, num_features, modelIdx)
            print 'ZERO-ONE-LOSS-DT ' + str(DecisionTree_Zr_Loss)
        if int(modelIdx) == 2:
            BaggedTree_ZrOneLoss = train_test_bagging(train_vectors, test_vectors, num_features, modelIdx)
            print 'ZERO-ONE-LOSS-BT ' + str(BaggedTree_ZrOneLoss)
        if int(modelIdx) == 3:
            RandomForest_ZrOneLoss = train_test_bagging(train_vectors, test_vectors, num_features, modelIdx)
            print 'ZERO-ONE-LOSS-RF ' + str(RandomForest_ZrOneLoss)
    else:
        print 'usage: python nbc.py train.csv test.csv int(modelIdx)'
        print 'exiting...'

def test(train_data_file, test_data_file):        
        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)

        top_ten = get_most_commons(train_data, skip=100, total=10)
        for i in range(len(top_ten)):
             print 'WORD' + str(i+1) +' '+ top_ten[i]

        common_words = get_most_commons(train_data, skip=100, total=1000)
        num_features = len(common_words)
        
        train_vectors = generate_vectors(train_data, common_words)
        test_vectors = generate_vectors(test_data, common_words)

        #zero_one_loss = naive_bayes_train_test(train_vectors, test_vectors)
        
        #Logistic_Zr_Loss = logisticRegression_train_test(train_vectors, test_vectors, num_features)
        #print 'Logistic ZERO-ONE-LOSS ' + str(Logistic_Zr_Loss)
        #print 'ZERO-ONE-LOSS ' + str(zero_one_loss)
        #SVM_train_test(train_vectors, test_vectors, num_features)
        #CrossFold_Validation(train_data)

class baggingTrees:
    def __init__(self, num_trees):
        self.rootNodes = []
        for k in range(num_trees):
            self.rootNodes.append(TreeNode())
        
    def Bag_predict(self, sample):
        self.counts_0 = 0
        self.counts_1 = 0
        #print 'len(self.rootNodes)' + str(len(self.rootNodes)) 
        for idx in range(len(self.rootNodes)):
            #print 'root index is' + str(idx)
            pred = self.rootNodes[idx].predict(sample)
            #print 'prediction for one sample by one tree' + str(pred)
            if pred == 1:
                self.counts_1 += 1
            else:
                self.counts_0 += 1
        #print 'count_0=' + str(self.counts_0) + ';counts_1=' + str(self.counts_1)
        if self.counts_0 > self.counts_1:
            return 0;
        else:
            return 1;
                
#        
def train_test_bagging(trainVectors, testVectors, num_features, modelIdx):
    
   bag = baggingTrees(50)
   BT_pred_labels = []
   num_trees = 50
   depth = 1
   F_list = range(0, num_features)
   for k in range(num_trees):
        bagged_train_vectors = []
        
        for i in range(len(trainVectors)):
            bagged_train_vectors.append(random.choice(trainVectors))
        
        BT_train_vectors, BT_labels = preprocessingVectors(bagged_train_vectors)
        #print(bag.rootNodes[k])
        Node_train(bag.rootNodes[k], BT_train_vectors, BT_labels, F_list, modelIdx)
        
   BT_test_vectors, BT_test_labels = preprocessingVectors(testVectors)
        
   for eachSample in BT_test_vectors:
       #print(bag.Bag_predict(eachSample))
       BT_pred_labels.append(bag.Bag_predict(eachSample))
   
   #print(BT_pred_labels)
   BT_ZrOneLoss = ZeroOneLoss(BT_pred_labels, BT_test_labels)
   return BT_ZrOneLoss

#def RandomForest():
    
def preprocessingVectors(vectors):
    
    reviewVectors = []
    labelVectors = []
    for i in range(len(vectors)):
        reviewVectors.append(vectors[i][1])
        labelVectors.append(vectors[i][0])
        
    return reviewVectors, labelVectors     


def CrossFold_Validation(train_data):
    
    DisjointSetS = []
    C_train_data = train_data
    random.shuffle(C_train_data)
    
    #LogisticZrOn_tss = numpy.zeros(shape = (6, 2))
    #NBC_Zr_Loss_tss = numpy.zeros(shape = (6, 2))
    SVM_Zr_Loss_tss = numpy.zeros(shape = (4, 2))
    DT_Zr_Loss = numpy.zeros(shape = (4,2))
    BT_Zr_Loss = numpy.zeros(shape = (4,2))
    RF_Zr_Loss = numpy.zeros(shape = (4,2))
    
    for i in range(10):
        DisjointSetS.append(C_train_data[(i* len(C_train_data)/10) : ((i+1)* len(C_train_data)/10)])
        #print("size range for disjoint sets")
        #print((i* len(C_train_data)/10))
        #print((i+1)* len(C_train_data)/10)
    #print(len(DisjointSetS))
    j = 0
    for d in [0.025, 0.05, 0.125, 0.25]:
    #for features in [200, 500, 1000, 1500]:
    #for depth in [5, 10, 15, 20]:
#        LogisticZrOn_List = numpy.zeros(10)
#        NBC_Zr_Loss_List = numpy.zeros(10)
#        SVM_Zr_Loss_List = numpy.zeros(10)
         DT_Zr_Loss_List = numpy.zeros(10)
         BT_Zr_Loss = numpy.zeros(10)
         RF_Zr_Loss = numpy.zeros(10)
#        print("printing lenght of train data original")
         #print(len(train_data))
         for i in range(10):
            
            Cross_testData = DisjointSetS[i]
            Dump_trainData = [x for x in C_train_data if x not in DisjointSetS[i]]
            #import pdb; pdb.set_trace()
            Cross_trainData = random.sample(Dump_trainData, int(d * len(C_train_data)))
            #print(int(d * len(C_train_data)))
            common_words = get_most_commons(Cross_trainData, skip=100, total=1000)
            num_features = len(common_words)

        
            train_vectors = generate_vectors(Cross_trainData, common_words)
            test_vectors = generate_vectors(DisjointSetS[i], common_words)
            #import pdb; pdb.set_trace()
#            NBC_Zr_Loss = naive_bayes_train_test(train_vectors, test_vectors)
#            NBC_Zr_Loss_List[i] = NBC_Zr_Loss
            DecisionTree_Zr_Loss = DT_train_test(train_vectors, test_vectors, num_features, modelIdx)
            DecisionTree_Zr_Loss[i] = DecisionTree_Zr_Loss
            BaggedTree_ZrOneLoss = train_test_bagging(train_vectors, test_vectors, num_features, modelIdx)
            BaggedTree_ZrOneLoss[i] = BaggedTree_ZrOneLoss
            RandomForest_ZrOneLoss = train_test_bagging(train_vectors, test_vectors, num_features, modelIdx)
            RandomForest_ZrOneLoss[i] = RandomForest_ZrOneLoss
            #Logistic_Zr_Loss = logisticRegression_train_test(train_vectors, test_vectors, num_features)
            #LogisticZrOn_List[i] = (Logistic_Zr_Loss)
            #print 'Logistic ZERO-ONE-LOSS ' + str(Logistic_Zr_Loss)
            #print 'ZERO-ONE-LOSS ' + str(zero_one_loss)
            #print(len(train_vectors[0][1]))
#            print("calling the SVM train test function")
#            print 'lenght of train_vectors' + str(len(train_vectors))
#            print 'lenght of test_vectors' + str(len(test_vectors))
#            print 'number of features in test vector' + str(len(test_vectors[0][1]))
            #SVM_Zr_Loss = SVM_train_test(train_vectors, test_vectors, num_features)
            #SVM_Zr_Loss_List[i] = SVM_Zr_Loss
            #print(len(train_vectors[0][1]))
            #print(SVM_Zr_Loss)
         
        #LogisticZrOn_tss[j][0] = numpy.mean(LogisticZrOn_List)
        #LogisticZrOn_tss[j][1] = numpy.std(LogisticZrOn_List)/(10**0.5)
#        NBC_Zr_Loss_tss[j][0] = numpy.mean(NBC_Zr_Loss_List)
#        NBC_Zr_Loss_tss[j][1] = numpy.std(NBC_Zr_Loss_List)/(10**0.5)
        #SVM_Zr_Loss_tss[j][0] = numpy.mean(SVM_Zr_Loss_List)
        #SVM_Zr_Loss_tss[j][1] = numpy.std(SVM_Zr_Loss_List)/(10**0.5)
        #DT_Zr_Loss[j][0] = numpy.mean(DecisionTree_Zr_Loss)
        #DT_Zr_Loss[j][1] = numpy.std(DecisionTree_Zr_Loss)
        #BT_Zr_Loss[j][0] = numpy.mean(BT_Zr_Loss)
        #BT_Zr_Loss[j][1] = numpy.std(BT_Zr_Loss)
        #RF_Zr_Loss[j][0] = numpy.mean(RF_Zr_Loss)
        #RF_Zr_Loss[j][1] = numpy.std(RF_Zr_Loss)
        
    #fname = "NBC-TSS.csv"
        
         j += 1
#    with open(fname,"w") as fp:
#        content = ",".join(SVM_Zr_Loss_tss)
#        fp.write(content)
    
        
    
#    print(NBC_Zr_Loss_tss)
    #print(LogisticZrOn_tss)
    #print(SVM_Zr_Loss_tss)

main()