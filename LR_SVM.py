
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

def naive_bayes_learn(train_vectors):
    likelihoods = []
    priors = [0.0, 0.0]
    train_vector_length = len(train_vectors[0][1])

    for vector in train_vectors:
        priors[ vector[0] ] += 1

    summed = sum(priors)
    priors[0] = priors[0] / summed
    priors[1] = priors[1] / summed

    for i in range(train_vector_length):
        likelihood = [[1.0, 1.0], [1.0, 1.0]] # class, value
        #uncomment for 0 1 2
        #likelihood = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] # class, value
        #above line of code is for 0 1 2

        for vector in train_vectors:
            likelihood[vector[0]][vector[1][i]] += 1 

        summed1 = sum(likelihood[0]) 
        summed2 = sum(likelihood[1])

        likelihood[0][0] = likelihood[0][0] / summed1 
        likelihood[0][1] = likelihood[0][1] / summed1 
        likelihood[1][0] = likelihood[1][0] / summed2 
        likelihood[1][1] = likelihood[1][1] / summed2 
         #uncomment for 0 1 2
#        likelihood[0][2] = likelihood[0][2] / summed1
#        likelihood[1][2] = likelihood[1][2] / summed2
#         #above block of code is for 0 1 2
        
        likelihoods.append(likelihood)
        
    return priors, likelihoods

def naive_bayes_classify(priors, likelihoods, vector):
    posterior = copy.deepcopy(priors)
    for index in range(len(vector)):
        for i in range(2): #class 0 or 1
            posterior[i] = posterior[i] * likelihoods[index][i][vector[index]]
    return posterior.index(max(posterior))

def naive_bayes_train_test(train_vectors, test_vectors):
    mistake = 0.0

    priors, likelihoods = naive_bayes_learn(train_vectors)
    
    for vector in test_vectors:
        classified = naive_bayes_classify(priors, likelihoods, vector[1])
        if classified != vector[0]:
            mistake += 1
        
    return mistake / len(test_vectors)

def DotProduct(a, b):
    dotProduct = 0
    if(len(a) != len(b)):
        print("Vectors are not of the same length!")
    else:
        for i in range(len(a)):
            temp = a[i] * b[i]
            dotProduct += temp
    
    return dotProduct

def sigmoid(hx):
    sig = 1/(1+ (math.exp(-hx)))
    return sig


def logisticRegression_learn(train_vectors, num_features, learn_Rate):
    maxiter = 100
    weightVector = []
    Log_train_vectors =[]
    True_labels = []
    
    for i in range(len(train_vectors)):
        Log_train_vectors.append(train_vectors[i][1])
        True_labels.append(train_vectors[i][0])
    #print(True_labels)
    #dot_product = 0
    
    #adding a bias term to each training vector
    for eachSample in Log_train_vectors:
        eachSample.append(1)    
    
    for i in range(num_features + 1):
        weightVector.append(0)
    # adding a bias term
    
    
    for iter in range(maxiter):
        errorVector = []
        diffVector = []
        regConst = 0.01
        #for j in range(0, num_features+1):
        
        i=0
        for eachVector in Log_train_vectors:
            sq_err = []
            Iter_dotProduct = 0
            
            #computing the dot product
            Iter_dotProduct = numpy.dot(eachVector, weightVector)
                
            #computing sigmoid
            hx = sigmoid(Iter_dotProduct)
            
            error = (True_labels[i] - hx)
            errorVector.append(error)
            i += 1
            
        for j in range(0, num_features+1):
            
            Update_weightVector = 0
            Gradient_sum = 0
            
            for i in range(len(Log_train_vectors)):
                Gradient_sum = Gradient_sum + (errorVector[i] * Log_train_vectors[i][j])
            
            wt = weightVector[j]
            Update_weightVector = wt + (learn_Rate * (Gradient_sum - (regConst*wt)))
            
            diffVector.append((Update_weightVector - wt))
            #print(type(weightVector))    
            weightVector[j] = Update_weightVector
        
        sqList = numpy.square(diffVector)
        EuclidDist = numpy.sqrt((sum(sqList)))
        if(EuclidDist <= (1E-6)):
            print("halting the updates as updates in weights is too small")   
            return weightVector
        mse = sum(numpy.square(errorVector))
        #print(mse)
    return weightVector
    
def logisticRegression_Predict(test_vectors, FinalWt_Vector):
    Log_test_vectors =[]
    True_test_labels = []
    YPredict = []
    
    for i in range(len(test_vectors)):
        Log_test_vectors.append(test_vectors[i][1])
        True_test_labels.append(test_vectors[i][0])
    
    for eachSample in Log_test_vectors:
        eachSample.append(1)    
    
    for eachSample in Log_test_vectors:
        hx = numpy.dot(eachSample, FinalWt_Vector)
        probability = sigmoid(hx)
        if(probability >= 0.5):
            YPredict.append(1)
        else:
            YPredict.append(0)
    
    return YPredict, True_test_labels
    
def ZeroOneLoss(PredictedLabels, ActualLabels):
    sum = 0
    Zero_One_Loss = 0
    for i in range(0, len(PredictedLabels)):
        if (PredictedLabels[i] != ActualLabels[i]):
            sum = sum + 1
    
    Zero_One_Loss = float(sum)/(len(PredictedLabels))
    #print(Zero_One_Loss)
    return Zero_One_Loss
    

def logisticRegression_train_test(train_vectors, test_vectors, num_features):
    
    learn_Rate = 0.01
    finalWt_vector = logisticRegression_learn(train_vectors, num_features, learn_Rate)
    PredictedLabels, ActualLabels = logisticRegression_Predict(test_vectors, finalWt_vector)
    
    Zero_One_Loss = ZeroOneLoss(PredictedLabels, ActualLabels)
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
    
def main():
    if len(sys.argv) == 4:
        train_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        modelIdx = sys.argv[3]

        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)

        common_words = get_most_commons(train_data, skip=100, total=4000)
        num_features = len(common_words) 
        
        train_vectors = generate_vectors(train_data, common_words)
        test_vectors = generate_vectors(test_data, common_words)
        
        #zero_one_loss = naive_bayes_train_test(train_vectors, test_vectors)
        if int(modelIdx) == 1:
            Logistic_Zr_Loss = logisticRegression_train_test(train_vectors, test_vectors, num_features)
            print 'ZERO-ONE-LOSS-LR ' + str(Logistic_Zr_Loss)
        else:
            if int(modelIdx) == 2:
                SVM_Zr_Loss = SVM_train_test(train_vectors, test_vectors, num_features)
                print 'ZERO-ONE-LOSS-SVM ' + str(SVM_Zr_Loss)
        
        #CrossFold_Validation(train_data)
        
    else:
        print 'usage: python nbc.py train.csv test.csv int(modelIdx)'
        print 'exiting...'

def test(train_data_file, test_data_file):        
        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)

        top_ten = get_most_commons(train_data, skip=100, total=10)
        for i in range(len(top_ten)):
             print 'WORD' + str(i+1) +' '+ top_ten[i]

        common_words = get_most_commons(train_data, skip=100, total=4000)
        num_features = len(common_words)
        
        train_vectors = generate_vectors(train_data, common_words)
        test_vectors = generate_vectors(test_data, common_words)

        #zero_one_loss = naive_bayes_train_test(train_vectors, test_vectors)
        
        #Logistic_Zr_Loss = logisticRegression_train_test(train_vectors, test_vectors, num_features)
        #print 'Logistic ZERO-ONE-LOSS ' + str(Logistic_Zr_Loss)
        #print 'ZERO-ONE-LOSS ' + str(zero_one_loss)
        #SVM_train_test(train_vectors, test_vectors, num_features)
        #CrossFold_Validation(train_data)
        
def CrossFold_Validation(train_data):
    
    DisjointSetS = []
    C_train_data = train_data
    random.shuffle(C_train_data)
    
    LogisticZrOn_tss = numpy.zeros(shape = (6, 2))
    NBC_Zr_Loss_tss = numpy.zeros(shape = (6, 2))
    SVM_Zr_Loss_tss = numpy.zeros(shape = (6, 2))
    
    for i in range(10):
        DisjointSetS.append(C_train_data[(i* len(C_train_data)/10) : ((i+1)* len(C_train_data)/10)])
        print("size range for disjoint sets")
        print((i* len(C_train_data)/10))
        print((i+1)* len(C_train_data)/10)
    print(len(DisjointSetS))
    j = 0
    for d in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]:
        LogisticZrOn_List = numpy.zeros(10)
        NBC_Zr_Loss_List = numpy.zeros(10)
        SVM_Zr_Loss_List = numpy.zeros(10)
        print("printing lenght of train data original")
        print(len(train_data))
        for i in range(10):
            
            Cross_testData = DisjointSetS[i]
            Dump_trainData = [x for x in C_train_data if x not in DisjointSetS[i]]
            #import pdb; pdb.set_trace()
            Cross_trainData = random.sample(Dump_trainData, int(d * len(C_train_data)))
            print(int(d * len(C_train_data)))
            common_words = get_most_commons(Cross_trainData, skip=100, total=4000)
            num_features = len(common_words)

        
            train_vectors = generate_vectors(Cross_trainData, common_words)
            test_vectors = generate_vectors(DisjointSetS[i], common_words)
            #import pdb; pdb.set_trace()
#            NBC_Zr_Loss = naive_bayes_train_test(train_vectors, test_vectors)
#            NBC_Zr_Loss_List[i] = NBC_Zr_Loss
            
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
        #fname = "NBC-TSS.csv"
        
        j += 1
#    with open(fname,"w") as fp:
#        content = ",".join(SVM_Zr_Loss_tss)
#        fp.write(content)
    
        
    
#    print(NBC_Zr_Loss_tss)
    #print(LogisticZrOn_tss)
    #print(SVM_Zr_Loss_tss)

main()