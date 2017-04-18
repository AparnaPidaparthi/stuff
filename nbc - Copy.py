# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 18:45:10 2017

@author: aparn
"""
import operator
import string, random
import csv
import sys
import re
from collections import defaultdict

text_yelp = open(sys.argv[1], "r")
#print("hello1")
#text_yelp
labelList = []
reviewList = []
words = []
Num_features = 500 
NumWordsToRemove = 100
NumBadLabels = 0
NumGoodLabels = 0

lines = text_yelp.readlines()
SampleSize = len(lines)

for line in lines:
    Splitline = line.split('\t')
    labelList.append(Splitline[1])
    Splitline[2] = Splitline[2].translate(None, string.punctuation)
    reviewList.append(Splitline[2].lower())

#print(labelList)
#print(reviewList)
NumGoodLabels = labelList.count('1')
NumBadLabels = labelList.count('0')

ProbGood = NumGoodLabels/SampleSize
ProbBad = NumBadLabels/SampleSize



freq_dict = {}

for review in reviewList:
    wordList = []
    wordList = set(review.split())
    for word in wordList:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
              
#print(freq_dict)

#sorting by descending order of frequency and storing it in a list
sorted_x = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
for i in range(0, 10):
    print ("Word"+str(i+1)+" "+sorted_x[100+i][0])

#removing 100 most frequent words and creating the feature list
sorted_featureList = [x[0] for x in sorted_x[NumWordsToRemove:NumWordsToRemove+(Num_features)]]
#creating the matrix with matching the features against each review
matrix_reviews = []
for review in reviewList:
    words = set(review.split())
    Review_Vector = []
    for i in range(0, len(sorted_featureList)):
        if(sorted_featureList[i] in words):
            Review_Vector.append(1)    
        else :
            Review_Vector.append(0)                     
    matrix_reviews.append(Review_Vector)

#import pdb; pdb.set_trace()
matrixReviews_Labels = []

#to compute the four cpd values for each of the features
YesGoodlabel_list =[]
NoGoodLabel_list =[]
YesBadLabel_list = []
NoBadLabel_list = []

for index in range(0, Num_features):
    YesGoodlabel_list.append(0)
    NoGoodLabel_list.append(0)
    YesBadLabel_list.append(0)
    NoBadLabel_list.append(0)
    
for index in range(0, Num_features):
    for j in range(0, SampleSize):
        eachReview = matrix_reviews[j]
        if(labelList[j] == '1'):
            if(eachReview[index] == 1):
                YesGoodlabel_list[index] += 1
            else:
                NoGoodLabel_list[index] += 1
        else:
            if(eachReview[index] == 1):
                YesBadLabel_list[index] += 1
            else:
                NoBadLabel_list[index] += 1
             
for index in range(0, Num_features):
    YesGoodlabel_list[index] = (float(YesGoodlabel_list[index])+1)/(NumGoodLabels+2)
    NoGoodLabel_list[index] = (float(NoGoodLabel_list[index])+1)/(NumGoodLabels+2)
    YesBadLabel_list[index] = (float(YesBadLabel_list[index])+1)/(NumBadLabels+2)
    NoBadLabel_list[index] = (float(NoBadLabel_list[index])+1)/(NumBadLabels+2)
                   
#import pdb; pdb.set_trace()
    
#Prediction phase
#Convert the given test review vector to binary
Predict_yelp = open(sys.argv[2], "r")
Predict_lines = Predict_yelp.readlines()
PredictSize = len(Predict_lines) 
Predict_labelList = []
Predict_reviewList =[]

#Converting tolowercase, stripping punctuation
for line in Predict_lines:
    Splitline = line.split('\t')
    Predict_labelList.append(Splitline[1])
    Splitline[2] = Splitline[2].translate(None, string.punctuation)
    Predict_reviewList.append(Splitline[2].lower())

for  i in range(0, PredictSize):
    Predict_labelList[i] = int(Predict_labelList[i])
#Splitting words and building a binary vector based on presence of defined features
Predictmatrix_reviews = []
for review in Predict_reviewList:
    words = set(review.split())
    Predict_Review_Vector = []
    for i in range(0, len(sorted_featureList)):
        if(sorted_featureList[i] in words):
            Predict_Review_Vector.append(1)    
        else :
            Predict_Review_Vector.append(0)                     
    Predictmatrix_reviews.append(Predict_Review_Vector)

#creating and initializing variables and lists to store CPDs

ProbGoodList =[]
ProbBadList = []
for i in range(0, PredictSize):
    ProbGoodList.append(0)
    ProbBadList.append(0)

#Multiplying the CPDs    
for i in range(0, PredictSize):
    cpdGood = 1.0
    cpdBad = 1.0
    row = Predictmatrix_reviews[i]
    for index in range(0, Num_features):
        if(row[index] == 0):
            cpdGood = float(NoGoodLabel_list[index] ) * cpdGood
            ProbGoodList[i] = cpdGood
            cpdBad = float(NoBadLabel_list[index]) * cpdBad
            ProbBadList[i] = cpdBad
        else:
            cpdGood = float(YesGoodlabel_list[index]) * cpdGood
            ProbGoodList[i] = cpdGood
            cpdBad = float(YesBadLabel_list[index] )* cpdBad
            ProbBadList[i] = cpdBad
    #print(cpdGood, cpdBad)

#Multiplying with prior and predicting class labels
PredictedClassLabel = []                      
for j in range(0, PredictSize):
    ProbGoodList[j]*ProbGood
    ProbBadList[j]*ProbBad
    if (ProbGoodList[j] > ProbBadList[j]):
        PredictedClassLabel.append(1)
    else:
        PredictedClassLabel.append(0)
        
#ZeroOneLoss
sum = 0
for i in range(0, PredictSize):
    if (Predict_labelList[i] != PredictedClassLabel[i]):
        sum = sum + 1

ZeroOneLoss = float(sum)/PredictSize

print("ZERO-ONE-LOSS"+" "+str(ZeroOneLoss))                 

########q3 and q4
## -*- coding: utf-8 -*-
#"""
#Created on Fri Feb 17 10:37:36 2017
#
#@author: aparn
#"""
#
## -*- coding: utf-8 -*-
#"""
#Created on Sun Feb 12 18:45:10 2017
#
#@author: aparn
#"""
#import operator
#import string
#import csv
#import sys
#import re
#from collections import defaultdict
#import numpy
#
#text_yelp = open(sys.argv[1], "r")
#Predict_yelp = open(sys.argv[2], "r")
#
#lines = text_yelp.readlines()
#DataSize = len(lines)
#NumWordsToRemove = 100
#
##train_lines = lines
##q3 qnd q4
#MeanZrOneLoss = []
#StdDevZrOneLoss = []
#AllZrOneLossData = []
#baselineq3 =[]
#baselineq4 = []
#
##q3
##expList = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
##Num_features = 500
#
##q4
#w = [10, 50, 250, 500, 1000, 4000]
#percent = 0.5
#
##q4
#for Num_features in w:
##q3
##for percent in expList:
#    #to repeat 10 times
#    ZeroOneLossList = []
#    for i in range(0, 10):
#        random.shuffle(lines)
#        train_lines = lines[ : int(percent * DataSize)]
#        test_lines = lines[(int(percent * DataSize)) : ]
#        SampleSize = len(train_lines)
#        
#        
#        labelList = []
#        reviewList = []
#        words = []
#        NumBadLabels = 0
#        NumGoodLabels = 0
#        
#        
#        #for splitting the index, labels and reviews
#        for line in train_lines:
#            Splitline = line.split('\t')
#            labelList.append(Splitline[1])
#            Splitline[2] = Splitline[2].translate(None, string.punctuation)
#            reviewList.append(Splitline[2].lower())
#             
#        #print(labelList)
#        #print(reviewList)
#        NumGoodLabels = labelList.count('1')
#        NumBadLabels = labelList.count('0')
#        
#        ProbGood = NumGoodLabels/SampleSize
#        ProbBad = NumBadLabels/SampleSize
#        
#        if (ProbGood > ProbBad):
#            Actualprob = 1
#        else:
#            Actualprob = 0
#        
#        
#        freq_dict = {}
#        
#        for review in reviewList:
#            wordList = set(review.split())
#            
#            for word in wordList:
#                if word in freq_dict:
#                    freq_dict[word] += 1
#                else:
#                    freq_dict[word] = 1
#                      
#        #print(freq_dict)
#        
#        #sorting by descending order of frequency and storing it in a list
#        sorted_x = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
#        
#        print(sorted_x)
#        for i in range(0, 10):
#            print ("Word"+str(i+1)+" "+sorted_x[100+i][0])
#
#        #removing 100 most frequent words and creating the feature list
#        sorted_featureList = [x[0] for x in sorted_x[NumWordsToRemove:NumWordsToRemove+(Num_features)]]
#         
#        #creating the matrix with matching the features against each review
#        matrix_reviews = []
#        for review in reviewList:
#            words = set(review.split())
#            Review_Vector = []
#            for i in range(0, len(sorted_featureList)):
#                if(sorted_featureList[i] in words):
#                    Review_Vector.append(1)    
#                else :
#                    Review_Vector.append(0)                     
#            matrix_reviews.append(Review_Vector)
#        
#        #import pdb; pdb.set_trace()
#        matrixReviews_Labels = []
#        """    
#        #def appendClassLabel(matrix_reviews, labels)
#        for index in range(0, len(matrix_reviews)):
#            row = matrix_reviews[index]
#            row.append(labelList[index])
#            matrixReviews_Labels.append(row)
#        """    
#        #to compute the four cpd values for each of the features
#        YesGoodlabel_list =[]
#        NoGoodLabel_list =[]
#        YesBadLabel_list = []
#        NoBadLabel_list = []
#        
#        for index in range(0, Num_features):
#            YesGoodlabel_list.append(0)
#            NoGoodLabel_list.append(0)
#            YesBadLabel_list.append(0)
#            NoBadLabel_list.append(0)
#            
#        for index in range(0, Num_features):
#            for j in range(0, SampleSize):
#                eachReview = matrix_reviews[j]
#                if(labelList[j] == '1'):
#                    if(eachReview[index] == 1):
#                        YesGoodlabel_list[index] += 1
#                    else:
#                        NoGoodLabel_list[index] += 1
#                else:
#                    if(eachReview[index] == 1):
#                        YesBadLabel_list[index] += 1
#                    else:
#                        NoBadLabel_list[index] += 1
#                     
#        for index in range(0, Num_features):
#            YesGoodlabel_list[index] = (float(YesGoodlabel_list[index]) + 1)/(NumGoodLabels + 2)
#            NoGoodLabel_list[index] = (float(NoGoodLabel_list[index])+ 1)/(NumGoodLabels + 2)
#            YesBadLabel_list[index] = (float(YesBadLabel_list[index]) + 1)/(NumBadLabels + 2)
#            NoBadLabel_list[index] = (float(NoBadLabel_list[index]) + 1)/(NumBadLabels + 2)
#                           
#        #import pdb; pdb.set_trace()
#            
#        #Prediction phase
#        #Convert the given test review vector to binary
#        
#        Predict_lines = test_lines
#        PredictSize = len(Predict_lines) 
#        Predict_labelList = []
#        Predict_reviewList =[]
#        
#        #Converting tolowercase, stripping punctuation
#        for line in Predict_lines:
#            Splitline = line.split('\t')
#            Predict_labelList.append(Splitline[1])
#            Splitline[2] = Splitline[2].translate(None, string.punctuation)
#            Predict_reviewList.append(Splitline[2].lower())
#        
#        for  i in range(0, PredictSize):
#            Predict_labelList[i] = int(Predict_labelList[i])
#        #Splitting words and building a binary vector based on presence of defined features
#        Predictmatrix_reviews = []
#        for review in Predict_reviewList:
#            words = set(review.split())
#            Predict_Review_Vector = []
#            for i in range(0, len(sorted_featureList)):
#                if(sorted_featureList[i] in words):
#                    Predict_Review_Vector.append(1)    
#                else :
#                    Predict_Review_Vector.append(0)                     
#            Predictmatrix_reviews.append(Predict_Review_Vector)
#        
#        #creating and initializing variables and lists to store CPDs
#        
#        ProbGoodList =[]
#        ProbBadList = []
#        for i in range(0, PredictSize):
#            ProbGoodList.append(0)
#            ProbBadList.append(0)
#        
#        #Multiplying the CPDs    
#        for i in range(0, PredictSize):
#            cpdGood = 1.0
#            cpdBad = 1.0
#            row = Predictmatrix_reviews[i]
#            for index in range(0, Num_features):
#                if(row[index] == 0):
#                    cpdGood = float(NoGoodLabel_list[index] ) * cpdGood
#                    ProbGoodList[i] = cpdGood
#                    cpdBad = float(NoBadLabel_list[index]) * cpdBad
#                    ProbBadList[i] = cpdBad
#                else:
#                    cpdGood = float(YesGoodlabel_list[index]) * cpdGood
#                    ProbGoodList[i] = cpdGood
#                    cpdBad = float(YesBadLabel_list[index] )* cpdBad
#                    ProbBadList[i] = cpdBad
#            #print(cpdGood, cpdBad)
#        
#        #Multiplying with prior and predicting class labels
#        PredictedClassLabel = []                      
#        for j in range(0, PredictSize):
#            ProbGoodList[j]*ProbGood
#            ProbBadList[j]*ProbBad
#            if (ProbGoodList[j] > ProbBadList[j]):
#                PredictedClassLabel.append(1)
#            else:
#                PredictedClassLabel.append(0)
#                
#        # Computing ZeroOneLoss
##        sum = 0
##        for i in range(0, PredictSize):
##            if (Predict_labelList[i] != PredictedClassLabel[i]):
##                sum = sum + 1
##        
#        sum = 0
#        for i in range(0, PredictSize):
#            if (Predict_labelList[i] != Actualprob):
#                sum = sum + 1
#        
#        
#        ZeroOneLoss = float(sum)/PredictSize
#        ZeroOneLossList.append(ZeroOneLoss)
#        print("ZERO-ONE-LOSS"+" "+str(ZeroOneLoss))
#    
#    #q3 and q4    
#    MeanZrOneLoss.append((numpy.mean(ZeroOneLossList)))
#    AllZrOneLossData.append(ZeroOneLossList)
#    StdDevZrOneLoss.append((numpy.std(ZeroOneLossList)))
#    
#
#        
#
#
#
