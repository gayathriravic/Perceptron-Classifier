import sys
import json
import collections
import string
from collections import defaultdict
import math
import itertools
import random
import operator

global y1  # for Fake/True
global y2  # for Neg / Pos

weight1 = {}
weight1[''] = {}

weight2 = {}
weight2[''] = {}

featureVector = {}  # feature vector for both the classifiers.
weights1 = {}
weights2 = {}
u1 = {}
u2 = {}

beta1 = 0
beta2 = 0


def readFile():
    file = open(sys.argv[1], 'r', encoding="utf-8")
    return file


biasOne = 0
biasTwo = 0


def calculateActivationFunctionForFirstClass(featureVector, weights, y1):
    #print("Calculating activation function for first class")

    frequencySum = 0
    for features in featureVector:
        frequency = featureVector[features]
        if features in weights1:
                frequencySum += frequency * weights1[features]
    frequencySum += biasOne
    if (y1 * frequencySum <= 0):
        updateWeight1 = 1
        return updateWeight1
    if (y1 * frequencySum > 0):
        updateWeight1 = 0
        return updateWeight1
      
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing",
                     "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
                     "it", "its", "itself", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
                     "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
                     "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"]

def calculateActivationFunctionForSecondClass(featureVector, weights, y2):
    frequencySum = 0
  
    for features in featureVector:
        frequency = featureVector[features]
        if features in weights2:
                frequencySum += frequency * weights2[features]
    frequencySum += biasTwo

    if (y2 * frequencySum <= 0):
        updateWeight2 = 1
        return updateWeight2
    if (y2 * frequencySum > 0):
        updateWeight2 = 0
        return updateWeight2


def updateWeightsForFirstClass(featureVector, y,c):
    global biasOne
    global beta1
    biasOne = biasOne + y
    beta1 = beta1 + y*c
    for features in featureVector:
        if features in weights1:
                weights1[features] = weights1[features] + y * featureVector[features]  # updating weights.
                u1[features] = u1[features] + y*c*featureVector[features]

def updateWeightsForSecondClass(featureVector, y,c):
    global biasTwo
    global beta2
    biasTwo = biasTwo + y
    beta2 = beta2+y*c
    for features in featureVector:
            if features in weights2:
                weights2[features] = weights2[features] + y * featureVector[features]  # updating weights.
                u2[features] = u2[features] + y*c*featureVector[features]


def trainData(file):
    c = 1
    for line in file:
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
        lines = line.translate(translator)
        line = lines.strip("\n").split(" ")
        for words in line:
            if (words != line[0] and words != line[1] and words != line[2] and words!=""):
                weights1[words.lower()] = 0
                weights2[words.lower()] = 0
                u1[words.lower()] = 0
                u2[words.lower()] = 0
    file.seek(0)
    iterations = 30
    file1 = file
    c = 0
    while(iterations):
        iterations -=1
        file1.seek(0)
        file = file1
        file = file.readlines()
        random.shuffle(file) 
        c += 1
        for line in file:
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
            lines = line.translate(translator)
            line = lines.strip("\n").split(" ")
            featureVector = {}
            for words in line:
                #words = words
                if words.lower() != line[0].lower():  #skip the keyword.
                    if words.lower() == line[1].lower(): #first label
                        if words == "Fake":
                            y1 = -1
                        else:
                            y1 = 1  # True
                    elif words.lower() == line[2].lower(): #second label
                        if words == "Neg":
                            y2 = -1
                        else:
                            y2 = 1  # Positive
                    else:  # the rest of the corpus
                        if words != "" and words.lower() not in stopwords:
                            if words.lower() in featureVector:
                                words = words.lower()
                                featureVector[words] += 1
                            else:
                                words = words.lower()
                                featureVector[words] = 1
            #print("feature vector!" + str(featureVector))
            decision1 = calculateActivationFunctionForFirstClass(featureVector, weights1, y1)
            decision2 = calculateActivationFunctionForSecondClass(featureVector, weights2, y2)
            if (decision1 == 1):  # update 1st classifier
                updateWeightsForFirstClass(featureVector, y1,c)
            if (decision2 == 1):  # update 2nd classifier
                updateWeightsForSecondClass(featureVector, y2,c)
    return c
def writeToFile(c):

    global biasOne
    global biasTwo
    combinedProbability = {"weights1": weights1, "weights2": weights2, "biasOne": biasOne, "biasTwo": biasTwo}
    with open('vanillamodel.txt', 'w') as outfile:
        json.dump(combinedProbability, outfile, indent=4)
    biasOne = biasOne - beta1/(c)
    biasTwo = biasTwo - beta2/(c)
    for weights in u1:
        u1[weights] = weights1[weights] - u1[weights]/c
        u2[weights] = weights2[weights] - u2[weights]/c

    averaged = {"weights1" : u1,"weights2":u2,"biasOne": biasOne,"biasTwo":biasTwo}
    with open('averagedmodel.txt', 'w') as outfile:
        json.dump(averaged, outfile, indent=4)
if __name__ == '__main__':
    file = readFile()
    count = trainData(file)
    writeToFile(count)