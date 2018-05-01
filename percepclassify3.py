import json
import itertools
import sys
import math
import collections
from collections import defaultdict
import string
import random

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing",
                     "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
                     "it", "its", "itself", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
                     "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
                     "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"]


def readTestData():
    file = open(sys.argv[2], 'r', encoding="utf-8")
    return file


def readTrainedFile():
    data = json.load(open(sys.argv[1]))
    return data


def perceptronOne(featureVector, data):
    # print("inside perceptron class")
    y1 = 0
    for features in featureVector:
        if features in data["weights1"]:
                y1 += featureVector[features] * data["weights1"][features]
  
    
    y1 = y1 + data["biasOne"]
    if (y1 < 0):
        return "Fake"
    if (y1 >= 0):
        return "True"


def perceptronTwo(featureVector, data):
    # print("inside perceptron class")
    y2 = 0
    for features in featureVector:
        if features in data["weights2"]:
                y2 += featureVector[features] * data["weights2"][features]
   
    y2 = y2 + data["biasTwo"]
    if (y2 < 0):
        return "Neg"
    if (y2 >= 0):
        return "Pos"


def writeToFile(result):
    with open("percepoutput.txt", 'w', encoding='utf-8') as file:
        file.write(result)


def predictClasses(file, data):
    resultString = ""
    for line in file:
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
     	# map punctuation to space
        lines = line.translate(translator)
        line = lines.strip("\n").split(" ")
        key = line[0]  # unique identifier
        featureVector = {}
        for words in line:
            if words != line[0]:
                if words != "" and words.lower() not in stopwords:
                    if words in featureVector:
                        featureVector[words.lower()] += 1
                    else:
                        featureVector[words.lower()] = 1
        # print(featureVector)
        classOne = perceptronOne(featureVector, data)
        classTwo = perceptronTwo(featureVector, data)
        resultString += key + " " + classOne + " " + classTwo + "\n" 

    writeToFile(resultString)


if __name__ == '__main__':
    file = readTestData()
    data = readTrainedFile()
    predictClasses(file, data)