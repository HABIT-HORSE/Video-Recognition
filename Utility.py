__author__ = 'GongLi'

import os
import numpy as np
from scipy.cluster.vq import *
import pickle

def normalizeSIFT(descriptor):
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)
    '''
Steve note: last line...
https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
Normalises matrix.
In vectors, this means change the coordinates, so that the direction of the vector remains the same, but the line length now equals 1. Same in matrices?
    '''

    if norm > 1.0:
        result = np.true_divide(descriptor, norm)
    else:
        result = None

    return result

# Read in video frames under a folder
def readVideoData(pathOfSingleVideo, subSampling = 5):
    frames = os.listdir(pathOfSingleVideo)

    stackOfSIFTFeatures = []
    for frame in frames:
        completePath = pathOfSingleVideo +"/"+ frame
        lines = open(completePath, "r").readlines()
        print completePath + " FRAME!!!" #Steve
        #for line in lines[1::subSampling]:
        # for line in lines[0::subSampling]: # Steve: it was starting to parse at line 2 (1 as numbering starts at 0)...to allow for header line?]
        n=1 #Steve:
        for line in lines: # Steve: it was starting to parse at line 2 (1 as numbering starts at 0)...to allow for header line?
            print "Line: " + str(n) + " of " + str(len(lines)) + " lines" # Steve:
            n=n+1		
            #print "Line: " + str(line) #Steve
            # data = line.split(" ") # Steve: split line into an array of value items
            feature = line.split(" ") # Steve: split line into an array of value items
            feature = feature[:-1] # Steve: slice off last item in line because it's an LF \n character
            #feature = data[4:]
            # feature = data[0:] # Steve: it was starting at item #4 in each line...not sure why?
            #print "Feature: " + str(feature) #Steve
            for i in range(len(feature)):
            #for i in range(len(feature)-1): # Steve: changed to -1 so that LF characeter doesn't break int conversion in a couple of lines
                #print "Item: " + feature[i] #Steve
                #item = int(feature[i])
                item = int(float(feature[i])) #Steve
                feature[i] = item
                #print "Item: " + str(item) #Steve
			            
            # normalize SIFT feature
            feature = normalizeSIFT(feature)
            stackOfSIFTFeatures.append(feature)
    
	# print "No of lines: " + str(len(lines)) # Steve:
    return np.array(stackOfSIFTFeatures)

def buildHistogramForVideo(pathToVideo, vocabulary):
    frames = os.listdir(pathToVideo)
    size = len(vocabulary)

    stackOfHistogram = np.zeros(size).reshape(1, size)
    for frame in frames:
        # build histogram for this frame
        completePath = pathToVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        print completePath

        frameFeatures = np.zeros(128).reshape(1, 128)
        for line in lines[1:]:
            data = line.split(" ")
            feature = data[4:]

            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            feature = normalizeSIFT(feature)
            frameFeatures = np.vstack((frameFeatures, feature))

        frameFeatures = frameFeatures[1:]
        codes, distance = vq(frameFeatures, vocabulary)

        histogram = np.zeros(size)
        for code in codes:
            histogram[code] += 1

        stackOfHistogram = np.vstack((stackOfHistogram, histogram.reshape(1,size)))

    return stackOfHistogram[1:]

def writeDataToFile(filePath, data):
    file = open(filePath, "w")
    pickle.dump(data, file)
    file.close()

def normalize(X):
    row = X.shape[0]
    column = X.shape[1]

    maxValues = np.amax(X, axis=1)
    minValues = np.amin(X, axis=1)

    for i in range(row):
        for j in range(column):
            X[i][j] = (X[i][j] - minValues[i]) * 2.0 / (maxValues[i] - minValues[i])

def storeObject(fileName, obj):
    file = open(fileName, "wb")
    pickle.dump(obj, file)
    file.close()

def loadObject(fileName):
    file = open(fileName, "rb")
    obj = pickle.load(file)
    return obj

if __name__ == "__main__":
    path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak/wedding/VTS_05_01_1318"
    for item in os.listdir(path):
        print path +"/"+item









