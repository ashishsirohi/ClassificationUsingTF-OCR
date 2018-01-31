# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    #print features
    #features = basicFeatureExtractor(datum)
    #print features

    #return features

    "*** YOUR CODE HERE ***"
    #print len(features)
    num_loops = 0
    flags = []
    loop_ready_flag = 0
    loop_open_flag = 0
    loop_close_flag = 0
    loop_start_index = 0
    loop_end_index = 0
    #print features
    for feature in features:
        indexes = []
        loop_flag = 0
        #print "feature: ", feature
        for i, f in enumerate(feature):
            if f == 1:
                indexes.append(i)

        print indexes

        if loop_ready_flag == 1 and loop_open_flag == 0 and len(indexes)>0 and len(indexes) == indexes[-1]-indexes[0]+1:
            first_index = indexes[0]
            last_index = indexes[-1]

        if loop_ready_flag == 1 and len(indexes)>0 and len(indexes) != indexes[-1]-indexes[0]+1:
            for i in range(len(indexes)-1):
                if abs(indexes[i]-indexes[i+1]) > 1:
                    curr_loop_start = indexes[i]
                    curr_loop_end = indexes[i+1]
                    break
            if curr_loop_start > first_index and curr_loop_end < last_index:
                loop_open_flag = 1
                loop_flag = 1
                loop_start_index = curr_loop_start
                loop_end_index = curr_loop_end
            else:
                loop_open_flag = 0

            first_index = indexes[0]
            last_index = indexes[-1]

        flags.append(loop_flag)

        if loop_open_flag == 1 and len(indexes)>0 and len(indexes) == indexes[-1]-indexes[0]+1:
            if indexes[0] <= loop_start_index and indexes[-1] >= loop_end_index:
                num_loops += 1

            loop_ready_flag = 0
            loop_open_flag = 0
            first_index = indexes[0]
            last_index = indexes[-1]

        if loop_ready_flag == 0 and len(indexes)>0 and len(indexes) == indexes[-1]-indexes[0]+1:
            loop_ready_flag = 1
            first_index = indexes[0]
            last_index = indexes[-1]

        #print loop_flag
    print flags
    """print "loop_open_flag: ", loop_open_flag
    if loop_open_flag == 0:
        util.raiseNotDefined()
    count = 0
    for x in flags:
        if x == 1:
            count += 1
        else:
            if count >= 3:
                num_loops += 1
            count = 0"""

    print "Number of Loops: ", num_loops

    #util.raiseNotDefined()

    extra_features = np.array([0, 0, 0])
    if num_loops == 0:
        extra_features = np.array([1, 0, 0])
    elif num_loops == 1:
        extra_features = np.array([0, 1, 0])
    elif num_loops == 2:
        extra_features = np.array([0, 0, 1])

    #num_loops = num_loops + 1
    #file1 = open("file2.txt", "a")
    #file1.write(str(num_loops))
    #file1.write("\n")
    features = basicFeatureExtractor(datum)
    return np.append(features, extra_features)
    print "----------"
    #util.raiseNotDefined()

    #return features


def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
