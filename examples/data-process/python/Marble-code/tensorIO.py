"""
File that contains common methods for tensor IO
"""
import numpy as np
import shelve
import csv
from collections import OrderedDict
from bson.binary import Binary
import cPickle

import sptensor

AXIS = "axis"
CLASS = "class"
MEM_MAT = "memMat"
MODE_DIM = "modeDim"
DATA_FILE = "data"
INFO_FILE = "info"


def construct_tensor(filename, axisIdx, valueIdx, sep=",", axisDict=None):
    """
    Parse a delimited file to obtain a sparse tensor

    Parameters
    ------------
    filename : the filename of the tensor file
    axisIdx : a list corresponding to the column index of a specific modes
              (first element = first column idx)
    valueIdx : the column index corresponding to the value
    sep : the delimiter of the file (default is comma)
    axisDict : a mapping between the indices and the actual axis values
    """
    N = len(axisIdx)
    # create the tensor dictionary
    if axisDict is None:
        axisDict = {}
        for n in range(N):
            axisDict[n] = OrderedDict(sorted({}.items(), key=lambda t: t[1]))
    tensorIdx = np.zeros((1, N), dtype=int)
    tensorVal = np.array([[0]], dtype=float)
    f = open(filename, "rb")
    for row in csv.reader(f, delimiter=sep):
        rowIdx = []
        for n in range(N):
            axisKey = row[axisIdx[n]]
            # add to the dictionary if it doesn't exist
            if axisKey not in axisDict[n]:
                axisDict[n][axisKey] = len(axisDict[n])
            rowIdx.append(axisDict[n].get(axisKey))
        tensorIdx = np.vstack((tensorIdx, np.array([rowIdx], dtype=int)))
        tensorVal = np.vstack((tensorVal, [[float(row[valueIdx])]]))
    f.close()
    tShape = [len(axisDict[n]) for n in range(N)]
    tenX = sptensor.sptensor(tensorIdx, tensorVal, np.array(tShape))
    return tenX, axisDict


def save_tensor(X, axisDict, patClass, outfilePattern):
    """
    Save a single tensor
    (the original data, axis information, and classification)
    The "data" file contains the raw tensor information in numpy binary format
    First is the number of tensors in the list, then the sptensor information,
    and finally shared modes

    Parameters
    ------------
    X : tensor type
    sharedModes: a 2-d numpy array specifying common shared modes
    axisDict : a mapping between the indices and the actual axis values
    patClass : a map between patients and the labels
    outFilename : the pattern for the output format, note that {0} is necessary
                  as 2 files are produced
    """
    # save tensor via sptensor
    outfile = open(outfilePattern.format(DATA_FILE), "wb")
    X.save(outfile)
    outfile.close()
    tensorInfo = shelve.open(outfilePattern.format(INFO_FILE), "c")
    tensorInfo[AXIS] = axisDict
    tensorInfo[CLASS] = patClass
    tensorInfo.close()


# Load a single tensor and the axis information
def load_tensor(inFilePattern):
    infile = open(inFilePattern.format(DATA_FILE), "rb")
    X = sptensor.load(infile)
    infile.close()
    tensorInfo = shelve.open(inFilePattern.format(INFO_FILE), "r")
    axisDict = tensorInfo[AXIS]
    classDict = tensorInfo[CLASS]
    tensorInfo.close()
    return X, axisDict, classDict


# Read the file with the class information
def readClassFile(filename, patDict, patIdx, classIdx):
    patClass = OrderedDict()
    f = open(filename, "rb")
    for row in csv.reader(f):
        if row[patIdx] not in patDict:
            print "Doesn't have: " + row[patIdx]
            continue
        patId = patDict.get(row[patIdx])
        patClass[patId] = int(row[classIdx])
    f.close()
    return patClass


def ktensor_to_mongo(M, axisDict):
    """
    Flatten the CP decomposition so that it can be written into a mongo format
    """
    output = {}
    output['L'] = M.lmbda
    for axis in axisDict.keys():
        axisTuple = (axis)
        output[axisTuple] = Binary(cPickle.dumps(M.U[axis], protocol=2))
    return output
