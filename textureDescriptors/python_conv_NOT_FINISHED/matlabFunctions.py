import numpy as np

class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def error(message):
    raise Error(message)

def isempty(x_obj):
    return x_obj == None

def size(nparray, index=None):
    ret = np.shape(nparray)

    if not isempty(index):
        ret = ret[index]

    return ret

def double(nparray):
    return nparray.astype(np.float64)