from statistics import mean, stdev
import numpy as np
import torch



def convertToEpidemic(y, st=1):
        m = mean(y)
        std = stdev(y)
        threshold = m + st * std
        newY = [1 if x > threshold else 0 for x in y]
        return np.array(newY)

def numberEpidemics(y):
        counter = 0
        indices = []
        lastOne = False
        for a in range(len(y)):
            if y[a] == 1 and (not lastOne):
                counter = counter + 1
                indices.append(a)
                lastOne = True
            elif y[a] == 0:
                lastOne = False

        print(counter)
        print(indices)
        return counter


