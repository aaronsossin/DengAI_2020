from statistics import mean, stdev
import numpy as np
import torch
class Epidemic:

    def convertToEpidemic(self, y):
        m = mean(y)
        std = stdev(y)
        threshold = m + 1.0 * std
        newY = [1 if x > threshold else 0 for x in y]
        return np.array(newY)

    def numberEpidemics(self, y):
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

    def returnOptimizedModel_binary(self, X, y):
        # SJ Optimization
        criterion = torch.nn.MSELoss()
        scores = []
        aa = []
        bb = []
        for a in range(10, 100, 20):  # 10, 100, 20
            for b in range(100, 501, 200):  # 100, 501, 200
                scores.append(cross_val(a, b, X, y))
                aa.append(a)
                bb.append(b)

        print("MAX SCORE: ", max(scores), " \n")
        index = scores.index(max(scores))
        print("Max SCORE: ", max(scores))
        print("Batch: ", aa[index], " epochs: ", bb[index])
        model = baseline()
        model.fit(X, y, batch_size=aa[index], epochs=bb[index], verbose=0)
        # model.fit(X, y)
        return model