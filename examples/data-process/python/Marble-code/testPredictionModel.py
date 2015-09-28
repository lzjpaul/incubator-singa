import tensor
import sptensor
import numpy as np
import ktensor
import predictionModel

subs = np.array([[0,3,1], [1,0,1], [1,2,1], [1,3,1], [2,4,0], [3,0,0],[4,4,1]]);
vals = np.array([[1],[1],[1],[1],[4],[3],[2]]);
siz = np.array([5,5,2]) # 5x5x2 tensor

np.random.seed(0)
X = sptensor.sptensor(subs, vals, siz)
Y = np.random.randint(2, size=5)
pm = predictionModel.predictionModel(X, Y, 2)
pm.evaluatePrediction()