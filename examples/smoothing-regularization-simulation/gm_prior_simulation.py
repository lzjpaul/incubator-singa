import numpy as np
# import matplotlib.pyplot as plt
import pickle

class Simulator():
    def __init__(self, gm_num, dimension, sample_num, pi, variance, covariace):
        # init simulator parameters
        self.gm_num, self.dimension, self.sample_num = gm_num, dimension, sample_num
        self.pi, self.variance, self.covariance = pi, variance, covariace

    def wGenerator(self):
        # generate w_origin from mixture coefficient
        self.w_origin = np.random.choice(4, size=(self.dimension), p=self.pi)
        self.w = np.ndarray(shape=(self.dimension), dtype='float32')

        # generate w with corresponding gaussian variance
        gm_count = np.bincount(self.w_origin)
        for gm_index in xrange(self.gm_num):
            self.w[self.w_origin==gm_index] = np.random.normal(0.0, self.variance[gm_index], size=(gm_count[gm_index]))

    def xGenerator(self):
        self.x = np.random.multivariate_normal(mean=np.zeros(shape=(self.dimension)), cov=self.covariance, size=(self.sample_num,))

    def labelGenerator(self):
        y_vals = 1/(1+np.exp(-np.dot(self.x, self.w)))
        uniform_vals = np.random.uniform(low=0.0, high=1.0, size=(self.sample_num))
        self.label = (y_vals>=uniform_vals)

        # save generated y_vals
        with open('generated_y_vals.dta', 'w') as saveFile:
            for index, item in enumerate(self.label):
                saveFile.write("%d: %f > %f\n" %(item, y_vals[index], uniform_vals[index]))


def generateCov(n=5, a=1, showCov=True):
    # see http://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor
    A = np.matrix([np.random.randn(n) + np.random.randn(1) * a for i in range(n)])
    A = A * np.transpose(A)
    D_half = np.diag(np.diag(A) ** (-0.5))
    C = D_half * A * D_half
    # show the cov matrix
    if showCov:
        vals = list(np.array(C.ravel())[0])
        plt.hist(vals, range=(-1,1))
        plt.show()
        plt.imshow(C, interpolation=None)
        plt.show()
    return C


if __name__ == '__main__':
    gm_num, dimension, sample_num = 4, 1000, 10000
    pi, variance, covariance = np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.01, 0.1, 0.2, 0.4]), generateCov(n=dimension, a=1, showCov=False)

    simulator = Simulator(gm_num, dimension, sample_num, pi, variance, covariance)
    simulator.wGenerator()
    simulator.xGenerator()
    simulator.labelGenerator()

    # save the generated simulator
    with open('simulator.pkl', 'w') as saveFile:
        pickle.dump(simulator, saveFile)

'''
generated results see generated_y_vals.dta
simulator file saved in simulator.pkl
'''
