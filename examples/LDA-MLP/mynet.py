from singa.proto.model_pb2 import kTrain, kEval
# from __init__ import __version__
# import tensor
# import layer
# import snapshot
# import cPickle as pickle
from singa import net as ffnet
# import os

# return probability as well
class ProbFeedForwardNet(ffnet.FeedForwardNet):
    
    def train(self, x, y):
        '''Run BP for one iteration.
        Currently only support nets with a single output layer, and a single
        loss objective and metric.
        For multiple outputs (with multiple loss/metric), please manually
        call forward, compute loss/metric and call backward. backward() is also
        more memory efficient than this function.
        Args:
            x: input data, a single input Tensor or a dict: layer name -> Tensor
            y: label data, a single input Tensor.
        Returns:
            gradients of parameters and the loss and metric values.
        '''
        out = self.forward(kTrain, x)
        l = self.loss.forward(kTrain, out, y)
        g = self.loss.backward()
        g /= x.shape[0]
        m = None
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        grads = []  # store all gradient tensors; memory inefficient
        for _, _, grad, _ in self.backward(g):
            grads.extend(grad[::-1])
        return grads[::-1], (l.l1(), m), out

    
    def evaluate(self, x, y):
        '''Evaluate the loss and metric of the given data.

        Currently only support nets with a single output layer, and a single
        loss objective and metric.
        TODO(wangwei) consider multiple loss objectives and metrics.

        Args:
            x: input data, a single input Tensor or a dict: layer name -> Tensor
            y: label data, a single input Tensor.
        '''
        out = self.forward(kEval, x)
        l = None
        m = None
        assert self.loss is not None or self.metric is not None,\
            'Cannot do evaluation, as neither loss nor metic is set'
        if self.loss is not None:
            l = self.loss.evaluate(kEval, out, y)
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        return l, m, out

