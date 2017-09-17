import math
import numpy as np

def numpy_initializer_uniform(random_seed_idx, t, fan_in=0, fan_out=0):
    '''Initialize the values of the input tensor following a uniform
    distribution with specific bounds.
    Args:
        fan_in(int): for the weight Tensor of a convolution layer,
            fan_in = nb_channel * kh * kw; for dense layer,
            fan_in = input_feature_length
        fan_out(int): for the convolution layer weight Tensor,
            fan_out = nb_filter * kh * kw; for the weight Tensor of a dense
            layer, fan_out = output_feature_length
    Ref: [Bengio and Glorot 2010]: Understanding the difficulty of
    training deep feedforward neuralnetworks.
    '''
    assert fan_in > 0 or fan_out > 0, \
        'fan_in and fan_out cannot be 0 at the same time'
    avg = 2
    if fan_in * fan_out == 0:
        avg = 1
    x = math.sqrt(3.0 * avg / (fan_in + fan_out))
    np.random.seed(random_seed_idx)
    return np.random.uniform(-x, x, size=t.shape)
    
def numpy_initializer_gaussian(random_seed_idx, t, fan_in=0, fan_out=0):
    '''Initialize the values of the input tensor following a Gaussian
    distribution with specific std.
    Args:
        fan_in(int): for the weight Tensor of a convolution layer,
            fan_in = nb_channel * kh * kw; for dense layer,
            fan_in = input_feature_length
        fan_out(int): for the convolution layer weight Tensor,
            fan_out = nb_filter * kh * kw; for the weight Tensor of a dense
            layer, fan_out = output_feature_length
    Ref Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Delving Deep into
    Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    '''
    assert fan_in > 0 or fan_out > 0, \
        'fan_in and fan_out cannot be 0 at the same time'
    avg = 2
    if fan_in * fan_out == 0:
        avg = 1
    std = math.sqrt(2.0 * avg / (fan_in + fan_out))
    np.random.seed(random_seed_idx)
    return np.random.normal(0, std, size=t.shape)


