from singa import tensor
import numpy as np
from singa import device

dev = device.create_cuda_gpu()
# dev1 = device.create_cuda_gpu()

# dev = device.get_default_device()
cpudev = device.get_default_device()

tw = tensor.from_numpy(np.ones((1,4), dtype = np.float32))
tw.to_device(dev)

texp = tensor.from_numpy(np.zeros((1,4)))

texp_res = tensor.exp(texp)
texp_res.to_device(cpudev)

print "texp: ", tensor.to_numpy(texp_res)
print "texp dtype: ", tensor.to_numpy(texp_res).dtype

tadd = tensor.from_numpy(np.array([2., 2., 2., 2.]))
tadd.to_device(dev)

eltwise_add_res = tw + tadd
eltwise_add_res.to_device(cpudev)
print "element-wise add: ", tensor.to_numpy(eltwise_add_res)

tdivide = tensor.from_numpy(np.array([2., 2., 2., 2.]))
tdivide.to_device(dev)

eltwise_div_res = tensor.div(tw, tdivide)
eltwise_div_res.to_device(cpudev)

eltwise_div_res_2 = tw / tdivide
eltwise_div_res_2.to_device(cpudev)

print "element-wise divide: ", tensor.to_numpy(eltwise_div_res)
eltwise_div_res.to_device(dev)

print "element-wise divide2: ", tensor.to_numpy(eltwise_div_res_2)
eltwise_div_res_2.to_device(dev)

eltwise_mul_res = tensor.eltwise_mult(eltwise_div_res, tdivide)

eltwise_mul_res.to_device(cpudev)
print "element-wise mul: ", tensor.to_numpy(eltwise_mul_res)
eltwise_mul_res.to_device(dev)

eltwise_mul_res_2 = eltwise_div_res * tdivide
eltwise_div_res.to_device(cpudev)
tdivide.to_device(cpudev)
print "eltwise_div_res shape: ", tensor.to_numpy(eltwise_div_res).shape
print "tdivide shape: ", tensor.to_numpy(tdivide).shape

eltwise_div_res.to_device(dev)
tdivide.to_device(dev)


eltwise_mul_res_2.to_device(cpudev)
print "element-wise mul2: ", tensor.to_numpy(eltwise_mul_res_2)
eltwise_mul_res_2.to_device(dev)

print "before x declaration"
tx = tensor.from_numpy(np.array([[1.,2.,3.,4.],[5.,6.,7.,8.]]))
print "after x declaration"
tx.to_device(dev)

mul_res = tensor.mult(tw, tx.T())

mul_res.to_device(cpudev)
print "mul: ", tensor.to_numpy(mul_res)
mul_res.to_device(dev)

add_x = tensor.from_numpy(np.array([10., 2.]))
add_x.to_device(dev)

tx.mult_column(add_x)
tx.to_device(cpudev)
print "mult_column: ", tensor.to_numpy(tx)
tx.to_device(dev)
tensor_sum_column_res = tensor.sum(tx, 0)
tensor_sum_column_res.to_device(cpudev)
print "tensor_sum_column_res: ", tensor.to_numpy(tensor_sum_column_res)
tensor_sum_column_res.to_device(dev)

tx.to_device(dev)
# twotx_1 = (-2.) * tx
twotx_1 = -1 * tx
twotx_2 = tensor.eltwise_mult(tx, 2.)
twotx_1.to_device(cpudev)
twotx_2.to_device(cpudev)
print "2 times x1: ", tensor.to_numpy(twotx_1)
print "2 times x: ", tensor.to_numpy(twotx_2)
tx = tx - 2.
tx.to_device(cpudev)
print "x minus 2: ", tensor.to_numpy(tx)
tx.to_device(dev)
