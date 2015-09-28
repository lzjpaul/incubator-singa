'''
Base code from pytensor: the python implementation of MATLAB based tensor code
https://code.google.com/p/pytensor

Matricization of a regular, dense tensor object.
The code is the python implementation of the @tenmat folder
in the MATLAB Tensor Toolbox
'''
import numpy as np

import dtensor
import tools


def _convert_np(dim):
    """ Convert an non-empty list to numpy array """
    if dim is not None and isinstance(dim, list):
        return np.array(dim)
    return dim


class tenmat:
    
    data = None
    rindices = None
    cindices = None
    tsize = None
    
    def __init__(self, T, rdim=None, cdim=None, tsiz=None, option=None):
        """
        A matricized tensor object.
        Converting a tensor to a matrix requires an ordered mapping
        of the tensor indices for the rows and columns
        """
        rdim = _convert_np(rdim)
        cdim = _convert_np(cdim)
        tsiz = _convert_np(tsiz)
        if rdim is None and cdim is None:
            raise ValueError("Both rdim and cdim are not given")
        if option not in (None, 'fc', 'bc', 't'):
            raise ValueError("unknown option {0}".format(option))
        if rdim is not None and cdim is not None and tsiz is not None:
            if isinstance(T, np.ndarray):
                self.data = T.copy()
            if isinstance(T, dtensor):
                self.data = T.data.copy()
            self.rindices = rdim
            self.cindices = cdim
            self.tsize = tuple(tsiz)
            n = len(self.tsize)
            temp = np.concatenate((self.rindices, self.cindices))
            temp.sort()
            if not ((np.arange(n) == temp).all()):
                raise ValueError("Incorrect specification of dimensions")
            elif (tools.getelts(self.tsize, self.rindices).prod()
                  != len(self.data)):
                raise ValueError("size(T,0) does not match size specified")
            elif (tools.getelts(self.tsize, self.cindices).prod()
                  != len(self.data[0])):
                raise ValueError("size(T,1) does not match size specified")
            return
        
        T = T.copy()    # copy the tensor
        self.tsize = T.shape
        n = T.ndims()
        
        if rdim is not None:
            rdims = rdim
            if cdim is not None:
                cdims = cdim
            elif option == 'fc':
                if rdims.size != 1:
                    raise ValueError("only one row dimension for 'fc' option")
                cdims = np.append(range(rdim[0] + 1, n),
                                  range(0, rdim[0]))
            elif option == 'bc':
                if rdims.size != 1:
                    raise ValueError("only one row dimension for 'bc' option")
                cdims = np.append(range(0, rdim[0])[::-1],
                                  range(rdim[0] + 1, n)[::-1])
            else:
                cdims = np.setdiff1d(np.arange(n), rdims)
        else:
            cdims = cdim
            rdims = np.setdiff1d(np.arange(n), cdims)
        #error check
        temp = np.concatenate((rdims, cdims))
        temp.sort()
        if not ((np.arange(n) == temp).all()):
            raise ValueError("error, Incorrect specification of dimensions")
            
        #permute T so that the dimensions specified by RDIMS come first
        #!!!! order of data in ndarray is different from that in Matlab!
        #this is (kind of odd process) needed to conform the result with Matlab!
        #lis = list(T.shape);
        #temp = lis[T.ndims()-1];
        #lis[T.ndims()-1] = lis[T.ndims()-2];
        #lis[T.ndims()-2] = temp;
        #T.data = T.data.reshape(lis).swapaxes(T.ndims()-1, T.ndims()-2);
        #print T;
        #T = T.permute([T.ndims()-1, T.ndims()-2]+(range(0,T.ndims()-2)));
        T = T.permute(np.concatenate((rdims, cdims)))
        #convert T to a matrix;
        
        row = tools.getelts(self.tsize, rdims).prod()
        col = tools.getelts(self.tsize, cdims).prod()
        
        self.data = T._data.reshape([row, col], order='F')
        self.rindices = rdims
        self.cindices = cdims
        
    def copy(self):
        return tenmat(self.data, self.rindices, self.cindices, self.tsize)
        
    def totensor(self):
        sz = self.tsize
        order = np.concatenate((self.rindices, self.cindices))
        order = order.tolist()
        data = self.data.reshape(tools.getelts(sz, order))
        data = dtensor.dtensor(data).ipermute(order).data
        return dtensor.dtensor(data, sz)
        
    def tondarray(self):
        """
        Returns an ndarray with the same values as the tenmat
        """
        return self.data
    
    def __str__(self):
        ret = ""
        ret += "matrix corresponding to a tensor of size {0}\n".format(self.tsize)
        ret += "rindices {0}\n".format(self.rindices)
        ret += "cindices {0}\n".format(self.cindices)
        ret += "{0}\n".format(self.data)
        return ret
