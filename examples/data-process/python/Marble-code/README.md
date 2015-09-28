# Overview #
A Python-based tensor library based on the [MATLAB Tensor Toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox) and [PyTensor](https://code.google.com/p/pytensor/). The library contains the following tensor factorizations:

* Standard CP (ALS)
* CP factorization using Alternating Poisson Regression [Paper](http://www.sandia.gov/~tgkolda/pubs/pubfiles/ChKo12.pdf)
* Marble (sparse CP-APR)

# Requirements #
The code requires numpy, scipy, scikit-learn, and munkres

# Layout #
The tensor implementations (tensor, ktensor, sptensor) and tensor factorization code are found in the base directory, while the subdirectories contain test and simulation code.