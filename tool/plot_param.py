
'''
The code is adapted from ConvNet's ShowNet.py

# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
#import pylab as plt
from pb2 import common_pb2
import numpy as np
from math import ceil, sqrt
import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
def make_filter_fig(filters, outfile,  combine_chans):
  filter_start = 0
  fignum = 1
  num_filters = filters.shape[-1]
  FILTERS_PER_ROW = 16
  MAX_ROWS = 16
  MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
  num_colors = filters.shape[0]
  f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
  filter_end = min(filter_start+MAX_FILTERS, num_filters)
  filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))

  filter_size = int(sqrt(filters.shape[1]))
  fig = plt.figure(1)
  #fig.text(.5, .95, '%s %dx%d filters %d-%d' % (title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center')
  num_filters = filter_end - filter_start
  if not combine_chans:
    bigpic = np.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=np.single)
  else:
    bigpic = np.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=np.single)

  for m in xrange(filter_start,filter_end):
    filter = filters[:,:,m]
    y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
    if not combine_chans or num_colors == 1:
      for c in xrange(num_colors):
        filter_pic = filter[c,:].reshape((filter_size,filter_size))
        bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
            1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
    else:
        filter_pic = filter.reshape((3, filter_size,filter_size))
        bigpic[:,
          1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
          1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic

  plt.xticks([])
  plt.yticks([])
  if not combine_chans:
    plt.imshow(bigpic, cmap=plt.cm.gray, interpolation='nearest')
  else:
    bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
    #print bigpic
    plt.imshow(bigpic, interpolation='nearest')
  plt.savefig(outfile)
  plt.clf()

def plot_filters(outfile, filters, yuv_to_rgb = False):
  '''
  filters: channels, h*w, num
  '''
  filter_start = 0 # First filter to show
  # Convert YUV filters to RGB
  if yuv_to_rgb and channels == 3:
    R = filters[0,:,:] + 1.28033 * filters[2,:,:]
    G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
    B = filters[0,:,:] + 2.12798 * filters[1,:,:]
    filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B

  # Make sure you don't modify the backing array itself here -- so no -= or /=
  filters = filters - filters.min()
  filters = filters / filters.max()
  make_filter_fig(filters, outfile, True)

def plot_all_params(infile, outfolder):
  fd = open(infile, 'rb')
  bps = common_pb2.BlobProtos()
  bps.ParseFromString(fd.read())
  outprefix = os.path.join(outfolder, os.path.splitext(os.path.split(infile)[1])[0])
  for (name, blob) in zip(bps.name, bps.blob):
    filters = np.asarray(blob.data, dtype = np.float32).reshape(tuple(blob.shape))
    #print filters.shape
    #W = np.swapaxes(filters, 0, 1).reshape(3, blob.shape[1]/3, blob.shape[0])
    #print W.shape
    plot_filters(outprefix + '-' + name + '.png', filters)
    break

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'Usage: python plot.py <path>'
    print 'the program use <path.dat> as input to generate a picture at <path.jpg>'
    sys.exit()

  plot_all_params(sys.argv[1], sys.argv[2])
