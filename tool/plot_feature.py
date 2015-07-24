from pb2 import common_pb2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib
import os
import sys
from bh_tsne import bhtsne

matplotlib.rcParams.update({'font.size': 17})
import numpy as np

def make_N_colors(cmap_name, N):
  cmap = cm.get_cmap(cmap_name, N)
  return cmap(np.arange(N))


def plot2d(x, y, label, fname):
  plt.clf()
  color=make_N_colors('gist_rainbow', 10)
  marker=["o","x","d","s","+","v",">","p",'*','.']
  count=[0]*10
  for i,lb in enumerate(label):
    plt.scatter(x[i],y[i], marker=marker[lb], s=40, c='w', edgecolor=color[lb])
    count[lb]+=1
  print count
  #plt.show()
  plt.savefig(fname)

def plot_all_feature(infile, outfolder):
  fd = open(infile, 'rb')
  bps = common_pb2.BlobProtos()
  bps.ParseFromString(fd.read())
  labels = None
  outprefix = os.path.join(outfolder, os.path.splitext(os.path.split(infile)[1])[0])
  for (name, blob) in zip(bps.name, bps.blob):
    if 'label' in name:
      assert blob.shape[0] == len(blob.data)
      labels = np.asarray(blob.data, dtype = np.int)
  for (name, blob) in zip(bps.name, bps.blob):
    if len(blob.shape) >1 :
      s = (blob.shape[0], len(blob.data) / blob.shape[0])
      fea = np.asarray(blob.data, dtype = np.float32).reshape(s)
      X = []
      Y = []
      for point in bhtsne.bh_tsne(fea):
        X.append(point[0])
        Y.append(point[1])
      plot2d(X, Y, labels, outprefix + '-' + name + '.png')

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print 'Usage: python plot.py <path>'
    print 'the program use <path.dat> as input to generate a picture at <path.jpg>'
    sys.exit()

  plot_all_features(sys.argv[1], sys.argv[2])
