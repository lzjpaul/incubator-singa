import numpy as np
import os
import random
import time
from multiprocessing import Process, Queue
from singa import data
from singa import image_tool


# cifar numpy iter: random crop and random flip
class CifarBatchIter(data.ImageBatchIter):
    def __init__(self, img_feature_file, img_label_file, batch_size, image_transform,
                 shuffle=True, capacity=10):
        self.img_feature_file = img_feature_file
        self.img_label_file = img_label_file
        self.queue = Queue(capacity)
        self.batch_size = batch_size
        self.image_transform = image_transform
        self.shuffle = shuffle
        self.stop = False
        self.p = None
        self.num_samples = img_feature_file.shape[0]
        print "self.num_samples: ", self.num_samples

    def run(self):
        random_seed_idx = 0
        img_list = []
        for i in range(self.img_feature_file.shape[0]):
            img_list.append((self.img_feature_file[i], self.img_label_file[i]))
        index = 0  # index for the image
        shuffle_timer = 100
        while not self.stop:
            if index == 0 and self.shuffle:
                random.seed(shuffle_timer)
                random.shuffle(img_list)
                shuffle_timer = shuffle_timer + 1
            if not self.queue.full():
                x = []
                y = np.empty(self.batch_size, dtype=np.int32)
                i = 0
                while i < self.batch_size:
                    img_feature, img_label = img_list[index]
                    aug_img_features = self.image_transform(random_seed_idx, img_feature, (32, 32), 4)
                    random_seed_idx = random_seed_idx + 1
                    assert i + len(aug_img_features) <= self.batch_size, \
                        'too many images (%d) in a batch (%d)' % \
                        (i + len(aug_img_features), self.batch_size)
                    for aug_img_feature in aug_img_features:
                        x.append(aug_img_feature)
                        y[i] = img_label
                        i += 1
                    index += 1
                    if index == self.num_samples:
                        index = 0  # reset to the first image
                # enqueue one mini-batch
                self.queue.put((np.asarray(x, np.float32), y))
            else:
                time.sleep(0.1)
        return

def batch_data_augment_tool(epoch, epoch_sample_num, img_idx, img_feature_file):
    # print "original cifar shape: ", img_feature_file.shape
    random_seed_base = epoch * epoch_sample_num  # modify here
    img_list = []
    for i in range(img_feature_file.shape[0]):
        img_list.append(img_feature_file[i])
    x = []
    i = 0
    while i < img_feature_file.shape[0]:
        img_feature = img_list[i]
        aug_img_feature = numpy_crop_flip((random_seed_base+img_idx[i]), img_feature, (32, 32), 4) # modify here img_idx[i]
        # assert i + len(aug_img_features) <= img_feature_file.shape[0], \
        #    'too many images (%d) in a batch (%d)' % \
        #    (i + len(aug_img_features), self.batch_size)
        x.append(aug_img_feature) # modify here
        i += 1
    # enqueue one mini-batch
    # print "augment cifar shape: ", np.asarray(x, np.float32).shape
    return np.asarray(x, np.float32)


def data_augment_tool(epoch, img_feature_file):
    print "original cifar shape: ", img_feature_file.shape
    random_seed_base = epoch * img_feature_file.shape[0]
    img_list = []
    for i in range(img_feature_file.shape[0]):
        img_list.append(img_feature_file[i])
    x = []
    i = 0
    while i < img_feature_file.shape[0]:
        img_feature = img_list[i]
        aug_img_features = numpy_crop_flip((random_seed_base+i), img_feature, (32, 32), 4)
        # assert i + len(aug_img_features) <= img_feature_file.shape[0], \
        #    'too many images (%d) in a batch (%d)' % \
        #    (i + len(aug_img_features), self.batch_size)
        for aug_img_feature in aug_img_features:
            x.append(aug_img_feature)
            i += 1
    # enqueue one mini-batch
    print "augment cifar shape: ", np.asarray(x, np.float32).shape
    return np.asarray(x, np.float32)


def numpy_crop(random_seed_idx, img_feature, crop_shape, pad):
    new_img_feature = np.zeros(img_feature.shape)
    np.random.seed(random_seed_idx)
    xoffs, yoffs = (np.random.random_integers(-pad, pad), np.random.random_integers(-pad, pad))
    input_y = (max(0, 0 + yoffs), min(32, 32 + yoffs))
    data_y = (max(0, 0 - yoffs), min(32, 32 - yoffs))
    input_x = (max(0, 0 + xoffs), min(32, 32 + xoffs))
    data_x = (max(0, 0 - xoffs), min(32, 32 - xoffs))
    new_img_feature[:, input_y[0]:input_y[1], input_x[0]:input_x[1]] = img_feature[:, data_y[0]:data_y[1], data_x[0]:data_x[1]] 
    return new_img_feature


def numpy_crop_flip(random_seed_idx, img_feature, crop_shape, pad):
    # print "img_feature shape: ", img_feature.shape
    img_feature = numpy_crop(random_seed_idx, img_feature, crop_shape, pad)
    np.random.seed(random_seed_idx)
    if np.random.randint(2)==0:
        img_feature = np.flip(img_feature, 2)
    return img_feature
