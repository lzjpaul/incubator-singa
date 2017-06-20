# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================

import sys, os
import traceback
import time
import urllib
import numpy as np
from argparse import ArgumentParser


from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType

import model


def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train Alexnet over CIFAR10")

        parser.add_argument('-p', '--port', default=9989, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=140)

        # Process arguments
        args = parser.parse_args()
        port = args.port

        use_cpu = args.use_cpu
        if use_cpu:
            print "runing with cpu"
            dev = device.get_default_device()
        else:
            print "runing with gpu"
            dev = device.create_cuda_gpu()

        # start to train
        agent = Agent(port)
        train(dev, agent, args.max_epoch, use_cpu)
        # wait the agent finish handling http request
        agent.stop()
    except SystemExit:
        return
    except:
        # p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")


def flat_data(records, idxs, visits_per_record, items_per_visit):
    x = np.zeros((len(idxs), 1, visits_per_record, items_per_visit),
                 dtype=np.float32)
    y = np.zeros((len(idxs),), dtype=np.int32)

    for (k, v) in enumerate(idxs):
        rec = records[v]
        y[k] = int(rec[0])
        idx = 1
        for visit_id in range(rec[1]):
            idx += 1
            num_items = rec[idx]
            for item_id in range(num_items):
                idx += 1
                x[k][0][visit_id][rec[idx]-1] = 1
    return x, y

def download_file(url, dest):
    '''
    download one file to dest
    '''
    if not os.path.exists(dest):
        os.makedirs(dest)
    if (url.startswith('http')):
        file_name = url.split('/')[-1]
        target = os.path.join(dest, file_name)
        urllib.urlretrieve(url, target)
    return

def get_data(file_url, delimiter=','):
    '''load data'''
    records = []
    max_dig_id = 0
    max_visits = 0

    if not os.path.exists('feature.csv'):
        download_file(file_url,'.')

    with open('feature.csv', 'r') as fd:
        for rec in fd.readlines(): 
            fields = [int(v) for v in rec.split(delimiter)]
            if max_dig_id < max(fields):
                max_dig_id = max(fields)
            if max_visits < fields[1]:
                max_visits = fields[1]
            records.append(fields)
    print len(records), max_visits, max_dig_id
    return records, (1, max_visits, max_dig_id)


def handle_cmd(agent):
    pause = False
    stop = False
    while not stop:
        key, val = agent.pull()
        if key is not None:
            msg_type = MsgType.parse(key)
            if msg_type.is_command():
                if MsgType.kCommandPause.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    pause = True
                elif MsgType.kCommandResume.equal(msg_type):
                    agent.push(MsgType.kStatus, "Success")
                    pause = False
                elif MsgType.kCommandStop.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    stop = True
                else:
                    agent.push(MsgType.kStatus,"Warning, unkown message type")
                    print "Unsupported command %s" % str(msg_type)
        if pause and not stop:
            time.sleep(0.1)
        else:
            break
    return stop


def get_lr(epoch):
    '''change learning rate as epoch goes up'''
    return 0.001


def train(dev, agent, max_epoch, use_cpu, batch_size=100):

    opt = optimizer.SGD(momentum=0.8, weight_decay=0.01)

    agent.push(MsgType.kStatus, 'Downlaoding data...')
    records, in_shape = get_data('http://comp.nus.edu.sg/~dbsystem/singa/assets/file/feature.csv')  # PUT THE DATA on/to dbsystem
    agent.push(MsgType.kStatus, 'Finish downloading data')
    tx = tensor.Tensor((batch_size, in_shape[0], in_shape[1], in_shape[2]), dev)
    ty = tensor.Tensor((batch_size, ), dev, core_pb2.kInt)
    num_train_batch = len(records) / batch_size
#    num_test_batch = test_x.shape[0] / (batch_size)
    idx = np.arange(len(records), dtype=np.int32)

    net = model.create_net(in_shape, use_cpu)
    net.to_device(dev)

    for epoch in range(max_epoch):
        if handle_cmd(agent):
            break
        np.random.shuffle(idx)
        print 'Epoch %d' % epoch
        
        '''
        loss, acc = 0.0, 0.0
        for b in range(num_test_batch):
            x, y = flat_data(records[b * batch_size:(b + 1) * batch_size])
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a
        print 'testing loss = %f, accuracy = %f' % (loss / num_test_batch,
                                                    acc / num_test_batch)
        # put test status info into a shared queue
        info = dict(
            phase='test',
            step = epoch,
            accuracy = acc / num_test_batch,
            loss = loss / num_test_batch,
            timestamp = time.time())
        agent.push(MsgType.kInfoMetric, info)
        '''

        loss, acc = 0.0, 0.0
        for b in range(num_train_batch):
            x, y = flat_data(records, idx[b * batch_size:(b + 1) * batch_size],
                             in_shape[1], in_shape[2])
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_specs(),
                                 net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s.name))
            info = 'training loss = %f, training accuracy = %f' % (l, a)
            utils.update_progress(b * 1.0 / num_train_batch, info)
        # put training status info into a shared queue
        info = dict(phase='train', step=epoch,
                    accuracy=acc/num_train_batch,
                    loss=loss/num_train_batch,
                    timestamp=time.time())
        agent.push(MsgType.kInfoMetric, info)
        info = 'training loss = %f, training accuracy = %f' \
            % (loss / num_train_batch, acc / num_train_batch)
        print info

        if epoch > 0 and epoch % 30 == 0:
            net.save('parameter_%d' % epoch)
    net.save('parameter_last')


if __name__ == '__main__':
    main()
