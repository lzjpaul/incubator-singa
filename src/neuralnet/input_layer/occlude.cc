/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/neuralnet/input_layer.h"
namespace singa {

using std::string;
using std::vector;

void OccludeInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  SingleLabelRecordLayer::Setup(conf, srclayers);
  encoded_ = conf.store_conf().encoded();
  test_sample_ = 0;
}

void OccludeInputLayer::LoadRecord(const string& backend,
    const string&path, Blob<float>* to) {
  io::Store* store = io::OpenStore(backend, path, io::kRead);
  string key, val;
  CHECK(store->Read(&key, &val));
  RecordProto image;
  image.ParseFromString(val);
  CHECK_EQ(to->count(), image.data_size());
  float* ptr = to->mutable_cpu_data();
  for (int i = 0; i< to->count(); i++)
    ptr[i] = image.data(i);
  delete store;
}

bool OccludeInputLayer::Parse(int k, int flag, const string& key,
    const string& value) {
  RecordProto image;
  image.ParseFromString(value);
  int size = data_.count() / batchsize_;
  // LOG(ERROR) << "size: " << size;
  // LOG(ERROR) << "key" << key;
  if (image.data_size() != 15324)
    LOG(ERROR) << "image.data_size(): " << image.data_size();
  if (image.data_size()) {
    if ( test_sample_ % 3000 == 0 )
      LOG(ERROR) << "begin step: " << (test_sample_)/batchsize_;
    CHECK_EQ(size, image.data_size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    for (int i = 0; i< size; i++)
      ptr[i] = image.data(i);
    /*begin occlude*/
    int height = 12;
    int width = 1277;
    int kernel_y = 3;
    int kernel_x = 80;
    int stride_y = 1; //because stride_y == 1, so can for j = 0; j < kernel_y
    int stride_x = 20;
    int sample_test_step = 30;
    int height_dim = (height - kernel_y) / stride_y + 1; // 10
    // LOG(ERROR) << "height_dim: " << height_dim;
    int width_dim = (width - kernel_x) / stride_x + 1; // 60
    // LOG(ERROR) << "width_dim: " << width_dim;
    // this is to calculate feature map, according to 18000 samples in total 600 steps
    int height_index = (test_sample_ / batchsize_ / sample_test_step)/width_dim;
    int width_index = (test_sample_ / batchsize_ / sample_test_step) - (width_dim * height_index);
    // calculate which part this piture should be occlude
    for (int j = 0; j < kernel_y; j++)
      for (int i = (height_index * stride_y + j) * width + width_index * stride_x; i < (height_index * stride_y + j) * width + width_index * stride_x + kernel_x; i++)
        ptr[i] = 0.0f;
    test_sample_ ++;
    /*end occlude*/
  } else if (image.pixel().size()) {
    CHECK_EQ(size, image.pixel().size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    string pixel = image.pixel();
    for (int i = 0; i < size; i++)
      ptr[i] =  static_cast<float>(static_cast<uint8_t>(pixel[i]));
  } else {
    LOG(ERROR) << "not pixel nor pixel";
  }
  if ((flag & kDeploy) == 0) {  // deploy mode does not have label
    aux_data_.at(k) = image.label();
  }
  return true;
}

}  // namespace singa
