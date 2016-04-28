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

/**
check:
group18dptr=group18_data_.mutable_cpu_data() + k * group18_dim_ * 12;
group1_data_.Reshape(vector<int>{batchsize, 1, 12, group1_dim_}); and previous k * group18_dim_ * 12 match?

**/
#include "singa/neuralnet/input_layer.h"
#include "mshadow/tensor.h"
namespace singa {

using std::string;
using std::vector;

void FloatRNNInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  SingleLabelRecordLayer::Setup(conf, srclayers);
  encoded_ = conf.store_conf().encoded();
  // group1_dim_ = conf.store_conf().group1_dim();
  LOG(ERROR) << "batchsize: " << batchsize_;
  unroll_len_ = conf.store_conf().unroll_len();
  case_feature_dim_ = conf.store_conf().case_feature_dim();
  LOG(ERROR) << "float RNN case_feature_dim: " << case_feature_dim_ << "unroll len: " << unroll_len_;
  datavec_.clear();
  // each unroll layer has a input blob
  for (int i = 0; i < unroll_len_; i++) {
    datavec_.push_back(new Blob<float>(batchsize_*case_feature_dim_));
  }
}

void FloatRNNInputLayer::LoadRecord(const string& backend,
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

bool FloatRNNInputLayer::Parse(int k, int flag, const string& key,
    const string& value) {
  RecordProto image;
  image.ParseFromString(value);
  int size = data_.count() / batchsize_;
  LOG(ERROR) << "parse batchsize number k:" << k;
  // LOG(ERROR) << "size: " << size;
  // LOG(ERROR) << "key" << key;
  if (image.data_size() != 15036)
    LOG(ERROR) << "image.data_size(): " << image.data_size();
  if (image.data_size()) {
    CHECK_EQ(size, image.data_size());
    float* ptr = data_.mutable_cpu_data() + k * size; //15036, 12 cases
    float* group1dptr;
    // if (group1_dim_ != 0){
    //  group1dptr=group1_data_.mutable_cpu_data() + k * group1_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    //}
    for (int i = 0; i < size; i++)
      ptr[i] = image.data(i);
    /*begin multi-dst*/
    //int height = 12;
    // int width = 1253;
    /*size = 15036 = 12 * 1253*/
    int i_dim = 0;
    int j_dim = 0;
    for (int i = 0; i < size; i++){
      i_dim = i / case_feature_dim_; //which case
      j_dim = i % case_feature_dim_; //which element of the case
      LOG(ERROR) << "i_dim: " << i_dim << " j_dim: " << j_dim;
      // !!!!! each datavector element is batchsize*case_feature_dim_
      float* case_ptr = datavec_[i_dim]->mutable_cpu_data();
      case_ptr[k * case_feature_dim_ + j_dim] = image.data(i);
    }
    // CHECK_EQ(group21_data_.mutable_cpu_data() + (k+1) * group21_dim_ * 12, group21dptr);
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
