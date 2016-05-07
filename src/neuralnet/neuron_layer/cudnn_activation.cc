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

#include "singa/neuralnet/neuron_layer.h"
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;

namespace singa {

void CudnnActivationLayer::InitCudnn() {
  CudnnBase::InitCudnn();

  // TODO(wangwei) make the mode case insensitive
  if (layer_conf_.activation_conf().type() == SIGMOID)
    mode_ = CUDNN_ACTIVATION_SIGMOID;
  else if (layer_conf_.activation_conf().type() == TANH)
    mode_ = CUDNN_ACTIVATION_TANH;
  else if (layer_conf_.activation_conf().type() == RELU)
    mode_ = CUDNN_ACTIVATION_RELU;
  else
    LOG(FATAL) << "Unkown activation: " << layer_conf_.activation_conf().type();

  const auto& shape = data_.shape();
  CHECK_GT(shape.size(), 0);
  const int nbdim = 4;
  // size of each dimension
  // int* sdim = new int[shape.size()];
  // int* stride = new int[shape.size()];
  // stride[shape.size() -1] = 1;
  int* sdim = new int[nbdim];
  int* stride = new int[nbdim];
  int i = shape.size() - 1;
  sdim[i] = shape[i];
  stride[i] = 1;
  for (--i; i >= 0; i--) {
    sdim[i] = shape[i];
    stride[i] = shape[i + 1] * stride[i + 1];
  }
  // LOG(ERROR) << "src_desc_: " << src_desc_;
  // LOG(ERROR) << "CUDNN_DATA_FLOAT: " << CUDNN_DATA_FLOAT;
  for (i = shape.size(); i < nbdim; i++){
    sdim[i] = 1;
    stride[i] = 1;
  }
  
  // LOG(ERROR) << "shape.size(): " << shape.size();
  // LOG(ERROR) << "sdim: " << sdim[0] << " " << sdim[1] << " " << sdim[2] << " " << sdim[3];
  // LOG(ERROR) << "stride: " << stride[0] << " " << stride[1] << " " << stride[2] << " " << stride[3]; 
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(src_desc_,
        CUDNN_DATA_FLOAT,
        // shape.size(),
        nbdim,
        sdim,
        stride));
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(my_desc_,
        CUDNN_DATA_FLOAT,
        // shape.size(),
        nbdim,
        sdim,
        stride));
  delete[] sdim;
  delete[] stride;
  srand((unsigned)time(NULL));
  run_version_ = rand()%1000;
}

void CudnnActivationLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  if (!has_init_cudnn_)
    InitCudnn();
  float alpha = 1.0f, beta = 0.0f;
   
  //print feature map
  
  // LOG(ERROR) << "flag indicate" << (flag&flag);
  // LOG(INFO) << "layer name: " << this->name().c_str();
  // LOG(ERROR) << "layer name: " << this->name().c_str();
  // LOG(INFO) << "RELU source data norm: " << Asum(srclayers[0]->data(this)); 
  // LOG(INFO) << "RELU source data shape0: " << srclayers[0]->data(this).shape()[0];
  // LOG(INFO) << "RELU source data shape1: " << srclayers[0]->data(this).shape()[1];
  // LOG(INFO) << "RELU source data shape2: " << srclayers[0]->data(this).shape()[2];
  // LOG(INFO) << "RELU source data shape3: " << srclayers[0]->data(this).shape()[3];
  // LOG(INFO) << "RELU source data shape4: " << srclayers[0]->data(this).shape()[4];
  // 528 log, 544 line
  //check : featuremapptr + = height * width && j is reused?
  // LOG(ERROR) << "before printting feature map";
  if (strcmp((this->name()).c_str(), "relu1@00") == 0 && (flag&flag) == 36){
    int topk = 10;
    int batchsize = srclayers[0]->data(this).shape()[0];
    int filter_num = srclayers[0]->data(this).shape()[1];
    int height = srclayers[0]->data(this).shape()[2];
    int width = srclayers[0]->data(this).shape()[3];
    int count = srclayers[0]->data(this).count();
    LOG(ERROR) << "filter_num: " << filter_num;
    LOG(ERROR) << "height: " << height;
    LOG(ERROR) << "width: " << width;
    LOG(ERROR) << "beign printting feature map";
    const float* featuremapptr = srclayers[0]->data(this).cpu_data();
    ofstream featuremapout;
    ofstream featuremapshape;
    featuremapout.open("/data/zhaojing/feature-map/map/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);
    featuremapshape.open("/data/zhaojing/feature-map/shape/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);
    featuremapshape << batchsize << "," << filter_num << "," << height << "," << width << "," << count << "\n";
    for(int n = 0; n < (batchsize * filter_num); n++){
      vector<std::pair<float, int> > vec;
      for (int j = 0; j < (height * width); j++)
        vec.push_back(std::make_pair(featuremapptr[j], j));
      std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());
      for (int j = 0; j < topk; j++)
        featuremapout  << static_cast<int> (vec.at(j).second) << "," << static_cast<float> (vec.at(j).first) << "\n";
      featuremapptr += (height * width); //important!!!
    }
    featuremapout.close();
    featuremapshape.close();
  }

  // currently only consider single src layer
  CHECK_EQ(srclayers.size(), 1);
  CHECK_CUDNN(cudnnActivationForward(handle_,
        mode_,
        &alpha,
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        &beta,
        my_desc_,
        data_.mutable_gpu_data()));
}

void CudnnActivationLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnActivationBackward(handle_,
        mode_,
        &alpha,
        my_desc_,
        data_.gpu_data(),
        my_desc_,
        grad_.gpu_data(),
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        &beta,
        src_desc_,
        srclayers[0]->mutable_grad(this)->mutable_gpu_data()));
}
}   // namespace singa
