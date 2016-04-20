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
#include "singa/utils/math_blob.h"
#include <time.h>
#include <fstream>
#include <iostream>
using namespace std;

namespace singa {

CudnnConvLayer::~CudnnConvLayer() {
  if (has_init_cudnn_) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc_));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }
}

void CudnnConvLayer::InitCudnn() {
  CudnnBase::InitCudnn();
  // convert MB to bytes
  workspace_byte_limit_
    = layer_conf_.convolution_conf().workspace_byte_limit() << 20;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc_));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));

  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
        pad_y_,
        pad_x_,
        stride_y_,
        stride_x_,
        1,
        1,
        CUDNN_CROSS_CORRELATION));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
        CUDNN_DATA_FLOAT,
        num_filters_,
        channels_,
        kernel_y_,
        kernel_x_));
  if (bias_) {
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          1,
          num_filters_,
          1,
          1));
  }
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(src_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchsize_,
        channels_,
        height_,
        width_));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(my_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchsize_,
        num_filters_,
        conv_height_,
        conv_width_));

  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(handle_,
        src_desc_,
        filter_desc_,
        conv_desc_,
        my_desc_,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        workspace_byte_limit_,
        &fp_alg_));

  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
        src_desc_,
        my_desc_,
        conv_desc_,
        filter_desc_,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        workspace_byte_limit_,
        &bp_filter_alg_));
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(handle_,
        filter_desc_,
        my_desc_,
        conv_desc_,
        src_desc_,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        workspace_byte_limit_,
        &bp_data_alg_));

  size_t fp_byte, bp_data_byte, bp_filter_byte;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle_,
        src_desc_,
        filter_desc_,
        conv_desc_,
        my_desc_,
        fp_alg_,
        &fp_byte));
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
        filter_desc_,
        my_desc_,
        conv_desc_,
        src_desc_,
        bp_data_alg_,
        &bp_data_byte));
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
        src_desc_,
        my_desc_,
        conv_desc_,
        filter_desc_,
        bp_filter_alg_,
        &bp_filter_byte));
  workspace_count_ = std::max(std::max(fp_byte, bp_data_byte), bp_filter_byte)
    / sizeof(float) + 1;
}

void CudnnConvLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (!has_init_cudnn_)
    InitCudnn();
  float alpha = 1.f, beta = 0.f;

  // LOG(ERROR) << "source layer name: " << srclayers[0]->name().c_str();
  // LOG(ERROR) << "layer name: " << this->name().c_str();
  // LOG(ERROR) << "CONV source data norm: " << Asum(srclayers[0]->data(this));
  // LOG(ERROR) << "CONV source data shape0: " << srclayers[0]->data(this).shape()[0];
  // LOG(ERROR) << "CONV source data shape1: " << srclayers[0]->data(this).shape()[1];
  // LOG(ERROR) << "CONV source data shape2: " << srclayers[0]->data(this).shape()[2];
  // LOG(ERROR) << "CONV source data shape3: " << srclayers[0]->data(this).shape()[3];
  // LOG(ERROR) << "CONV source data shape4: " << srclayers[0]->data(this).shape()[4];
  LOG(ERROR) << "CONV weight shape0: " << weight_->data().shape()[0];
  LOG(ERROR) << "CONV weight shape1: " << weight_->data().shape()[1];
  LOG(ERROR) << "CONV weight shape2: " << weight_->data().shape()[2];
  
  int run_version = rand()%1000;
  if (strcmp((this->name()).c_str(), "conv1@00") == 0 && (flag&flag) == 36){
    int count = weight_->data().count();
    LOG(ERROR) << "weight count: " << count;
    LOG(ERROR) << "beign printting weight_matrix";
    const float* weightptr = weight_->data().cpu_data();
    ofstream weightout;
    weightout.open("/data/zhaojing/feature-map/filter-weight/version" + std::to_string(static_cast<int>(run_version)) + ".csv", ios::app);
    int j;
    for (j = 0; j < (count - 1); j++)
      weightout  << static_cast<float> (weightptr[j]) << ",";
    weightout  << static_cast<float> (weightptr[j]) << "\n";
    weightout.close();
  }
  /*begin check input data!!!!*/
  /*auto src = Tensor4(srclayers[0]->mutable_data(this));
  float* srcdptr = src.dptr;
  LOG(ERROR) << "begin print data outside loop";
  if (strcmp((this->name()).c_str(), "conv1@00") == 0){
    LOG(ERROR) << "begin print data";
    for (int i = 0; i < 48; i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<"ith: "<<(i+1)<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 1st round";
    for (int i = 15324; i < (15324+48); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<"ith: "<<(i+1)<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 2nd round";
    for (int i = 15324*2; i < (15324*2+48); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<"ith: "<<(i+1)<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 3rd round";
    for (int i = 15324*3; i < (15324*3+48); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<"ith: "<<(i+1)<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 4th round";
    for (int i = 15324*4; i < (15324*4+48); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<"ith: "<<(i+1)<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 5th round";
    LOG(ERROR) << "end print data";
  }*/
  /*end check input data!!!*/

  Blob<float> workspace(vector<int>{static_cast<int>(workspace_count_)});
  CHECK_CUDNN(cudnnConvolutionForward(handle_,
        &alpha,
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        filter_desc_,
        weight_->data().gpu_data(),
        conv_desc_,
        fp_alg_,
        workspace.mutable_gpu_data(),
        workspace_count_ * sizeof(float),
        &beta,
        my_desc_,
        data_.mutable_gpu_data()));
  if (bias_) {
    beta = 1.f;
    CHECK_CUDNN(cudnnAddTensor(handle_,
          CUDNN_ADD_SAME_C,
          &alpha,
          bias_desc_,
          bias_->data().gpu_data(),
          &beta,
          my_desc_,
          data_.mutable_gpu_data()));
  }
}

void
CudnnConvLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  float alpha = 1.f, beta = 0.f;
  Blob<float> workspace(vector<int>{static_cast<int>(workspace_count_)});
  // LOG(ERROR) << "backward bias";
  if (bias_) {
    CHECK_CUDNN(cudnnConvolutionBackwardBias(handle_,
          &alpha,
          my_desc_,
          grad_.gpu_data(),
          &beta,
          bias_desc_,
          bias_->mutable_grad()->mutable_gpu_data()));
  }
  // LOG(ERROR) << "backward w";
  CHECK_CUDNN(cudnnConvolutionBackwardFilter_v3(handle_,
        &alpha,
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        my_desc_,
        grad_.gpu_data(),
        conv_desc_,
        bp_filter_alg_,
        workspace.mutable_gpu_data(),
        workspace_count_ * sizeof(float),
        &beta,
        filter_desc_,
        weight_->mutable_grad()->mutable_gpu_data()));
  // LOG(ERROR) << "backward src";
  if (srclayers[0]->mutable_grad(this) != nullptr) {
    CHECK_CUDNN(cudnnConvolutionBackwardData_v3(handle_,
          &alpha,
          filter_desc_,
          weight_->data().gpu_data(),
          my_desc_,
          grad_.gpu_data(),
          conv_desc_,
          bp_data_alg_,
          workspace.mutable_gpu_data(),
          workspace_count_ * sizeof(float),
          &beta,
          src_desc_,
          srclayers[0]->mutable_grad(this)->mutable_gpu_data()));
  }
}
}  // namespace singa
