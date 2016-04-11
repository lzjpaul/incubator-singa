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
/*1 600 2 n, n is varying...
  at the end of Setup, add concate_dim_ = 1 manualy*/
#include "singa/neuralnet/connection_layer.h"
#include "singa/utils/singleton.h"
#include "singa/utils/context.h"

namespace singa {

void ConcateLayer::Setup(const LayerProto& conf,
                         const vector<Layer*>& srclayers) {
  LOG(ERROR) << "concate layer setup begins" << "\n";
  CHECK_GT(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  vector<int> shape = srclayers[0]->data(this).shape();
  concate_dim_ = conf.concate_conf().concate_dim();
  num_concates_ = conf.concate_conf().num_concates();
  CHECK_GE(concate_dim_, 0);
  CHECK_LT(concate_dim_, shape.size());
  CHECK_EQ(num_concates_, srclayers.size());
  LOG(ERROR) << "src layer0 concate shape 0: " << srclayers[0]->data(this).shape()[0] << "\n";
  LOG(ERROR) << "src layer0 concate shape 1: " << srclayers[0]->data(this).shape()[1] << "\n";
  LOG(ERROR) << "src layer0 concate shape 2: " << srclayers[0]->data(this).shape()[2] << "\n";
  LOG(ERROR) << "src layer0 concate shape 3: " << srclayers[0]->data(this).shape()[3] << "\n";
  for (size_t i = 1; i < srclayers.size(); i++) {
    const vector<int>& src_shape = srclayers[i]->data(this).shape();
    for (size_t j = 0; j < shape.size(); j++)
      if (static_cast<int>(j) == concate_dim_)
        shape[j] += src_shape[j];
      else{
        // LOG(ERROR) << " concate i: " <<i;
        // LOG(ERROR) << " concate j: " <<j;
        CHECK_EQ(shape[j], src_shape[j]);
      }
  }
  LOG(ERROR) << "concate shape 0: " << shape[0] << "\n";
  LOG(ERROR) << "concate shape 1: " << shape[1] << "\n";
  LOG(ERROR) << "concate shape 2: " << shape[2] << "\n";
  LOG(ERROR) << "concate shape 3 used for fan_out: " << shape[3] << "\n";
  data_.Reshape(shape);
  grad_.Reshape(shape);
  concate_dim_ = 1;
  if (concate_dim_ == 1)
    LOG(ERROR) << "Attention, the concate_dim_ is manually set to 1";
}

void ConcateLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  // LOG(ERROR) << "concate layer compute feature begins concate dim: " << concate_dim_;
  CHECK_GT(srclayers.size(), 1);
  CHECK_EQ(num_concates_, srclayers.size());
  // calculate step for each memcpy
  int step[srclayers.size()];
  for (unsigned i = 0; i < srclayers.size(); i++)
    step[i] = srclayers[i]->data(this).shape()[concate_dim_];
  for (unsigned j = 0; j < srclayers.size(); j++)
    for (unsigned i = concate_dim_ + 1; i < data_.shape().size(); ++i)
      step[j] *= srclayers[j]->data(this).shape()[i];
  int srclayer_offset[srclayers.size()];
  for (unsigned i = 0; i < srclayers.size(); i++)
    srclayer_offset[i] = 0;
  int concate_offset = 0;
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  // LOG(ERROR) << "concate layer compute feature before while" << "\n";
  while (concate_offset < data_.count()) {
    for (size_t i = 0; i < srclayers.size(); ++i) {
      // LOG(ERROR) << "i: " << i;
      if (device == -1) {
        const float* src = srclayers[i]->data(this).cpu_data()
          + srclayer_offset[i];
        float* dst = data_.mutable_cpu_data() + concate_offset;
        memcpy(dst, src, step[i] * sizeof(float));
      } else {
#ifdef USE_GPU
        const float* src = srclayers[i]->data(this).gpu_data()
          + srclayer_offset[i];
        float* dst = data_.mutable_gpu_data() + concate_offset;
        cudaMemcpy(dst, src, step[i] * sizeof(float), cudaMemcpyDefault);
#else
        LOG(FATAL) << "GPU is supported";
#endif
      }
      concate_offset += step[i];
    }
    // LOG(ERROR) << "finish for";
    for (unsigned i = 0; i < srclayers.size(); i++)
      srclayer_offset[i] += step[i];
    // LOG(ERROR) << "after srclayer_offset +";
  }
  CHECK_EQ(srclayer_offset[0], srclayers[0]->data(this).count());
  CHECK_EQ(srclayer_offset[1], srclayers[1]->data(this).count());
  if (srclayers.size() > 3){
    CHECK_EQ(srclayer_offset[5], srclayers[5]->data(this).count());
    CHECK_EQ(srclayer_offset[6], srclayers[6]->data(this).count());
  }
  CHECK_EQ(concate_offset, data_.count());
  // LOG(ERROR) << "concate computefeature finish";
}

void ConcateLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  CHECK_GT(srclayers.size(), 1);
  CHECK_EQ(num_concates_, srclayers.size());
  // calculate step for each memcpy
  int step[srclayers.size()];
  for (unsigned i = 0; i < srclayers.size(); i++)
    step[i] = srclayers[i]->grad(this).shape()[concate_dim_];
  for (unsigned j = 0; j < srclayers.size(); j++)
    for (unsigned i = concate_dim_ + 1; i < grad_.shape().size(); ++i)
      step[j] *= srclayers[j]->grad(this).shape()[i];
  int srclayer_offset[srclayers.size()];
  for (unsigned i = 0; i < srclayers.size(); i++)
    srclayer_offset[i] = 0;
  int concate_offset = 0;
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  while (concate_offset < grad_.count()) {
    for (size_t i = 0; i < srclayers.size(); ++i) {
      if (device == -1) {
        const float* src = grad_.cpu_data() + concate_offset;
        float* dst = srclayers[i]->mutable_grad(this)->mutable_cpu_data()
          + srclayer_offset[i];
        memcpy(dst, src, step[i] * sizeof(float));
      } else {
#ifdef USE_GPU
        const float* src = grad_.gpu_data() + concate_offset;
        float* dst = srclayers[i]->mutable_grad(this)->mutable_gpu_data()
          + srclayer_offset[i];
        cudaMemcpy(dst, src, step[i] * sizeof(float), cudaMemcpyDefault);
#else
        LOG(FATAL) << "GPU is supported";
#endif
      }
      concate_offset += step[i];
    }
    for (unsigned i = 0; i < srclayers.size(); i++)
      srclayer_offset[i] += step[i];
  }
}

}  // namespace singa
